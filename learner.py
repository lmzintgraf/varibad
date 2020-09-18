"""
Base Learner, without Meta-Learning.
Can be used to train for good average performance, or for the oracle environment.
"""

import os
import time

import gym
import numpy as np
import torch

from algorithms.a2c import A2C
from algorithms.online_storage import OnlineStorage
from algorithms.ppo import PPO
from environments.parallel_envs import make_vec_envs
from models.policy import Policy
from utils import evaluation as utl_eval
from utils import helpers as utl
from utils.tb_logger import TBLogger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Learner:
    """
    Learner (no meta-learning), can be used to train avg/oracle/belief-oracle policies.
    """
    def __init__(self, args):

        self.args = args
        utl.seed(self.args.seed, self.args.deterministic_execution)

        # calculate number of updates and keep count of frames/iterations
        self.num_updates = int(args.num_frames) // args.policy_num_steps // args.num_processes
        self.frames = 0
        self.iter_idx = 0

        # initialise tensorboard logger
        self.logger = TBLogger(self.args, self.args.exp_label)

        # initialise environments
        self.envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                                  gamma=args.policy_gamma, device=device,
                                  episodes_per_task=self.args.max_rollouts_per_task,
                                  normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                                  )

        # calculate what the maximum length of the trajectories is
        args.max_trajectory_len = self.envs._max_episode_steps
        args.max_trajectory_len *= self.args.max_rollouts_per_task

        # get policy input dimensions
        self.args.state_dim = self.envs.observation_space.shape[0]
        self.args.task_dim = self.envs.task_dim
        self.args.belief_dim = self.envs.belief_dim
        self.args.num_states = self.envs.num_states
        # get policy output (action) dimensions
        self.args.action_space = self.envs.action_space
        if isinstance(self.envs.action_space, gym.spaces.discrete.Discrete):
            self.args.action_dim = 1
        else:
            self.args.action_dim = self.envs.action_space.shape[0]

        # initialise policy
        self.policy_storage = self.initialise_policy_storage()
        self.policy = self.initialise_policy()

    def initialise_policy_storage(self):
        return OnlineStorage(args=self.args,
                             num_steps=self.args.policy_num_steps,
                             num_processes=self.args.num_processes,
                             state_dim=self.args.state_dim,
                             latent_dim=0,  # use metalearner.py if you want to use the VAE
                             belief_dim=self.args.belief_dim,
                             task_dim=self.args.task_dim,
                             action_space=self.args.action_space,
                             hidden_size=0,
                             normalise_rewards=self.args.norm_rew_for_policy,
                             )

    def initialise_policy(self):

        if hasattr(self.envs.action_space, 'low'):
            action_low = self.envs.action_space.low
            action_high = self.envs.action_space.high
        else:
            action_low = action_high = None

        # initialise policy network
        policy_net = Policy(
            args=self.args,
            #
            pass_state_to_policy=self.args.pass_state_to_policy,
            pass_latent_to_policy=False,  # use metalearner.py if you want to use the VAE
            pass_belief_to_policy=self.args.pass_belief_to_policy,
            pass_task_to_policy=self.args.pass_task_to_policy,
            dim_state=self.args.state_dim,
            dim_latent=0,
            dim_belief=self.args.belief_dim,
            dim_task=self.args.task_dim,
            #
            hidden_layers=self.args.policy_layers,
            activation_function=self.args.policy_activation_function,
            policy_initialisation=self.args.policy_initialisation,
            #
            action_space=self.envs.action_space,
            init_std=self.args.policy_init_std,
            norm_actions_of_policy=self.args.norm_actions_of_policy,
            action_low=action_low,
            action_high=action_high,
        ).to(device)

        # initialise policy trainer
        if self.args.policy == 'a2c':
            policy = A2C(
                self.args,
                policy_net,
                self.args.policy_value_loss_coef,
                self.args.policy_entropy_coef,
                policy_optimiser=self.args.policy_optimiser,
                policy_anneal_lr=self.args.policy_anneal_lr,
                train_steps=self.num_updates,
                lr=self.args.lr_policy,
                eps=self.args.policy_eps,
            )
        elif self.args.policy == 'ppo':
            policy = PPO(
                self.args,
                policy_net,
                self.args.policy_value_loss_coef,
                self.args.policy_entropy_coef,
                policy_optimiser=self.args.policy_optimiser,
                policy_anneal_lr=self.args.policy_anneal_lr,
                train_steps=self.num_updates,
                lr=self.args.lr_policy,
                eps=self.args.policy_eps,
                ppo_epoch=self.args.ppo_num_epochs,
                num_mini_batch=self.args.ppo_num_minibatch,
                use_huber_loss=self.args.ppo_use_huberloss,
                use_clipped_value_loss=self.args.ppo_use_clipped_value_loss,
                clip_param=self.args.ppo_clip_param,
            )
        else:
            raise NotImplementedError

        return policy

    def train(self):
        """ Main training loop """
        start_time = time.time()

        # reset environments
        state, belief, task = utl.reset_env(self.envs, self.args)

        # insert initial observation / embeddings to rollout storage
        self.policy_storage.prev_state[0].copy_(state)

        # log once before training
        with torch.no_grad():
            self.log(None, None, start_time)

        for self.iter_idx in range(self.num_updates):

            # rollout policies for a few steps
            for step in range(self.args.policy_num_steps):

                # sample actions from policy
                with torch.no_grad():
                    value, action, action_log_prob = utl.select_action(
                        args=self.args,
                        policy=self.policy,
                        state=state,
                        belief=belief,
                        task=task,
                        deterministic=False)

                # observe reward and next obs
                [state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(self.envs, action, self.args)

                # create mask for episode ends
                masks_done = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
                # bad_mask is true if episode ended because time limit was reached
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos]).to(device)

                # reset environments that are done
                done_indices = np.argwhere(done.flatten()).flatten()
                if len(done_indices) > 0:
                    state, belief, task = utl.reset_env(self.envs, self.args,
                                                        indices=done_indices, state=state)

                # add experience to policy buffer
                self.policy_storage.insert(
                    state=state,
                    belief=belief,
                    task=task,
                    actions=action,
                    action_log_probs=action_log_prob,
                    rewards_raw=rew_raw,
                    rewards_normalised=rew_normalised,
                    value_preds=value,
                    masks=masks_done,
                    bad_masks=bad_masks,
                    done=torch.from_numpy(np.array(done, dtype=float)).unsqueeze(1),
                )

                self.frames += self.args.num_processes

            # --- UPDATE ---

            train_stats = self.update(state=state, belief=belief, task=task)

            # log
            run_stats = [action, action_log_prob, value]
            if train_stats is not None:
                with torch.no_grad():
                    self.log(run_stats, train_stats, start_time)

            # clean up after update
            self.policy_storage.after_update()

    def get_value(self, state, belief, task):
        return self.policy.actor_critic.get_value(state=state, belief=belief, task=task, latent=None).detach()

    def update(self, state, belief, task):
        """
        Meta-update.
        Here the policy is updated for good average performance across tasks.
        :return:    policy_train_stats which are: value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_epoch
        """
        # bootstrap next value prediction
        with torch.no_grad():
            next_value = self.get_value(state=state, belief=belief, task=task)

        # compute returns for current rollouts
        self.policy_storage.compute_returns(next_value, self.args.policy_use_gae, self.args.policy_gamma,
                                            self.args.policy_tau,
                                            use_proper_time_limits=self.args.use_proper_time_limits)

        policy_train_stats = self.policy.update(policy_storage=self.policy_storage)

        return policy_train_stats, None

    def log(self, run_stats, train_stats, start):
        """
        Evaluate policy, save model, write to tensorboard logger.
        """

        # --- visualise behaviour of policy ---

        if self.iter_idx % self.args.vis_interval == 0:

            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None
            utl_eval.visualise_behaviour(args=self.args,
                                         policy=self.policy,
                                         image_folder=self.logger.full_output_folder,
                                         iter_idx=self.iter_idx,
                                         ret_rms=ret_rms,
                                         )

        # --- evaluate policy ----

        if self.iter_idx % self.args.eval_interval == 0:

            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None

            returns_per_episode = utl_eval.evaluate(args=self.args,
                                                    policy=self.policy,
                                                    ret_rms=ret_rms,
                                                    iter_idx=self.iter_idx
                                                    )

            # log the average return across tasks (=processes)
            returns_avg = returns_per_episode.mean(dim=0)
            returns_std = returns_per_episode.std(dim=0)
            for k in range(len(returns_avg)):
                self.logger.add('return_avg_per_iter/episode_{}'.format(k + 1), returns_avg[k], self.iter_idx)
                self.logger.add('return_avg_per_frame/episode_{}'.format(k + 1), returns_avg[k], self.frames)
                self.logger.add('return_std_per_iter/episode_{}'.format(k + 1), returns_std[k], self.iter_idx)
                self.logger.add('return_std_per_frame/episode_{}'.format(k + 1), returns_std[k], self.frames)

            print("Updates {}, num timesteps {}, FPS {} \n Mean return (train): {:.5f} \n".
                  format(self.iter_idx, self.frames, int(self.frames / (time.time() - start)),
                         returns_avg[-1].item()))

        # save model
        if self.iter_idx % self.args.save_interval == 0:
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            idx_labels = ['']
            if self.args.save_intermediate_models:
                idx_labels.append(int(self.iter_idx))

            for idx_label in idx_labels:

                torch.save(self.policy.actor_critic, os.path.join(save_path, f"policy{idx_label}.pt"))

                # save normalisation params of envs
                if self.args.norm_rew_for_policy:
                    rew_rms = self.envs.venv.ret_rms
                    utl.save_obj(rew_rms, save_path, f"env_rew_rms{idx_label}")
                # TODO: grab from policy and save?
                # if self.args.norm_obs_for_policy:
                #     obs_rms = self.envs.venv.obs_rms
                #     utl.save_obj(obs_rms, save_path, f"env_obs_rms{idx_label}")

        # --- log some other things ---

        if (self.iter_idx % self.args.log_interval == 0) and (train_stats is not None):

            train_stats, _ = train_stats

            self.logger.add('policy_losses/value_loss', train_stats[0], self.iter_idx)
            self.logger.add('policy_losses/action_loss', train_stats[1], self.iter_idx)
            self.logger.add('policy_losses/dist_entropy', train_stats[2], self.iter_idx)
            self.logger.add('policy_losses/sum', train_stats[3], self.iter_idx)

            # writer.add_scalar('policy/action', action.mean(), j)
            self.logger.add('policy/action', run_stats[0][0].float().mean(), self.iter_idx)
            if hasattr(self.policy.actor_critic, 'logstd'):
                self.logger.add('policy/action_logstd', self.policy.actor_critic.dist.logstd.mean(), self.iter_idx)
            self.logger.add('policy/action_logprob', run_stats[1].mean(), self.iter_idx)
            self.logger.add('policy/value', run_stats[2].mean(), self.iter_idx)

            param_list = list(self.policy.actor_critic.parameters())
            param_mean = np.mean([param_list[i].data.cpu().numpy().mean() for i in range(len(param_list))])
            param_grad_mean = np.mean([param_list[i].grad.cpu().numpy().mean() for i in range(len(param_list))])
            self.logger.add('weights/policy', param_mean, self.iter_idx)
            self.logger.add('weights/policy_std', param_list[0].data.cpu().mean(), self.iter_idx)
            self.logger.add('gradients/policy', param_grad_mean, self.iter_idx)
            self.logger.add('gradients/policy_std', param_list[0].grad.cpu().numpy().mean(), self.iter_idx)
