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
    Learner (no meta-learning), can be used to train Oracle policies.
    """

    def __init__(self, args):
        self.args = args

        # make sure everything has the same seed
        utl.seed(self.args.seed, self.args.deterministic_execution)

        # initialise tensorboard logger
        self.logger = TBLogger(self.args, self.args.exp_label)

        # initialise environments
        self.envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                                  gamma=args.policy_gamma, log_dir=args.agent_log_dir, device=device,
                                  allow_early_resets=False,
                                  episodes_per_task=self.args.max_rollouts_per_task,
                                  obs_rms=None, ret_rms=None,
                                  )

        # calculate what the maximum length of the trajectories is
        args.max_trajectory_len = self.envs._max_episode_steps
        args.max_trajectory_len *= self.args.max_rollouts_per_task

        # calculate number of meta updates
        self.args.num_updates = int(args.num_frames) // args.policy_num_steps // args.num_processes

        # get action / observation dimensions
        if isinstance(self.envs.action_space, gym.spaces.discrete.Discrete):
            self.args.action_dim = 1
        else:
            self.args.action_dim = self.envs.action_space.shape[0]
        self.args.obs_dim = self.envs.observation_space.shape[0]
        self.args.num_states = self.envs.num_states if str.startswith(self.args.env_name, 'Grid') else None
        self.args.act_space = self.envs.action_space

        self.initialise_policy()

        # count number of frames and updates
        self.frames = 0
        self.iter_idx = 0

    def initialise_policy(self):

        # variables for task encoder (used for oracle)
        state_dim = self.envs.observation_space.shape[0]

        # TODO: this isn't ideal, find a nicer way to get the task dimension!
        if 'BeliefOracle' in self.args.env_name:
            task_dim = gym.make(self.args.env_name).observation_space.shape[0] - \
                       gym.make(self.args.env_name.replace('BeliefOracle', '')).observation_space.shape[0]
            latent_dim = self.args.latent_dim
            state_embedding_size = self.args.state_embedding_size
            use_task_encoder = True
        elif 'Oracle' in self.args.env_name:
            task_dim = gym.make(self.args.env_name).observation_space.shape[0] - \
                       gym.make(self.args.env_name.replace('Oracle', '')).observation_space.shape[0]
            latent_dim = self.args.latent_dim
            state_embedding_size = self.args.state_embedding_size
            use_task_encoder = True
        else:
            task_dim = latent_dim = state_embedding_size = 0
            use_task_encoder = False

        # initialise rollout storage for the policy
        self.policy_storage = OnlineStorage(self.args,
                                            self.args.policy_num_steps,
                                            self.args.num_processes,
                                            self.args.obs_dim,
                                            self.args.act_space,
                                            hidden_size=0,
                                            latent_dim=self.args.latent_dim,
                                            normalise_observations=self.args.norm_obs_for_policy,
                                            normalise_rewards=self.args.norm_rew_for_policy,
                                            )

        if hasattr(self.envs.action_space, 'low'):
            action_low = self.envs.action_space.low
            action_high = self.envs.action_space.high
        else:
            action_low = action_high = None

        # initialise policy network
        policy_net = Policy(
            # general
            state_dim=int(self.args.condition_policy_on_state) * state_dim,
            action_space=self.envs.action_space,
            init_std=self.args.policy_init_std,
            hidden_layers=self.args.policy_layers,
            activation_function=self.args.policy_activation_function,
            use_task_encoder=use_task_encoder,
            # task encoding things (for oracle)
            task_dim=task_dim,
            latent_dim=latent_dim,
            state_embed_dim=state_embedding_size,
            #
            normalise_actions=self.args.normalise_actions,
            action_low=action_low,
            action_high=action_high,
        ).to(device)

        # initialise policy
        if self.args.policy == 'a2c':
            # initialise policy trainer (A2C)
            self.policy = A2C(
                policy_net,
                self.args.policy_value_loss_coef,
                self.args.policy_entropy_coef,
                lr=self.args.lr_policy,
                eps=self.args.policy_eps,
                alpha=self.args.a2c_alpha,
            )
        elif self.args.policy == 'ppo':
            # initialise policy network
            self.policy = PPO(
                policy_net,
                self.args.policy_value_loss_coef,
                self.args.policy_entropy_coef,
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

    def train(self):
        """
        Given some stream of environments and a logger (tensorboard),
        (meta-)trains the policy.
        """

        start_time = time.time()

        # reset environments
        (prev_obs_raw, prev_obs_normalised) = self.envs.reset()
        prev_obs_raw = prev_obs_raw.to(device)
        prev_obs_normalised = prev_obs_normalised.to(device)

        # insert initial observation / embeddings to rollout storage
        self.policy_storage.prev_obs_raw[0].copy_(prev_obs_raw)
        self.policy_storage.prev_obs_normalised[0].copy_(prev_obs_normalised)
        self.policy_storage.to(device)

        for self.iter_idx in range(self.args.num_updates):

            # check if we flushed the policy storage
            assert len(self.policy_storage.latent_mean) == 0

            # rollouts policies for a few steps
            for step in range(self.args.policy_num_steps):

                # sample actions from policy
                with torch.no_grad():
                    value, action, action_log_prob = utl.select_action(
                        policy=self.policy,
                        args=self.args,
                        obs=prev_obs_normalised if self.args.norm_obs_for_policy else prev_obs_raw,
                        deterministic=False)

                # observe reward and next obs
                (next_obs_raw, next_obs_normalised), (rew_raw, rew_normalised), done, infos = utl.env_step(self.envs,
                                                                                                           action)
                action = action.float()

                # create mask for episode ends
                masks_done = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
                # bad_mask is true if episode ended because time limit was reached
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos]).to(device)

                # add the obs before reset to the policy storage
                self.policy_storage.next_obs_raw[step] = next_obs_raw.clone()
                self.policy_storage.next_obs_normalised[step] = next_obs_normalised.clone()

                # reset environments that are done
                done_indices = np.argwhere(done.flatten()).flatten()
                if len(done_indices) == self.args.num_processes:
                    [next_obs_raw, next_obs_normalised] = self.envs.reset()
                    if not self.args.sample_embeddings:
                        latent_sample = latent_sample
                else:
                    for i in done_indices:
                        [next_obs_raw[i], next_obs_normalised[i]] = self.envs.reset(index=i)
                        if not self.args.sample_embeddings:
                            latent_sample[i] = latent_sample[i]

                # add experience to policy buffer
                self.policy_storage.insert(
                    obs_raw=next_obs_raw.clone(),
                    obs_normalised=next_obs_normalised.clone(),
                    actions=action.clone(),
                    action_log_probs=action_log_prob.clone(),
                    rewards_raw=rew_raw.clone(),
                    rewards_normalised=rew_normalised.clone(),
                    value_preds=value.clone(),
                    masks=masks_done.clone(),
                    bad_masks=bad_masks.clone(),
                    done=torch.from_numpy(np.array(done, dtype=float)).unsqueeze(1).clone(),
                )

                prev_obs_normalised = next_obs_normalised
                prev_obs_raw = next_obs_raw

                self.frames += self.args.num_processes

            # --- UPDATE ---

            train_stats = self.update(prev_obs_normalised if self.args.norm_obs_for_policy else prev_obs_raw)

            # log
            run_stats = [action, action_log_prob, value]
            if train_stats is not None:
                self.log(run_stats, train_stats, start_time)

            # clean up after update
            self.policy_storage.after_update()

    def get_value(self, obs):
        obs = utl.get_augmented_obs(args=self.args, obs=obs)
        return self.policy.actor_critic.get_value(obs).detach()

    def update(self, obs):
        """
        Meta-update.
        Here the policy is updated for good average performance across tasks.
        :return:    policy_train_stats which are: value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_epoch
        """
        # bootstrap next value prediction
        with torch.no_grad():
            next_value = self.get_value(obs)

        # compute returns for current rollouts
        self.policy_storage.compute_returns(next_value, self.args.policy_use_gae, self.args.policy_gamma,
                                            self.args.policy_tau,
                                            use_proper_time_limits=self.args.use_proper_time_limits)

        policy_train_stats = self.policy.update(args=self.args, policy_storage=self.policy_storage)

        return policy_train_stats, None

    def log(self, run_stats, train_stats, start):
        """
        Evaluate policy, save model, write to tensorboard logger.
        """
        train_stats, meta_train_stats = train_stats

        # --- visualise behaviour of policy ---

        if self.iter_idx % self.args.vis_interval == 0:
            obs_rms = self.envs.venv.obs_rms if self.args.norm_obs_for_policy else None
            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None

            utl_eval.visualise_behaviour(args=self.args,
                                         policy=self.policy,
                                         image_folder=self.logger.full_output_folder,
                                         iter_idx=self.iter_idx,
                                         obs_rms=obs_rms,
                                         ret_rms=ret_rms,
                                         )

        # --- evaluate policy ----

        if self.iter_idx % self.args.eval_interval == 0:

            obs_rms = self.envs.venv.obs_rms if self.args.norm_obs_for_policy else None
            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None

            returns_per_episode = utl_eval.evaluate(args=self.args,
                                                    policy=self.policy,
                                                    obs_rms=obs_rms,
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
            torch.save(self.policy.actor_critic, os.path.join(save_path, "policy{0}.pt".format(self.iter_idx)))

            # save normalisation params of envs
            if self.args.norm_rew_for_policy:
                # save rolling mean and std
                rew_rms = self.envs.venv.ret_rms
                utl.save_obj(rew_rms, save_path, "env_rew_rms{0}.pkl".format(self.iter_idx))
            if self.args.norm_obs_for_policy:
                obs_rms = self.envs.venv.obs_rms
                utl.save_obj(obs_rms, save_path, "env_obs_rms{0}.pkl".format(self.iter_idx))

        # --- log some other things ---

        if self.iter_idx % self.args.log_interval == 0:
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
            param_mean = np.mean([param_list[i].data.mean() for i in range(len(param_list))])
            param_grad_mean = np.mean([param_list[i].grad.mean() for i in range(len(param_list))])
            self.logger.add('weights/policy', param_mean, self.iter_idx)
            self.logger.add('weights/policy_std', param_list[0].data.mean(), self.iter_idx)
            self.logger.add('gradients/policy', param_grad_mean, self.iter_idx)
            self.logger.add('gradients/policy_std', param_list[0].grad.mean(), self.iter_idx)
