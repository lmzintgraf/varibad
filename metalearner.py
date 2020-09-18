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
from vae import VaribadVAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MetaLearner:
    """
    Meta-Learner class with the main training loop for variBAD.
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
        self.args.max_trajectory_len = self.envs._max_episode_steps
        self.args.max_trajectory_len *= self.args.max_rollouts_per_task

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

        # initialise VAE and policy
        self.vae = VaribadVAE(self.args, self.logger, lambda: self.iter_idx)
        self.policy_storage = self.initialise_policy_storage()
        self.policy = self.initialise_policy()

    def initialise_policy_storage(self):
        return OnlineStorage(args=self.args,
                             num_steps=self.args.policy_num_steps,
                             num_processes=self.args.num_processes,
                             state_dim=self.args.state_dim,
                             latent_dim=self.args.latent_dim,
                             belief_dim=self.args.belief_dim,
                             task_dim=self.args.task_dim,
                             action_space=self.args.action_space,
                             hidden_size=self.args.encoder_gru_hidden_size,
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
            pass_latent_to_policy=self.args.pass_latent_to_policy,
            pass_belief_to_policy=self.args.pass_belief_to_policy,
            pass_task_to_policy=self.args.pass_task_to_policy,
            dim_state=self.args.state_dim,
            dim_latent=self.args.latent_dim * 2,
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
                optimiser_vae=self.vae.optimiser_vae,
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
                optimiser_vae=self.vae.optimiser_vae,
            )
        else:
            raise NotImplementedError

        return policy

    def train(self):
        """ Main Meta-Training loop """
        start_time = time.time()

        # reset environments
        prev_state, belief, task = utl.reset_env(self.envs, self.args)

        # insert initial observation / embeddings to rollout storage
        self.policy_storage.prev_state[0].copy_(prev_state)

        # log once before training
        with torch.no_grad():
            self.log(None, None, start_time)

        vae_is_pretrained = False
        for self.iter_idx in range(self.num_updates):

            # First, re-compute the hidden states given the current rollouts (since the VAE might've changed)
            with torch.no_grad():
                latent_sample, latent_mean, latent_logvar, hidden_state = self.encode_running_trajectory()

            # add this initial hidden state to the policy storage
            assert len(self.policy_storage.latent_mean) == 0  # make sure we emptied buffers
            self.policy_storage.hidden_states[0].copy_(hidden_state)
            self.policy_storage.latent_samples.append(latent_sample.clone())
            self.policy_storage.latent_mean.append(latent_mean.clone())
            self.policy_storage.latent_logvar.append(latent_logvar.clone())

            # rollout policies for a few steps
            for step in range(self.args.policy_num_steps):

                # sample actions from policy
                with torch.no_grad():
                    value, action, action_log_prob = utl.select_action(
                        args=self.args,
                        policy=self.policy,
                        state=prev_state,
                        belief=belief,
                        task=task,
                        deterministic=False,
                        latent_sample=latent_sample,
                        latent_mean=latent_mean,
                        latent_logvar=latent_logvar,
                    )

                # take step in the environment
                [next_state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(self.envs, action, self.args)

                done = torch.from_numpy(np.array(done, dtype=int)).to(device).float().view((-1, 1))
                # create mask for episode ends
                masks_done = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
                # bad_mask is true if episode ended because time limit was reached
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos]).to(device)

                with torch.no_grad():
                    # compute next embedding (for next loop and/or value prediction bootstrap)
                    latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(encoder=self.vae.encoder,
                                                                                                  next_obs=next_state,
                                                                                                  action=action,
                                                                                                  reward=rew_raw,
                                                                                                  done=done,
                                                                                                  hidden_state=hidden_state)

                # before resetting, update the embedding and add to vae buffer
                # (last state might include useful task info)
                if not (self.args.disable_decoder and self.args.disable_stochasticity_in_latent):
                    self.vae.rollout_storage.insert(prev_state.clone(),
                                                    action.detach().clone(),
                                                    next_state.clone(),
                                                    rew_raw.clone(),
                                                    done.clone(),
                                                    task.clone() if task is not None else None)

                if self.args.rlloss_through_encoder:
                    # add the obs before reset to the policy storage
                    # (only used to recompute embeddings if rlloss is backpropagated through encoder)
                    self.policy_storage.next_state[step] = next_state.clone()

                # reset environments that are done
                done_indices = np.argwhere(done.cpu().flatten()).flatten()
                if len(done_indices) > 0:
                    next_state, belief, task = utl.reset_env(self.envs, self.args,
                                                             indices=done_indices, state=next_state)

                # TODO: deal with resampling for posterior sampling algorithm
                #     latent_sample = latent_sample
                #     latent_sample[i] = latent_sample[i]

                # add experience to policy buffer
                self.policy_storage.insert(
                    state=next_state,
                    belief=belief,
                    task=task,
                    actions=action,
                    action_log_probs=action_log_prob,
                    rewards_raw=rew_raw,
                    rewards_normalised=rew_normalised,
                    value_preds=value,
                    masks=masks_done,
                    bad_masks=bad_masks,
                    done=done,
                    hidden_states=hidden_state.squeeze(0),
                    latent_sample=latent_sample,
                    latent_mean=latent_mean,
                    latent_logvar=latent_logvar,
                )

                prev_state = next_state

                self.frames += self.args.num_processes

            # --- UPDATE ---

            if self.args.precollect_len <= self.frames:

                # check if we are pre-training the VAE
                if self.args.pretrain_len > 0 and not vae_is_pretrained:
                    for _ in range(self.args.pretrain_len):
                        self.vae.compute_vae_loss(update=True)
                    vae_is_pretrained = True

                # otherwise do the normal update (policy + vae)
                else:

                    train_stats = self.update(state=prev_state,
                                              belief=belief,
                                              task=task,
                                              latent_sample=latent_sample,
                                              latent_mean=latent_mean,
                                              latent_logvar=latent_logvar)

                    # log
                    run_stats = [action, action_log_prob, value]
                    with torch.no_grad():
                        self.log(run_stats, train_stats, start_time)

            # clean up after update
            self.policy_storage.after_update()

    def encode_running_trajectory(self):
        """
        (Re-)Encodes (for each process) the entire current trajectory.
        Returns sample/mean/logvar and hidden state (if applicable) for the current timestep.
        :return:
        """

        # for each process, get the current batch (zero-padded obs/act/rew + length indicators)
        prev_obs, next_obs, act, rew, lens = self.vae.rollout_storage.get_running_batch()

        # get embedding - will return (1+sequence_len) * batch * input_size -- includes the prior!
        all_latent_samples, all_latent_means, all_latent_logvars, all_hidden_states = self.vae.encoder(actions=act,
                                                                                                       states=next_obs,
                                                                                                       rewards=rew,
                                                                                                       hidden_state=None,
                                                                                                       return_prior=True)

        # get the embedding / hidden state of the current time step (need to do this since we zero-padded)
        latent_sample = (torch.stack([all_latent_samples[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_mean = (torch.stack([all_latent_means[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_logvar = (torch.stack([all_latent_logvars[lens[i]][i] for i in range(len(lens))])).to(device)
        hidden_state = (torch.stack([all_hidden_states[lens[i]][i] for i in range(len(lens))])).to(device)

        return latent_sample, latent_mean, latent_logvar, hidden_state

    def get_value(self, state, belief, task, latent_sample, latent_mean, latent_logvar):
        latent = utl.get_latent_for_policy(self.args, latent_sample=latent_sample, latent_mean=latent_mean, latent_logvar=latent_logvar)
        return self.policy.actor_critic.get_value(state=state, belief=belief, task=task, latent=latent).detach()

    def update(self, state, belief, task, latent_sample, latent_mean, latent_logvar):
        """
        Meta-update.
        Here the policy is updated for good average performance across tasks.
        :return:
        """
        # bootstrap next value prediction
        with torch.no_grad():
            next_value = self.get_value(state=state,
                                        belief=belief,
                                        task=task,
                                        latent_sample=latent_sample,
                                        latent_mean=latent_mean,
                                        latent_logvar=latent_logvar)

        # compute returns for current rollouts
        self.policy_storage.compute_returns(next_value, self.args.policy_use_gae, self.args.policy_gamma,
                                            self.args.policy_tau,
                                            use_proper_time_limits=self.args.use_proper_time_limits)

        # update agent (this will also call the VAE update!)
        policy_train_stats = self.policy.update(
            policy_storage=self.policy_storage,
            encoder=self.vae.encoder,
            rlloss_through_encoder=self.args.rlloss_through_encoder,
            compute_vae_loss=self.vae.compute_vae_loss)

        return policy_train_stats

    def log(self, run_stats, train_stats, start_time):

        # --- visualise behaviour of policy ---

        if self.iter_idx % self.args.vis_interval == 0:

            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None
            utl_eval.visualise_behaviour(args=self.args,
                                         policy=self.policy,
                                         image_folder=self.logger.full_output_folder,
                                         iter_idx=self.iter_idx,
                                         ret_rms=ret_rms,
                                         encoder=self.vae.encoder,
                                         reward_decoder=self.vae.reward_decoder,
                                         state_decoder=self.vae.state_decoder,
                                         task_decoder=self.vae.task_decoder,
                                         compute_rew_reconstruction_loss=self.vae.compute_rew_reconstruction_loss,
                                         compute_state_reconstruction_loss=self.vae.compute_state_reconstruction_loss,
                                         compute_task_reconstruction_loss=self.vae.compute_task_reconstruction_loss,
                                         compute_kl_loss=self.vae.compute_kl_loss,
                                         )

        # --- evaluate policy ----

        if self.iter_idx % self.args.eval_interval == 0:

            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None
            returns_per_episode = utl_eval.evaluate(args=self.args,
                                                    policy=self.policy,
                                                    ret_rms=ret_rms,
                                                    encoder=self.vae.encoder,
                                                    iter_idx=self.iter_idx
                                                    )

            # log the return avg/std across tasks (=processes)
            returns_avg = returns_per_episode.mean(dim=0)
            returns_std = returns_per_episode.std(dim=0)
            for k in range(len(returns_avg)):
                self.logger.add('return_avg_per_iter/episode_{}'.format(k + 1), returns_avg[k], self.iter_idx)
                self.logger.add('return_avg_per_frame/episode_{}'.format(k + 1), returns_avg[k], self.frames)
                self.logger.add('return_std_per_iter/episode_{}'.format(k + 1), returns_std[k], self.iter_idx)
                self.logger.add('return_std_per_frame/episode_{}'.format(k + 1), returns_std[k], self.frames)

            print(f"Updates {self.iter_idx}, "
                  f"Frames {self.frames}, "
                  f"FPS {int(self.frames / (time.time() - start_time))}, "
                  f"\n Mean return (train): {returns_avg[-1].item()} \n"
                  )

        # --- save models ---

        if self.iter_idx % self.args.save_interval == 0:
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            idx_labels = ['']
            if self.args.save_intermediate_models:
                idx_labels.append(int(self.iter_idx))

            for idx_label in idx_labels:

                torch.save(self.policy.actor_critic, os.path.join(save_path, f"policy{idx_label}.pt"))
                torch.save(self.vae.encoder, os.path.join(save_path, f"encoder{idx_label}.pt"))
                if self.vae.state_decoder is not None:
                    torch.save(self.vae.state_decoder, os.path.join(save_path, f"state_decoder{idx_label}.pt"))
                if self.vae.reward_decoder is not None:
                    torch.save(self.vae.reward_decoder, os.path.join(save_path, f"reward_decoder{idx_label}.pt"))
                if self.vae.task_decoder is not None:
                    torch.save(self.vae.task_decoder, os.path.join(save_path, f"task_decoder{idx_label}.pt"))

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

            self.logger.add('policy_losses/value_loss', train_stats[0], self.iter_idx)
            self.logger.add('policy_losses/action_loss', train_stats[1], self.iter_idx)
            self.logger.add('policy_losses/dist_entropy', train_stats[2], self.iter_idx)
            self.logger.add('policy_losses/sum', train_stats[3], self.iter_idx)

            self.logger.add('policy/action', run_stats[0][0].float().mean(), self.iter_idx)
            if hasattr(self.policy.actor_critic, 'logstd'):
                self.logger.add('policy/action_logstd', self.policy.actor_critic.dist.logstd.mean(), self.iter_idx)
            self.logger.add('policy/action_logprob', run_stats[1].mean(), self.iter_idx)
            self.logger.add('policy/value', run_stats[2].mean(), self.iter_idx)

            self.logger.add('encoder/latent_mean', torch.cat(self.policy_storage.latent_mean).mean(), self.iter_idx)
            self.logger.add('encoder/latent_logvar', torch.cat(self.policy_storage.latent_logvar).mean(), self.iter_idx)

            # log the average weights and gradients of all models (where applicable)
            for [model, name] in [
                [self.policy.actor_critic, 'policy'],
                [self.vae.encoder, 'encoder'],
                [self.vae.reward_decoder, 'reward_decoder'],
                [self.vae.state_decoder, 'state_transition_decoder'],
                [self.vae.task_decoder, 'task_decoder']
            ]:
                if model is not None:
                    param_list = list(model.parameters())
                    param_mean = np.mean([param_list[i].data.cpu().numpy().mean() for i in range(len(param_list))])
                    self.logger.add('weights/{}'.format(name), param_mean, self.iter_idx)
                    if name == 'policy':
                        self.logger.add('weights/policy_std', param_list[0].data.mean(), self.iter_idx)
                    if param_list[0].grad is not None:
                        param_grad_mean = np.mean([param_list[i].grad.cpu().numpy().mean() for i in range(len(param_list))])
                        self.logger.add('gradients/{}'.format(name), param_grad_mean, self.iter_idx)
