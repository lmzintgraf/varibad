"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import torch.nn as nn
import torch.optim as optim

from utils import helpers as utl


class A2C:
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 optimiser_vae=None,
                 lr=None,
                 eps=None,
                 alpha=None,
                 ):

        # the model
        self.actor_critic = actor_critic

        # coefficients for mixing the value and entropy loss
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        # optimiser
        self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)
        self.optimiser_vae = optimiser_vae

    def update(self,
               args,
               policy_storage,
               encoder=None,  # variBAD encoder
               rlloss_through_encoder=False,  # whether or not to backprop RL loss through encoder
               compute_vae_loss=None  # function that can compute the VAE loss
               ):

        # -- get action values --
        advantages = policy_storage.returns[:-1] - policy_storage.value_preds[:-1]

        if rlloss_through_encoder:
            # re-compute encoding (to build the computation graph from scratch)
            utl.recompute_embeddings(policy_storage, encoder, sample=False, update_idx=0)

        data_generator = policy_storage.feed_forward_generator(advantages, 1)
        for sample in data_generator:

            obs_batch, actions_batch, latent_sample_batch, latent_mean_batch, latent_logvar_batch, value_preds_batch, \
            return_batch, old_action_log_probs_batch, adv_targ = sample

            if not rlloss_through_encoder:
                obs_batch = obs_batch.detach()
                if latent_sample_batch is not None:
                    latent_sample_batch = latent_sample_batch.detach()
                    latent_mean_batch = latent_mean_batch.detach()
                    latent_logvar_batch = latent_logvar_batch.detach()

            obs_aug = utl.get_augmented_obs(args=args,
                                            obs=obs_batch,
                                            latent_sample=latent_sample_batch, latent_mean=latent_mean_batch,
                                            latent_logvar=latent_logvar_batch
                                            )

            values, action_log_probs, dist_entropy, action_mean, action_logstd = \
                self.actor_critic.evaluate_actions(obs_aug, actions_batch, return_action_mean=True)

            # --  UPDATE --

            # zero out the gradients
            self.optimizer.zero_grad()
            if rlloss_through_encoder:
                self.optimiser_vae.zero_grad()

            # compute policy loss and backprop
            value_loss = (return_batch - values).pow(2).mean()
            action_loss = -(adv_targ.detach() * action_log_probs).mean()

            # (loss = value loss + action loss + entropy loss, weighted)
            loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

            # compute vae loss and backprop
            if rlloss_through_encoder:
                loss += args.vae_loss_coeff * compute_vae_loss()

            # compute gradients (will attach to all networks involved in this computation)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), args.policy_max_grad_norm)
            if encoder is not None and rlloss_through_encoder:
                nn.utils.clip_grad_norm_(encoder.parameters(), args.policy_max_grad_norm)

            # update
            self.optimizer.step()
            if rlloss_through_encoder:
                self.optimiser_vae.step()

        if (not rlloss_through_encoder) and (self.optimiser_vae is not None):
            for _ in range(args.num_vae_updates - 1):
                compute_vae_loss(update=True)

        return value_loss, action_loss, dist_entropy, loss

    def act(self, obs, deterministic=False):
        return self.actor_critic.act(obs, deterministic=deterministic)
