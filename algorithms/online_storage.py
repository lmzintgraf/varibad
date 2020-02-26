"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

Used for on-policy rollout storages.
"""
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _flatten_helper(T, N, _tensor):
    return _tensor.reshape(T * N, *_tensor.size()[2:])


class OnlineStorage(object):
    def __init__(self, args, num_steps, num_processes, obs_shape, action_space,
                 hidden_size, latent_dim, normalise_observations, normalise_rewards,
                 env_state_dim=None, belief_dim=None):

        self.args = args
        self.env_state_dim = env_state_dim
        self.belief_dim = belief_dim

        # normalisation for PPO
        self.normalise_observations = normalise_observations
        self.normalise_rewards = normalise_rewards

        # latent
        self.latent_dim = latent_dim
        if latent_dim is not None:
            self.latent_samples = []
            self.latent_mean = []
            self.latent_logvar = []
        self.hidden_size = hidden_size
        self.hidden_states = torch.zeros(num_steps + 1, num_processes, hidden_size)

        # rollouts
        # this will include s_0 when state was reset, skipping s_N
        self.done = torch.zeros(num_steps + 1, num_processes, 1)
        # this will include s_0 when state was reset, skipping s_N
        self.prev_obs_raw = torch.zeros(num_steps + 1, num_processes, obs_shape)
        self.prev_obs_normalised = torch.zeros(num_steps + 1, num_processes, obs_shape)
        # this will include s_N when state was reset, skipping s_0
        self.next_obs_raw = torch.zeros(num_steps, num_processes, obs_shape)
        self.next_obs_normalised = torch.zeros(num_steps, num_processes, obs_shape)
        if self.env_state_dim is not None:
            self.env_states = torch.zeros(num_steps, num_processes, env_state_dim)
        if self.belief_dim is not None:
            self.beliefs = torch.zeros(num_steps, num_processes, belief_dim)
        # rewards
        self.rewards_raw = torch.zeros(num_steps, num_processes, 1)
        self.rewards_normalised = torch.zeros(num_steps, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.reset_task = torch.zeros(num_steps + 1, num_processes, 1)
        # masks that indicate whether it's a true terminal state (false) or time limit end state (true)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.num_processes = num_processes
        self.step = 0

    def to(self, device):
        self.done = self.done.to(device)
        self.prev_obs_raw = self.prev_obs_raw.to(device)
        self.prev_obs_normalised = self.prev_obs_normalised.to(device)
        self.next_obs_raw = self.next_obs_raw.to(device)
        self.next_obs_normalised = self.next_obs_normalised.to(device)
        self.rewards_raw = self.rewards_raw.to(device)
        self.rewards_normalised = self.rewards_normalised.to(device)
        self.reset_task = self.reset_task.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.hidden_states = self.hidden_states.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        if self.latent_dim is not None:
            self.latent_samples = [t.to(device) for t in self.latent_samples]
            self.latent_mean = [t.to(device) for t in self.latent_mean]
            self.latent_logvar = [t.to(device) for t in self.latent_logvar]
        if self.env_state_dim is not None:
            self.env_states = self.env_states.to(device)
        if self.belief_dim is not None:
            self.beliefs = self.beliefs.to(device)

    def insert(self,
               obs_raw,
               obs_normalised,
               actions,
               action_log_probs,
               rewards_raw,
               rewards_normalised,
               value_preds,
               masks,
               bad_masks,
               done,
               #
               reset_task=None,
               hidden_states=None,
               latent_sample=None,
               latent_mean=None,
               latent_logvar=None,
               #
               env_states=None,
               beliefs=None,
               ):

        self.prev_obs_raw[self.step + 1].copy_(obs_raw)
        self.prev_obs_normalised[self.step + 1].copy_(obs_normalised)
        self.actions[self.step] = actions.clone()
        if action_log_probs is not None:
            self.action_log_probs[self.step].copy_(action_log_probs)
        self.rewards_raw[self.step].copy_(rewards_raw)
        self.rewards_normalised[self.step].copy_(rewards_normalised)
        if isinstance(value_preds, list):
            self.value_preds[self.step].copy_(value_preds[0])
        else:
            self.value_preds[self.step].copy_(value_preds)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.done[self.step + 1].copy_(done)
        if hidden_states is not None:
            self.hidden_states[self.step + 1].copy_(hidden_states)
        else:
            self.hidden_states = None
        if reset_task is not None:
            if reset_task.dim() == 1:
                reset_task = reset_task.unsqueeze(1)
            self.reset_task[self.step + 1].copy_(reset_task)
        else:
            self.reset_task = None
        if latent_sample is not None:
            self.latent_samples.append(latent_sample.detach().clone())
            self.latent_mean.append(latent_mean.detach().clone())
            self.latent_logvar.append(latent_logvar.detach().clone())
        else:
            self.latent_samples = None
            self.latent_mean = None
            self.latent_logvar = None
        if env_states is not None:
            self.env_states[self.step].copy_(env_states)
        if beliefs is not None:
            self.beliefs[self.step].copy_(beliefs)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.done[0].copy_(self.done[-1])
        self.prev_obs_raw[0].copy_(self.prev_obs_raw[-1])
        self.prev_obs_normalised[0].copy_(self.prev_obs_normalised[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        if self.hidden_states is not None:
            self.hidden_states[0].copy_(self.hidden_states[-1])
        if self.reset_task is not None:
            self.reset_task[0].copy_(self.reset_task[-1])
        if self.latent_dim is not None:
            self.latent_samples = []
            self.latent_mean = []
            self.latent_logvar = []

    def compute_returns(self, next_value, use_gae, gamma, tau, use_proper_time_limits=True):

        if self.normalise_rewards:
            rewards = self.rewards_normalised.clone()
        else:
            rewards = self.rewards_raw.clone()

        self._compute_returns(next_value=next_value, rewards=rewards, value_preds=self.value_preds,
                              returns=self.returns,
                              gamma=gamma, tau=tau, use_gae=use_gae, use_proper_time_limits=use_proper_time_limits)

    def _compute_returns(self, next_value, rewards, value_preds, returns, gamma, tau, use_gae, use_proper_time_limits):

        if use_proper_time_limits:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.size(0))):
                    returns[step] = (returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]) * self.bad_masks[
                        step + 1] + (1 - self.bad_masks[step + 1]) * value_preds[step]
        else:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.size(0))):
                    returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]

    def num_transitions(self):
        return len(self.prev_obs_raw) * self.num_processes

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards_raw.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:

            if self.normalise_observations:
                prev_obs = self.prev_obs_normalised
            else:
                prev_obs = self.prev_obs_raw

            obs_batch = prev_obs[:-1].reshape(-1, *prev_obs.size()[2:])[indices]
            actions_batch = self.actions.reshape(-1, self.actions.size(-1))[indices]

            if self.latent_dim is not None and self.latent_mean is not None:
                latent_sample_batch = torch.cat(self.latent_samples[:-1])[indices]
                latent_mean_batch = torch.cat(self.latent_mean[:-1])[indices]
                latent_logvar_batch = torch.cat(self.latent_logvar[:-1])[indices]
            else:
                latent_sample_batch = latent_mean_batch = latent_logvar_batch = None

            value_preds_batch = self.value_preds[:-1].reshape(-1, 1)[indices]
            return_batch = self.returns[:-1].reshape(-1, 1)[indices]

            old_action_log_probs_batch = self.action_log_probs.reshape(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.reshape(-1, 1)[indices]

            yield obs_batch, actions_batch, latent_sample_batch, latent_mean_batch, latent_logvar_batch, \
                  value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ
