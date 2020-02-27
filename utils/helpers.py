import os
import pickle
import random
import warnings

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def env_step(envs, action):
    next_obs, reward, done, infos = envs.step(action.detach())

    if isinstance(next_obs, list):
        next_obs = [o.to(device) for o in next_obs]
    else:
        next_obs = next_obs.to(device)
    if isinstance(reward, list):
        reward = [r.to(device) for r in reward]
    else:
        reward = reward.to(device)

    return next_obs, reward, done, infos


def select_action(args,
                  policy,
                  obs,
                  deterministic,
                  latent_sample=None, latent_mean=None, latent_logvar=None):
    """
    Select action using the policy.
    """

    # augment the observation with the latent distribution
    obs = get_augmented_obs(args, obs, latent_sample, latent_mean, latent_logvar)
    action = policy.act(obs, deterministic)
    if isinstance(action, list) or isinstance(action, tuple):
        value, action, action_log_prob = action
    else:
        value = None
        action_log_prob = None
    action = action.to(device)
    return value, action, action_log_prob


def get_augmented_obs(args, obs,
                      latent_sample=None, latent_mean=None, latent_logvar=None):
    obs_augmented = obs.clone()

    if latent_sample is None:
        sample_embeddings = False
    else:
        sample_embeddings = args.sample_embeddings

    if not args.condition_policy_on_state:
        obs_augmented = torch.zeros(0, ).to(device)

    if sample_embeddings and (latent_sample is not None):
        obs_augmented = torch.cat((obs_augmented, latent_sample), dim=1)
    elif (latent_mean is not None) and (latent_logvar is not None):
        latent_mean = latent_mean.reshape((-1, latent_mean.shape[-1]))
        latent_logvar = latent_logvar.reshape((-1, latent_logvar.shape[-1]))
        obs_augmented = torch.cat((obs_augmented, latent_mean, latent_logvar), dim=1)

    return obs_augmented


def update_encoding(encoder, next_obs, action, reward, done, hidden_state):
    # reset hidden state of the recurrent net when we reset the task
    if done is not None:
        hidden_state = encoder.reset_hidden(hidden_state, done)

    with torch.no_grad():
        latent_sample, latent_mean, latent_logvar, hidden_state = encoder(actions=action.float(),
                                                                          states=next_obs,
                                                                          rewards=reward,
                                                                          hidden_state=hidden_state,
                                                                          return_prior=False)

    return latent_sample, latent_mean, latent_logvar, hidden_state


def seed(seed, deterministic_execution=False):
    print('Seeding random, torch, numpy.')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    if deterministic_execution:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('Note that due to parallel processing results will be similar but not identical. '
              'Use only one process and set --deterministic_execution to True if you want identical results. '
              '(Not recommended; will slow code down and might not be possible with A2C) ')


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def recompute_embeddings(
        policy_storage,
        encoder,
        sample,
        update_idx,
):
    # get the prior
    latent_sample = [policy_storage.latent_samples[0].detach().clone()]
    latent_mean = [policy_storage.latent_mean[0].detach().clone()]
    latent_logvar = [policy_storage.latent_logvar[0].detach().clone()]

    latent_sample[0].requires_grad = True
    latent_mean[0].requires_grad = True
    latent_logvar[0].requires_grad = True

    # loop through experience and update hidden state
    # (we need to loop because we sometimes need to reset the hidden state)
    h = policy_storage.hidden_states[0].detach()
    for i in range(policy_storage.actions.shape[0]):
        # reset hidden state of the GRU when we reset the task
        reset_task = policy_storage.done[i + 1]
        h = encoder.reset_hidden(h, reset_task)

        ts, tm, tl, h = encoder(policy_storage.actions.float()[i:i + 1],
                                policy_storage.next_obs_raw[i:i + 1],
                                policy_storage.rewards_raw[i:i + 1],
                                h,
                                sample=sample,
                                return_prior=False
                                )

        # print(i, reset_task.sum())
        # print(i, (policy_storage.latent_mean[i + 1] - tm).sum())
        # print(i, (policy_storage.latent_logvar[i + 1] - tl).sum())
        # print(i, (policy_storage.hidden_states[i + 1] - h).sum())

        latent_sample.append(ts)
        latent_mean.append(tm)
        latent_logvar.append(tl)

    if update_idx == 0:
        try:
            assert (torch.cat(policy_storage.latent_mean) - torch.cat(latent_mean)).sum() == 0
            assert (torch.cat(policy_storage.latent_logvar) - torch.cat(latent_logvar)).sum() == 0
        except AssertionError:
            warnings.warn('You are not recomputing the embeddings correctly!')
            import pdb
            pdb.set_trace()

    policy_storage.latent_samples = latent_sample
    policy_storage.latent_mean = latent_mean
    policy_storage.latent_logvar = latent_logvar


class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """

    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return torch.zeros(0, ).to(device)


def sample_gaussian(mu, logvar, num=None):
    if num is None:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    else:
        std = torch.exp(0.5 * logvar).repeat(num, 1)
        eps = torch.randn_like(std)
        mu = mu.repeat(num, 1)
        return eps.mul(std).add_(mu)


def save_obj(obj, folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'rb') as f:
        return pickle.load(f)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # PyTorch version.
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).float().to(device)
        self.var = torch.ones(shape).float().to(device)
        self.count = epsilon

    def update(self, x):
        x = x.view((-1, x.shape[-1]))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
