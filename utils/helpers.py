import os
import gym
import pickle
import random
import warnings
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from environments.parallel_envs import make_vec_envs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def save_models(args, logger, policy, vae, envs, iter_idx):
#     # TODO: save parameters, not entire model
#
#     save_path = os.path.join(logger.full_output_folder, 'models')
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     try:
#         torch.save(policy.actor_critic, os.path.join(save_path, "policy{0}.pt".format(iter_idx)))
#     except AttributeError:
#         torch.save(policy.policy, os.path.join(save_path, "policy{0}.pt".format(iter_idx)))
#     torch.save(vae.encoder, os.path.join(save_path, "encoder{0}.pt".format(iter_idx)))
#     if vae.state_decoder is not None:
#         torch.save(vae.state_decoder, os.path.join(save_path, "state_decoder{0}.pt".format(iter_idx)))
#     if vae.reward_decoder is not None:
#         torch.save(vae.reward_decoder,
#                    os.path.join(save_path, "reward_decoder{0}.pt".format(iter_idx)))
#     if vae.task_decoder is not None:
#         torch.save(vae.task_decoder, os.path.join(save_path, "task_decoder{0}.pt".format(iter_idx)))
#
#     # save normalisation params of envs
#     if args.norm_rew_for_policy:
#         rew_rms = envs.venv.ret_rms
#         save_obj(rew_rms, save_path, "env_rew_rms{0}.pkl".format(iter_idx))
#     if args.norm_obs_for_policy:
#         obs_rms = envs.venv.obs_rms
#         save_obj(obs_rms, save_path, "env_obs_rms{0}.pkl".format(iter_idx))


def reset_env(env, args, indices=None, state=None):
    """ env can be many environments or just one """
    # reset all environments
    if indices is not None:
        assert not isinstance(indices[0], bool)
    if (indices is None) or (len(indices) == args.num_processes):
        state = env.reset().to(device)
    # reset only the ones given by indices
    else:
        assert state is not None
        for i in indices:
            state[i] = env.reset(index=i)

    belief = torch.from_numpy(env.get_belief()).to(device) if args.pass_belief_to_policy else None
    task = torch.from_numpy(env.get_task()).to(device) if args.pass_task_to_policy else None
        
    return state, belief, task


def env_step(env, action, args):

    next_obs, reward, done, infos = env.step(action.detach())

    if isinstance(next_obs, list):
        next_obs = [o.to(device) for o in next_obs]
    else:
        next_obs = next_obs.to(device)
    if isinstance(reward, list):
        reward = [r.to(device) for r in reward]
    else:
        reward = reward.to(device)

    belief = torch.from_numpy(env.get_belief()).to(device) if args.pass_belief_to_policy else None
    task = torch.from_numpy(env.get_task()).to(device) if args.pass_task_to_policy else None

    return [next_obs, belief, task], reward, done, infos


def select_action(args,
                  policy,
                  deterministic,
                  state=None,
                  belief=None,
                  task=None,
                  latent_sample=None, latent_mean=None, latent_logvar=None):
    """ Select action using the policy. """
    latent = get_latent_for_policy(args=args, latent_sample=latent_sample, latent_mean=latent_mean, latent_logvar=latent_logvar)
    action = policy.act(state=state, latent=latent, belief=belief, task=task, deterministic=deterministic)
    if isinstance(action, list) or isinstance(action, tuple):
        value, action, action_log_prob = action
    else:
        value = None
        action_log_prob = None
    action = action.to(device)
    return value, action, action_log_prob


def get_latent_for_policy(args, latent_sample=None, latent_mean=None, latent_logvar=None):

    if (latent_sample is None) and (latent_mean is None) and (latent_logvar is None):
        return None

    if args.add_nonlinearity_to_latent:
        latent_sample = F.relu(latent_sample)
        latent_mean = F.relu(latent_mean)
        latent_logvar = F.relu(latent_logvar)

    if args.sample_embeddings:
        latent = latent_sample
    else:
        latent = torch.cat((latent_mean, latent_logvar), dim=-1)

    if latent.shape[0] == 1:
        latent = latent.squeeze(0)

    return latent


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

    # TODO: move the sampling out of the encoder!

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
              'Use only one process and set --deterministic_execution to True if you want identical results '
              '(only recommended for debugging).')


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
        detach_every
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
        h = encoder.reset_hidden(h, policy_storage.done[i + 1])

        ts, tm, tl, h = encoder(policy_storage.actions.float()[i:i + 1],
                                policy_storage.next_state[i:i + 1],
                                policy_storage.rewards_raw[i:i + 1],
                                h,
                                sample=sample,
                                return_prior=False,
                                detach_every=detach_every
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
    std = torch.exp(0.5 * logvar)
    if num is not None:
        std = std.repeat(num, 1)
        mu = mu.repeat(num, 1)
    eps = torch.randn_like(std)
    return mu + std * eps


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


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))


def get_task_dim(args):
    env = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                        gamma=args.policy_gamma, device=device,
                        episodes_per_task=args.max_rollouts_per_task,
                        normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                        )
    return env.task_dim


def get_num_tasks(args):
    env = gym.make(args.env_name)
    try:
        num_tasks = env.num_tasks
    except AttributeError:
        num_tasks = None
    return num_tasks


def clip(value, low, high):
    """Imitates `{np,tf}.clip`.

    `torch.clamp` doesn't support tensor valued low/high so this provides the
    clip functionality.

    TODO(hartikainen): The broadcasting hasn't been extensively tested yet,
        but works for the regular cases where
        `value.shape == low.shape == high.shape` or when `{low,high}.shape == ()`.
    """
    low, high = torch.tensor(low), torch.tensor(high)

    assert torch.all(low <= high), (low, high)

    clipped_value = torch.max(torch.min(value, high), low)
    return clipped_value
