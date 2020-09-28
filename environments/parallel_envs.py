"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import gym
import torch

from environments.env_utils.vec_env import VecEnvWrapper
from environments.env_utils.vec_env.dummy_vec_env import DummyVecEnv
from environments.env_utils.vec_env.subproc_vec_env import SubprocVecEnv
from environments.env_utils.vec_env.vec_normalize import VecNormalize
from environments.wrappers import TimeLimitMask, VariBadWrapper


def make_env(env_id, seed, rank, episodes_per_task, **kwargs):
    def _thunk():
        env = gym.make(env_id, **kwargs)
        if seed is not None:
            env.seed(seed + rank)
        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)
        env = VariBadWrapper(env=env, episodes_per_task=episodes_per_task)
        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, gamma,
                  device, episodes_per_task,
                  normalise_rew, ret_rms, rank_offset=0,
                  **kwargs):
    """
    :param ret_rms: running return and std for rewards
    """
    envs = [make_env(env_id=env_name, seed=seed, rank=rank_offset + i,
                     episodes_per_task=episodes_per_task, **kwargs)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, normalise_rew=normalise_rew, ret_rms=ret_rms)
        else:
            envs = VecNormalize(envs, normalise_rew=normalise_rew, ret_rms=ret_rms, gamma=gamma)

    envs = VecPyTorch(envs, device)

    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset_mdp(self, index=None):
        obs = self.venv.reset_mdp(index=index)
        if isinstance(obs, list):
            obs = [torch.from_numpy(o).float().to(self.device) for o in obs]
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def reset(self, index=None, task=None):
        if task is not None:
            assert isinstance(task, list)
        state = self.venv.reset(index=index, task=task)
        if isinstance(state, list):
            state = [torch.from_numpy(s).float().to(self.device) for s in state]
        else:
            state = torch.from_numpy(state).float().to(self.device)
        return state

    def step_async(self, actions):
        # actions = actions.squeeze(1).cpu().numpy()
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        state, reward, done, info = self.venv.step_wait()
        if isinstance(state, list):  # raw + normalised
            state = [torch.from_numpy(s).float().to(self.device) for s in state]
        else:
            state = torch.from_numpy(state).float().to(self.device)
        if isinstance(reward, list):  # raw + normalised
            reward = [torch.from_numpy(r).unsqueeze(dim=1).float().to(self.device) for r in reward]
        else:
            reward = torch.from_numpy(reward).unsqueeze(dim=1).float().to(self.device)
        return state, reward, done, info

    def __getattr__(self, attr):
        """ If env does not have the attribute then call the attribute in the wrapped_env """

        if attr in ['_max_episode_steps', 'task_dim', 'belief_dim', 'num_states']:
            return self.unwrapped.get_env_attr(attr)

        try:
            orig_attr = self.__getattribute__(attr)
        except AttributeError:
            orig_attr = self.unwrapped.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr
