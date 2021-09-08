import random

import numpy as np
import torch
from gym import spaces

from environments.mujoco.half_cheetah import HalfCheetahEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HalfCheetahDirEnv(HalfCheetahEnv):
    """Half-cheetah environment with target direction, as described in [1]. The
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand_direc.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each
    time step a reward composed of a control cost and a reward equal to its
    velocity in the target direction. The tasks are generated by sampling the
    target directions from a Bernoulli distribution on {-1, 1} with parameter
    0.5 (-1: backward, +1: forward).

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """

    def __init__(self, max_episode_steps=200, **kwargs):
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        super(HalfCheetahDirEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self.goal_direction * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     task=self.get_task())
        return observation, reward, done, infos

    def sample_tasks(self, n_tasks):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        return [random.choice([-1.0, 1.0]) for _ in range(n_tasks, )]

    def set_task(self, task):
        if isinstance(task, np.ndarray):
            task = task[0]
        self.goal_direction = task

    def get_task(self):
        return np.array([self.goal_direction])

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)


_ANT_DIR_ACTION_DIM = 8
_ANT_DIR_OBS_DIM = 28

_CHEETAH_DIR_ACTION_DIM = 6
_CHEETAH_DIR_OBS_DIM = 21


class CheetahDirUniEnv(HalfCheetahDirEnv):
    """Has the same action and state dim as AntDir."""
    def __init__(self, max_episode_steps=200, **kwargs):
        self.orig_action_dim = _CHEETAH_DIR_ACTION_DIM
        self.obs_dim_added = _ANT_DIR_OBS_DIM - _CHEETAH_DIR_OBS_DIM
        # initialise original cheetah env
        super().__init__(max_episode_steps, **kwargs)
        # overriding action space
        self.action_space = spaces.Box(low=np.concatenate((self.action_space.low,
                                                           -np.ones(_ANT_DIR_ACTION_DIM-self.orig_action_dim))),
                                       high=np.concatenate((self.action_space.high,
                                                            np.ones(_ANT_DIR_ACTION_DIM-self.orig_action_dim))),
                                       shape=(_ANT_DIR_ACTION_DIM,))
        # Note: the observation space is automatically generated from the _get_obs function, so we don't overwrite it

    def _get_obs(self):
        obs = super()._get_obs()
        # add zeros to match observation space we defined above
        obs = np.concatenate((obs, np.zeros(self.obs_dim_added)))
        return obs

    def step(self, action):
        # remove unused zeros
        action = action[:self.orig_action_dim]
        return super().step(action)
