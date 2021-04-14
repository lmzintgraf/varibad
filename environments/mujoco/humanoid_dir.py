"""
    Based on environment in PEARL:
    https://github.com/katerakelly/oyster/blob/master/rlkit/envs/humanoid_dir.py
"""
import numpy as np
from gym.envs.mujoco import HumanoidEnv as HumanoidEnv
from gym import spaces

import random

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


class HumanoidDirEnv(HumanoidEnv):

    def __init__(self, task={}, max_episode_steps=200, n_tasks=2, randomize_tasks=True):
        self.set_task(self.sample_tasks(1)[0])
        self.task_dim = 1
        self._max_episode_steps = max_episode_steps
        self.action_scale = 1 # Mujoco environment initialization takes a step, 
        super(HumanoidDirEnv, self).__init__()

        # Override action space to make it range from  (-1, 1)
        assert (self.action_space.low == -self.action_space.high).all() 
        self.action_scale = self.action_space.high[0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self.action_space.shape) # Overriding original action_space which is (-0.4, 0.4, shape = (17, ))
        
        

    def step(self, action):
        pos_before = np.copy(mass_center(self.model, self.sim)[:2])

        rescaled_action = action * self.action_scale # Scale the action from (-1, 1) to original.
        self.do_simulation(rescaled_action, self.frame_skip) 
        pos_after = mass_center(self.model, self.sim)[:2]

        alive_bonus = 5.0
        data = self.sim.data
        goal_direction = (np.cos(self._goal), np.sin(self._goal))
        lin_vel_cost = 0.25 * np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos


        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0)) 
        # done = False

        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus,
                                                   reward_impact=-quad_impact_cost)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    # def get_all_task_idx(self):
    #     return range(len(self.tasks))

    # def reset_task(self, idx):
    #     self._task = self.tasks[idx]
    #     self._goal = self._task['goal'] # assume parameterization of task by single vector

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    def set_task(self, task):
        self._goal = task

    def get_task(self):
        return np.array([self._goal])

    def sample_tasks(self, num_tasks):
        return [random.uniform(0., 2.0 * np.pi) for _ in range(num_tasks)]