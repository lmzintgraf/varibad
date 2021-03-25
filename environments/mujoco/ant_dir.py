import random

import numpy as np

from environments.mujoco.ant import AntEnv


class AntDirEnv(AntEnv):
    """
    Forward/backward ant direction environment
    """

    def __init__(self, max_episode_steps=200):
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        super(AntDirEnv, self).__init__()

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self.goal_direction), np.sin(self.goal_direction))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2] / self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
            task=self.get_task()
        )

    def sample_tasks(self, n_tasks):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        return [random.choice([-1.0, 1.0]) for _ in range(n_tasks, )]

    def set_task(self, task):
        if isinstance(task, np.ndarray):
            task = task[0]
        self.goal_direction = task

    def get_task(self):
        return np.array([self.goal_direction])


class AntDir2DEnv(AntDirEnv):
    def sample_tasks(self, n_tasks):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        directions = np.array([random.gauss(mu=0, sigma=1) for _ in range(n_tasks * 2)]).reshape((n_tasks, 2))
        directions /= np.linalg.norm(directions, axis=1)[..., np.newaxis]
        return directions


class AntDirOracleEnv(AntDirEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            [self.goal_direction],
        ])


class AntDir2DOracleEnv(AntDir2DEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            [self.goal_direction],
        ])
