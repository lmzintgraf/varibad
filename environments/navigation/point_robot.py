import numpy as np
from gym import Env
from gym import spaces


class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, max_episode_steps=100):
        self.reset_task()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))
        self._max_episode_steps = max_episode_steps

    def sample_task(self):
        goal = np.array([np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)])
        return goal

    def set_task(self, task):
        self._goal = task

    def get_task(self):
        return self._goal

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_task()
        self.set_task(task)

    def reset_model(self):
        # reset to a random location on the unit square
        self._state = np.random.uniform(-1., 1., size=(2,))
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        self._state = self._state + action
        x, y = self._state.flat
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = False
        ob = self._get_obs()
        info = {'task': self.get_task()}
        return ob, reward, done, info


class SparsePointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''

    def __init__(self, goal_radius=0.2, max_episode_steps=100):
        super().__init__(max_episode_steps=max_episode_steps)
        self.goal_radius = goal_radius
        self.reset_task()

    def sample_task(self):
        radius = 1.0
        angle = np.random.uniform(0, np.pi)
        xs = radius * np.cos(angle)
        ys = radius * np.sin(angle)
        return np.array([xs, ys])

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        # return ob, reward, done, d
        return ob, sparse_reward, done, d


class PointEnvOracle(PointEnv):
    def __init__(self, max_episode_steps=100):
        super().__init__(max_episode_steps=max_episode_steps)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))

    def _get_obs(self):
        return np.concatenate([self._state.flatten(), self._goal])


class SparsePointEnvOracle(SparsePointEnv):
    def __init__(self, max_episode_steps=100):
        super().__init__(max_episode_steps=max_episode_steps)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))

    def _get_obs(self):
        return np.concatenate([self._state.flatten(), self._goal])
