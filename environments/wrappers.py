import gym
import numpy as np
from gym import spaces
from gym.envs.registration import load

from environments.mujoco import rand_param_envs

try:
    # this is to suppress some warnings (in the newer mujoco versions)
    gym.logger.set_level(40)
except AttributeError:
    pass


def mujoco_wrapper(entry_point, **kwargs):
    # Load the environment from its entry point
    env_cls = load(entry_point)
    env = env_cls(**kwargs)
    return env


class VariBadWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 episodes_per_task,
                 add_done_info=None,  # force to turn this on/off
                 ):
        """
        Wrapper, creates a multi-episode (BA)MDP around a one-episode MDP. Automatically deals with
        - horizons H in the MDP vs horizons H+ in the BAMDP,
        - resetting the tasks
        - adding the done info to the state (might be needed to make states markov)
        """

        super().__init__(env)

        # make sure we can call these attributes even if the orig env does not have them
        if not hasattr(self.env.unwrapped, 'task_dim'):
            self.env.unwrapped.task_dim = 0
        if not hasattr(self.env.unwrapped, 'belief_dim'):
            self.env.unwrapped.belief_dim = 0
        if not hasattr(self.env.unwrapped, 'get_belief'):
            self.env.unwrapped.get_belief = lambda: None
        if not hasattr(self.env.unwrapped, 'num_states'):
            self.env.unwrapped.num_states = None

        if add_done_info is None:
            if episodes_per_task > 1:
                self.add_done_info = True
            else:
                self.add_done_info = False
        else:
            self.add_done_info = add_done_info

        if self.add_done_info:
            if isinstance(self.observation_space, spaces.Box) or isinstance(self.observation_space,
                                                                            rand_param_envs.gym.spaces.box.Box):
                if len(self.observation_space.shape) > 1:
                    raise ValueError  # can't add additional info for obs of more than 1D
                self.observation_space = spaces.Box(low=np.array([*self.observation_space.low, 0]),
                                                    # shape will be deduced from this
                                                    high=np.array([*self.observation_space.high, 1])
                                                    )
            else:
                # TODO: add something simliar for the other possible spaces,
                # "Space", "Discrete", "MultiDiscrete", "MultiBinary", "Tuple", "Dict", "flatdim", "flatten", "unflatten"
                raise NotImplementedError

        # calculate horizon length H^+
        self.episodes_per_task = episodes_per_task
        # counts the number of episodes
        self.episode_count = 0

        # count timesteps in BAMDP
        self.step_count_bamdp = 0.0
        # the horizon in the BAMDP is the one in the MDP times the number of episodes per task,
        # and if we train a policy that maximises the return over all episodes
        # we add transitions to the reset start in-between episodes
        try:
            self.horizon_bamdp = self.episodes_per_task * self.env._max_episode_steps
        except AttributeError:
            self.horizon_bamdp = self.episodes_per_task * self.env.unwrapped._max_episode_steps

        # add dummy timesteps in-between episodes for resetting the MDP
        self.horizon_bamdp += self.episodes_per_task - 1

        # this tells us if we have reached the horizon in the underlying MDP
        self.done_mdp = True

    def reset(self, task=None):
        """ Resets the BAMDP """

        # reset task
        self.env.reset_task(task)
        # normal reset
        try:
            state = self.env.reset()
        except AttributeError:
            state = self.env.unwrapped.reset()

        self.episode_count = 0
        self.step_count_bamdp = 0
        self.done_mdp = False
        if self.add_done_info:
            state = np.concatenate((state, [0.0]))

        return state

    def reset_mdp(self):
        """ Resets the underlying MDP only (*not* the task). """
        state = self.env.reset()
        if self.add_done_info:
            state = np.concatenate((state, [0.0]))
        self.done_mdp = False
        return state

    def step(self, action):

        # do normal environment step in MDP
        state, reward, self.done_mdp, info = self.env.step(action)

        info['done_mdp'] = self.done_mdp

        if self.add_done_info:
            state = np.concatenate((state, [float(self.done_mdp)]))

        self.step_count_bamdp += 1
        # if we want to maximise performance over multiple episodes,
        # only say "done" when we collected enough episodes in this task
        done_bamdp = False
        if self.done_mdp:
            self.episode_count += 1
            if self.episode_count == self.episodes_per_task:
                done_bamdp = True

        if self.done_mdp and not done_bamdp:
            info['start_state'] = self.reset_mdp()

        return state, reward, done_bamdp, info

    def __getattr__(self, attr):
        """
        If env does not have the attribute then call the attribute in the wrapped_env
        (This one's only needed for mujoco 131)
        """
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


class TimeLimitMask(gym.Wrapper):

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True
        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def __getattr__(self, attr):
        """
        If env does not have the attribute then call the attribute in the wrapped_env
        (This one's only needed for mujoco 131)
        """
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
