"""
Taken from https://github.com/openai/baselines
"""
import numpy as np

from environments.env_utils.running_mean_std import RunningMeanStd
from . import VecEnvWrapper


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, clipobs=10., cliprew=10., gamma=0.99, epsilon=1e-8,
                 normalise_obs=False, normalise_rew=False,
                 obs_rms=None, ret_rms=None):
        VecEnvWrapper.__init__(self, venv)

        self.normalise_obs = normalise_obs
        self.normalise_rew = normalise_rew

        # clip params
        self.clipobs = clipobs
        self.cliprew = cliprew

        # set the running mean and std values
        if self.normalise_obs:
            if obs_rms is None:
                self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
            else:
                self.obs_rms = obs_rms

        if self.normalise_rew:
            if ret_rms is None:
                self.ret_rms = RunningMeanStd(shape=())
            else:
                self.ret_rms = ret_rms

        # discounted return for each environment
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def step_wait(self):
        # execute action
        obs, rews, news, infos = self.venv.step_wait()
        # update discounted return
        self.ret = self.ret * self.gamma + rews
        self.ret[news] = 0.
        # normalise
        obs = self._obfilt(obs)
        rews = self._rewfilt(rews)
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.normalise_obs:
            # update rolling mean / std
            if self.training:
                self.obs_rms.update(obs)
            # normalise
            obs_norm = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clipobs,
                               self.clipobs)
            return [obs, obs_norm]
        else:
            return [obs, obs]

    def _rewfilt(self, rews):
        if self.normalise_rew:
            # update rolling mean / std
            if self.training:
                self.ret_rms.update(self.ret)
            # normalise
            rews_norm = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            return [rews, rews_norm]
        else:
            return [rews, rews]

    def reset_mdp(self, index=None):
        if index is None:
            obs = self.venv.reset_mdp()
            obs = self._obfilt(obs)
        else:
            self.venv.remotes[index].send(('reset_mdp', None))
            obs = self.venv.remotes[index].recv()
            obs = self._obfilt(obs)
        return obs

    def reset(self, index=None, task=None):
        self.ret = np.zeros(self.num_envs)
        if index is None:
            obs = self.venv.reset(task=task)
            obs = self._obfilt(obs)
        else:
            try:
                self.venv.remotes[index].send(('reset', task))
                obs = self.venv.remotes[index].recv()
            except AttributeError:
                obs = self.venv.envs[index].reset(task=task)
            obs = self._obfilt(obs)
        return obs

    def __getattr__(self, attr):
        """
        If env does not have the attribute then call the attribute in the wrapped_env
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

