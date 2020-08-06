import gym


class ExampleEnv(gym.Env):
    def __init__(self):
        super(ExampleEnv, self).__init__()

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: state, reward, done, info
        where info has to include a field 'task'.
        """
        pass

    def reset(self):
        """
        Reset the environment. This should *NOT* automatically reset the task!
        Resetting the task is handled in the varibad wrapper (see wrappers.py).
        """
        pass

    def get_task(self):
        """
        Return a task description, such as goal position or target velocity.
        """
        pass

    def reset_task(self, task=None):
        """
        Reset the task, either at random (if task=None) or the given task.
        Should *not* reset the environment.
        """
        pass

    def visualise_behaviour(self,
                            env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            reward_decoder=None,
                            state_decoder=None,
                            task_decoder=None,
                            image_folder=None,
                            **kwargs):
        """
        Optional. If this is not overwritten, a default visualisation will be used (see utils/evaluation.py).
        Should return the following:
            episode_latent_means, episode_latent_logvars, episode_prev_obs,
            episode_next_obs, episode_actions, episode_rewards, episode_returns
        where each element is either a list of length num_episodes,
        or "None" if not applicable.
        """
        pass
