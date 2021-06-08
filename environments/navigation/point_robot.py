import matplotlib as mpl
import random
import numpy as np
import torch
from utils import helpers as utl
import matplotlib.pyplot as plt
import seaborn as sns

from gym import Env
from gym import spaces

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def semi_circle_goal_sampler():
    r = 1.0
    angle = random.uniform(0, np.pi)
    goal = r * np.array((np.cos(angle), np.sin(angle)))
    return goal


def circle_goal_sampler():
    r = 1.0
    angle = random.uniform(0, 2*np.pi)
    goal = r * np.array((np.cos(angle), np.sin(angle)))
    return goal


GOAL_SAMPLERS = {
    'semi-circle': semi_circle_goal_sampler,
    'circle': circle_goal_sampler,
}


class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, max_episode_steps=100, goal_sampler=None):
        if callable(goal_sampler):
            self.goal_sampler = goal_sampler
        elif isinstance(goal_sampler, str):
            self.goal_sampler = GOAL_SAMPLERS[goal_sampler]
        elif goal_sampler is None:
            self.goal_sampler = semi_circle_goal_sampler
        else:
            raise NotImplementedError(goal_sampler)

        self.reset_task()
        self.task_dim = 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        # we convert the actions from [-1, 1] to [-0.1, 0.1] in the step() function
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self._max_episode_steps = max_episode_steps

    def sample_task(self):
        goal = self.goal_sampler()
        return goal

    def set_task(self, task):
        self._goal = task

    def get_task(self):
        return self._goal

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        return task

    def reset_model(self):
        self._state = np.zeros(2)
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):

        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action), action

        self._state = self._state + 0.1 * action
        reward = - np.linalg.norm(self._state - self._goal, ord=2)
        done = False
        ob = self._get_obs()
        info = {'task': self.get_task()}
        return ob, reward, done, info

    def visualise_behaviour(self,
                            env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            image_folder=None,
                            return_pos=False,
                            **kwargs,
                            ):

        num_episodes = args.max_rollouts_per_task

        # --- initialise things we want to keep track of ---

        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]

        episode_returns = []
        episode_lengths = []

        if encoder is not None:
            episode_latent_samples = [[] for _ in range(num_episodes)]
            episode_latent_means = [[] for _ in range(num_episodes)]
            episode_latent_logvars = [[] for _ in range(num_episodes)]
        else:
            episode_latent_samples = episode_latent_means = episode_latent_logvars = None

        # --- roll out policy ---

        # (re)set environment
        env.reset_task()
        state, belief, task = utl.reset_env(env, args)
        start_obs_raw = state.clone()
        task = task.view(-1) if task is not None else None

        # initialise actions and rewards (used as initial input to policy if we have a recurrent policy)
        if hasattr(args, 'hidden_size'):
            hidden_state = torch.zeros((1, args.hidden_size)).to(device)
        else:
            hidden_state = None

        # keep track of what task we're in and the position of the cheetah
        pos = [[] for _ in range(args.max_rollouts_per_task)]
        start_pos = state

        for episode_idx in range(num_episodes):

            curr_rollout_rew = []
            pos[episode_idx].append(start_pos[0])

            if episode_idx == 0:
                if encoder is not None:
                    # reset to prior
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                    curr_latent_sample = curr_latent_sample[0].to(device)
                    curr_latent_mean = curr_latent_mean[0].to(device)
                    curr_latent_logvar = curr_latent_logvar[0].to(device)
                else:
                    curr_latent_sample = curr_latent_mean = curr_latent_logvar = None

            if encoder is not None:
                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs_raw.clone())
                else:
                    episode_prev_obs[episode_idx].append(state.clone())
                # act
                latent = utl.get_latent_for_policy(args,
                                                   latent_sample=curr_latent_sample,
                                                   latent_mean=curr_latent_mean,
                                                   latent_logvar=curr_latent_logvar)
                _, action = policy.act(state=state.view(-1), latent=latent, belief=belief, task=task,
                                       deterministic=True)

                (state, belief, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                state = state.float().reshape((1, -1)).to(device)
                task = task.view(-1) if task is not None else None

                # keep track of position
                pos[episode_idx].append(state[0])

                if encoder is not None:
                    # update task embedding
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                        action.reshape(1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device),
                        hidden_state, return_prior=False)

                    episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.clone())

                if info[0]['done_mdp'] and not done:
                    start_obs_raw = info[0]['start_state']
                    start_obs_raw = torch.from_numpy(start_obs_raw).float().reshape((1, -1)).to(device)
                    start_pos = start_obs_raw
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        # clean up
        if encoder is not None:
            episode_latent_means = [torch.stack(e) for e in episode_latent_means]
            episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.stack(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        figsize = (5.5, 4)
        figure, axis = plt.subplots(1, 1, figsize=figsize)
        xlim = (-1.3, 1.3)
        if self.goal_sampler == semi_circle_goal_sampler:
            ylim = (-0.3, 1.3)
        else:
            ylim = (-1.3, 1.3)
        color_map = mpl.colors.ListedColormap(sns.color_palette("husl", num_episodes))

        observations = torch.stack([episode_prev_obs[i]for i in range(num_episodes)]).cpu().numpy()
        curr_task = env.get_task()

        # plot goal
        axis.scatter(*curr_task, marker='x', color='k', s=50)
        # radius where we get reward
        if hasattr(self, 'goal_radius'):
            circle1 = plt.Circle(curr_task, self.goal_radius, color='c', alpha=0.2, edgecolor='none')
            plt.gca().add_artist(circle1)

        for i in range(num_episodes):
            color = color_map(i)
            path = observations[i]

            # plot (semi-)circle
            r = 1.0
            if self.goal_sampler == semi_circle_goal_sampler:
                angle = np.linspace(0, np.pi, 100)
            else:
                angle = np.linspace(0, 2*np.pi, 100)
            goal_range = r * np.array((np.cos(angle), np.sin(angle)))
            plt.plot(goal_range[0], goal_range[1], 'k--', alpha=0.1)

            # plot trajectory
            axis.plot(path[:, 0], path[:, 1], '-', color=color, label=i)
            axis.scatter(*path[0, :2], marker='.', color=color, s=50)

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks([])
        plt.yticks([])
        plt.legend()
        plt.tight_layout()
        if image_folder is not None:
            plt.savefig('{}/{}_behaviour.png'.format(image_folder, iter_idx), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        plt_rew = [episode_rewards[i][:episode_lengths[i]] for i in range(len(episode_rewards))]
        plt.plot(torch.cat(plt_rew).view(-1).cpu().numpy())
        plt.xlabel('env step')
        plt.ylabel('reward per step')
        plt.tight_layout()
        if image_folder is not None:
            plt.savefig('{}/{}_rewards.png'.format(image_folder, iter_idx), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        if not return_pos:
            return episode_latent_means, episode_latent_logvars, \
                   episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns
        else:
            return episode_latent_means, episode_latent_logvars, \
                   episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns, pos


class SparsePointEnv(PointEnv):
    """ Reward is L2 distance given only within goal radius """

    def __init__(self, goal_radius=0.2, max_episode_steps=100, goal_sampler='semi-circle'):
        super().__init__(max_episode_steps=max_episode_steps, goal_sampler=goal_sampler)
        self.goal_radius = goal_radius
        self.reset_task()

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
        d.update({'dense_reward': reward})
        return ob, sparse_reward, done, d
