import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RolloutStorageVAE(object):
    def __init__(self, num_processes, max_trajectory_len, zero_pad, max_num_rollouts,
                 obs_dim, action_dim, vae_buffer_add_thresh, action_embedding_size,
                 rew_dim=1):
        """
        Store everything that is needed for the VAE update
        :param num_processes:
        """

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_embedding_size = action_embedding_size
        self.rew_dim = rew_dim
        self.vae_buffer_add_thresh = vae_buffer_add_thresh

        # count the number of datapoints seen so far (so we can do reservoir sampling)
        self.num_rollouts_seen = 0
        self.max_num_rollouts = max_num_rollouts
        self.max_traj_len = max_trajectory_len

        # whether to zero-pad to maximum length (zero's at the end!)
        self.zero_pad = zero_pad

        # buffers for completed rollouts (stored on CPU)
        self.prev_obs = torch.zeros(0, )
        self.next_obs = torch.zeros(0, )
        self.actions = torch.zeros(0, )
        self.rewards = torch.zeros(0, )
        self.tasks = torch.zeros(0, )
        self.trajectory_lens = []

        # storage for each running process (stored on GPU)
        self.num_processes = num_processes
        self.running_prev_obs = [[] for _ in range(num_processes)]  # for each episode will have obs 0...N-1
        self.running_next_obs = [[] for _ in range(num_processes)]  # for each episode will have obs 1...N
        self.running_rewards = [[] for _ in range(num_processes)]
        self.running_actions = [[] for _ in range(num_processes)]
        self.running_tasks = [None for _ in range(num_processes)]

    def get_running_batch(self):
        """
        Returns the batch of data from the current running environments
        (zero-padded to maximal trajectory length since different processes can have different trajectory lengths)
        :return:
        """
        prev_obs = []
        next_obs = []
        actions = []
        rewards = []
        lengths = []
        for i in range(self.num_processes):
            pad_len = self.max_traj_len - len(self.running_prev_obs[i])
            if pad_len > 0 and len(self.running_prev_obs[i]) > 0:
                prev_obs.append(torch.cat((torch.cat(self.running_prev_obs[i]),
                                           torch.zeros(pad_len, self.running_prev_obs[i][0].shape[1]).to(device))))
                next_obs.append(torch.cat((torch.cat(self.running_next_obs[i]),
                                           torch.zeros(pad_len, self.running_next_obs[i][0].shape[1]).to(device))))
                actions.append(torch.cat((torch.cat(self.running_actions[i]),
                                          torch.zeros(pad_len, self.running_actions[i][0].shape[1]).to(device))))
                rewards.append(torch.cat((torch.cat(self.running_rewards[i]),
                                          torch.zeros(pad_len, self.running_rewards[i][0].shape[1]).to(device))))
            else:
                prev_obs.append(torch.zeros(pad_len, self.obs_dim).to(device))
                next_obs.append(torch.zeros(pad_len, self.obs_dim).to(device))
                actions.append(torch.zeros(pad_len, self.action_dim).to(device))
                rewards.append(torch.zeros(pad_len, self.rew_dim).to(device))
            lengths.append(len(self.running_prev_obs[i]))
        return torch.stack(prev_obs, dim=1), \
               torch.stack(next_obs, dim=1), \
               torch.stack(actions, dim=1), \
               torch.stack(rewards, dim=1), \
               lengths

    def insert(self, prev_obs, actions, next_obs, rewards, reset_task, task):

        actions = actions
        for i in range(self.num_processes):
            actions[i] = actions[i]
            self.running_prev_obs[i].append(prev_obs[i].unsqueeze(0))
            self.running_next_obs[i].append(next_obs[i].unsqueeze(0))
            self.running_rewards[i].append(rewards[i].unsqueeze(0))
            self.running_actions[i].append(actions[i].unsqueeze(0).float())

            if self.running_tasks[i] is None:
                self.running_tasks[i] = task[i]

            # if we are at the end of a task, dump the data into the larger buffer
            if reset_task[i]:

                # count up the number of trajectories we saw to completion
                self.num_rollouts_seen += 1

                self.running_prev_obs[i] = torch.cat(self.running_prev_obs[i]).to('cpu')
                self.running_next_obs[i] = torch.cat(self.running_next_obs[i]).to('cpu')
                self.running_actions[i] = torch.cat(self.running_actions[i]).to('cpu')
                self.running_rewards[i] = torch.cat(self.running_rewards[i]).to('cpu')
                self.running_tasks[i] = self.running_tasks[i].to('cpu')

                # pad with zeros before inserting
                if self.zero_pad:
                    traj_len = len(self.running_prev_obs[i])  # length of the current trajectory
                    pad_len = self.max_traj_len - traj_len  # how much we want to pad
                    if pad_len > 0:
                        self.running_prev_obs[i] = torch.cat(
                            (self.running_prev_obs[i], torch.zeros(pad_len, self.running_prev_obs[i].shape[1])))
                        self.running_next_obs[i] = torch.cat(
                            (self.running_next_obs[i], torch.zeros(pad_len, self.running_next_obs[i].shape[1])))
                        self.running_actions[i] = torch.cat(
                            (self.running_actions[i], torch.zeros(pad_len, self.running_actions[i].shape[1])))
                        self.running_rewards[i] = torch.cat(
                            (self.running_rewards[i], torch.zeros(pad_len, self.running_rewards[i].shape[1])))

                # get the length of the current trajectory
                traj_len = len(self.running_prev_obs[i])

                # insert trajectory into buffer
                # (we're collecting number of trajectories along dim=1,
                #  and the trajectory length is along dim=0
                #  to match the RNN interface of pytorch)
                if self.num_rollouts_seen > self.max_num_rollouts:
                    if self.prev_obs.shape[1] == np.random.uniform(0, 1):
                        insert_index = (self.num_rollouts_seen - 1) % self.max_num_rollouts
                        # insert by replacing oldest rollout in buffer
                        self.prev_obs[:, insert_index] = self.running_prev_obs[i]
                        self.next_obs[:, insert_index] = self.running_next_obs[i]
                        self.actions[:, insert_index] = self.running_actions[i]
                        self.rewards[:, insert_index] = self.running_rewards[i]
                        self.tasks[insert_index] = self.running_tasks[i]
                        self.trajectory_lens[insert_index] = traj_len
                else:
                    self.prev_obs = torch.cat((self.prev_obs, self.running_prev_obs[i].unsqueeze(1)), dim=1)
                    self.next_obs = torch.cat((self.next_obs, self.running_next_obs[i].unsqueeze(1)), dim=1)
                    self.rewards = torch.cat((self.rewards, self.running_rewards[i].unsqueeze(1)), dim=1)
                    self.actions = torch.cat((self.actions, self.running_actions[i].unsqueeze(1)), dim=1)
                    self.tasks = torch.cat((self.tasks, self.running_tasks[i].unsqueeze(0)))
                    self.trajectory_lens.append(traj_len)

                # empty running buffer
                self.running_prev_obs[i] = []
                self.running_next_obs[i] = []
                self.running_rewards[i] = []
                self.running_actions[i] = []
                self.running_tasks[i] = None

    def ready_for_update(self):
        return len(self) > 0

    def __len__(self):
        return len(self.trajectory_lens)

    def get_batch(self, num_enc_len=1, num_rollouts=5, include_final=True, replace=False):

        rollouts_in_buffer = len(self.trajectory_lens)

        # select the indices for the processes from which we pick
        rollout_indices = np.random.choice(range(rollouts_in_buffer), min(rollouts_in_buffer, num_rollouts),
                                           replace=replace)

        # trajectory length of the individual rollouts we picked
        trajectory_lens = np.array(self.trajectory_lens)[rollout_indices]

        # select the rollouts we want
        prev_obs = self.prev_obs[:, rollout_indices, :]
        next_obs = self.next_obs[:, rollout_indices, :]
        actions = self.actions[:, rollout_indices, :]
        rewards = self.rewards[:, rollout_indices, :]
        tasks = self.tasks[rollout_indices]

        # choose where to chop up the trajectories
        if num_enc_len is not None:
            len_encoder = np.stack(
                [np.random.choice(range(0, t + int(include_final)), num_enc_len, replace=False) for t in
                 trajectory_lens])
        else:
            len_encoder = np.stack([range(0, t + 1) for t in trajectory_lens])

        return prev_obs.to(device), next_obs.to(device), actions.to(device), \
               rewards.to(device), tasks.to(device), len_encoder, trajectory_lens
