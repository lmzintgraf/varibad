import argparse

import torch

from utils.cli import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # --- GENERAL ---

    # training parameters
    parser.add_argument('--num_frames', type=int, default=1e8, help='number of frames to train')
    parser.add_argument('--max_rollouts_per_task', type=int, default=4)

    # RL2
    parser.add_argument('--exp_label', default='rl2', help='label for the experiment')
    parser.add_argument('--disable_varibad', type=boolean_argument, default=False,
                        help='Train policy w/o variBAD architecture')
    parser.add_argument('--disable_decoder', type=boolean_argument, default=True)
    parser.add_argument('--disable_stochasticity_in_latent', type=boolean_argument, default=True)
    parser.add_argument('--rlloss_through_encoder', type=boolean_argument, default=True,
                        help='backprop rl loss through encoder')
    parser.add_argument('--latent_dim', type=int, default=32, help='dimensionality of latent space')
    parser.add_argument('--condition_policy_on_state', type=boolean_argument, default=False,
                        help='Train a normal FF policy without the variBAD architecture')

    # env
    parser.add_argument('--env_name', default='GridNavi-v0', help='environment to train on')
    parser.add_argument('--norm_obs_for_policy', type=boolean_argument, default=False)
    parser.add_argument('--norm_rew_for_policy', type=boolean_argument, default=False)
    parser.add_argument('--normalise_actions', type=boolean_argument, default=False, help='output normalised actions')

    # --- POLICY ---

    # network
    parser.add_argument('--policy_layers', nargs='+', default=[32])
    parser.add_argument('--policy_activation_function', type=str, default='tanh')

    # algo
    parser.add_argument('--policy', type=str, default='a2c', help='choose: a2c, ppo, optimal, oracle')

    # a2c specific
    parser.add_argument('--a2c_alpha', type=float, default=0.99, help='RMSprop optimizer alpha (default: 0.99)')

    # other hyperparameters
    parser.add_argument('--lr_policy', type=float, default=0.0007, help='learning rate (default: 7e-4)')
    parser.add_argument('--lr_vae', type=float, default=0.0007, help='learning rate (default: 7e-4)')
    parser.add_argument('--policy_num_steps', type=int, default=60,
                        help='number of env steps to do (per process) before updating (for A2C ~ 10, for PPO ~100)')
    parser.add_argument('--policy_eps', type=float, default=1e-5, help='optimizer epsilon (1e-8 for ppo, 1e-5 for a2c)')
    parser.add_argument('--policy_init_std', type=float, default=1.0)
    parser.add_argument('--policy_value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--policy_entropy_coef', type=float, default=0.1,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--policy_gamma', type=float, default=0.95, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--policy_use_gae', type=boolean_argument, default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--policy_tau', type=float, default=0.95, help='gae parameter (default: 0.95)')
    parser.add_argument('--use_proper_time_limits', type=boolean_argument, default=False)
    parser.add_argument('--policy_max_grad_norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--precollect_len', type=int, default=0,
                        help='how many frames to pre-collect before training begins')

    # --- VAE ---

    # we keep the VAE buffer only to keep track of the current trajectory
    parser.add_argument('--size_vae_buffer', type=int, default=0, help='how many trajectories to keep in VAE buffer')
    parser.add_argument('--vae_buffer_add_thresh', type=float, default=0, help='prob of adding a new traj to buffer')

    # - encoder (now part of the policy!)
    parser.add_argument('--aggregator_hidden_size', type=int, default=128,
                        help='dimensionality of hidden state of the rnn')
    parser.add_argument('--layers_before_aggregator', nargs='+', type=int, default=[])
    parser.add_argument('--layers_after_aggregator', nargs='+', type=int, default=[])
    parser.add_argument('--state_embedding_size', type=int, default=32)
    parser.add_argument('--action_embedding_size', type=int, default=0)
    parser.add_argument('--reward_embedding_size', type=int, default=8)

    # decoder: rewards
    parser.add_argument('--decode_reward', type=boolean_argument, default=False, help='use reward decoder')
    parser.add_argument('--reward_loss_coeff', type=float, default=1.0, help='weight for rew loss')

    # decoder: state transitions
    parser.add_argument('--decode_state', type=boolean_argument, default=False, help='use state decoder')
    parser.add_argument('--state_loss_coeff', type=float, default=1.0, help='weight for state loss')

    # decoder: ground-truth task ("varibad oracle", after Humplik et al. 2019)
    parser.add_argument('--decode_task', type=boolean_argument, default=False, help='use state decoder')
    parser.add_argument('--task_loss_coeff', type=float, default=1.0, help='weight for task loss')

    # --- ABLATIONS ---

    parser.add_argument('--sample_embeddings', type=boolean_argument, default=False,
                        help='sample the embedding (otherwise: pass mean)')
    parser.add_argument('--pretrain_len', type=int, default=0)
    parser.add_argument('--vae_loss_coeff', type=float, default=1.0, help='weight for VAE loss (vs RL loss)')

    # --- OTHERS ---

    # logging, saving, evaluation
    parser.add_argument('--log_interval', type=int, default=500,
                        help='log interval, one log per n updates (default: 500)')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='save interval, one save per n updates (default: 1000)')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='eval interval, one eval per n updates (default: 1000)')
    parser.add_argument('--vis_interval', type=int, default=500,
                        help='visualisation interval, one eval per n updates (default: None)')
    parser.add_argument('--agent_log_dir', default='/tmp/gym/', help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--results_log_dir', default=None, help='directory to save agent logs (default: ./data)')

    # general settings
    parser.add_argument('--seed', type=int, default=73, help='random seed (default: 73)')
    parser.add_argument('--deterministic_execution', type=boolean_argument, default=False,
                        help='Make code fully deterministic. Expects 1 process and uses deterministic CUDNN')
    parser.add_argument('--num_processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--port', type=int, default=8097, help='port to run the server on (default: 8097)')
    args = parser.parse_args(rest_args)

    args.cuda = torch.cuda.is_available()

    return args
