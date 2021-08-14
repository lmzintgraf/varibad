"""
Script to roll out a trained agent on the environment for several episodes.
"""
import json
import os
import re
import argparse

import numpy as np
import torch

from utils import helpers as utl
from utils.evaluation import evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def evaluate_varibad(env_name,
                     method_name,
                     num_episodes,
                     rollouts_per_seed,
                     recompute_results,  # recompute or load from disk
                     model_path, 
                     config_path
                     ):

    # check if we already evaluated this; in that case just load from disk
    #precomputed_results_path = os.path.join(exp_directory, 'results', 'end_performance_per_episode')
    precomputed_results_path = 'final_performance_per_episode'
    if not os.path.exists(precomputed_results_path):
        os.mkdir(precomputed_results_path)
    precomputed_results_file = os.path.join(precomputed_results_path, f'{method_name}-{env_name}')
    if os.path.exists(precomputed_results_file + '.npy') and not recompute_results:
        return np.load(precomputed_results_file + '.npy')

    # the folder for this environment and this method
    #exp_directory = os.path.join(exp_directory, env_name, method_name,
    #                              os.listdir(os.path.join(exp_directory, env_name, method_name))[-1])
    exp_directory = os.path.join(model_path, os.listdir(model_path)[-1])

    results = []
    # loop through different seeds
    for r_fold in os.listdir(exp_directory):
        if r_fold[0] == '.':
            continue

        # this is the current results folder we're in
        results_path = os.path.join(exp_directory, r_fold)

        # get config file
        #with open(os.path.join(results_path, 'config.json')) as json_data_file:
        config_path = os.path.join(config_path, os.listdir(config_path)[-1])
        config_path = os.path.join(config_path, os.listdir(config_path)[-1])
        with open(os.path.join(config_path, 'config.json')) as json_data_file:
            config = json.load(json_data_file)
            config = Bunch(config)

            # TODO: remove again, this is a hack for CheetahDir
            if env_name == 'cheetah_dir':
                config.env_name = 'HalfCheetahDir-v0'
            elif env_name == 'cheetah_hop':
                config.env_name = 'Hop-v0'

        # get the latest model
        model_path = os.path.join(exp_directory, r_fold, 'models')
        print('Loading latest model from run ', model_path)
        model_files = os.listdir(model_path)
        try:
            model_idx = max([int(''.join(re.findall(r'[0-9]', m))) for m in model_files])
            print('loadig model policy{}.pt'.format(model_idx))
        except ValueError:
            model_idx = ''

        # get policy network
        policy = torch.load(os.path.join(results_path, 'models', 'policy{}.pt'.format(model_idx)), map_location=device)

        # try to get encoder
        try:
            encoder = torch.load(os.path.join(results_path, 'models', 'encoder{}.pt'.format(model_idx)), map_location=device)
        except FileNotFoundError:
            encoder = None

        # get the normalisation parameters for the environment
        try:
            ret_rms = utl.load_obj(os.path.join(results_path, 'models'), "env_rew_rms{0}".format(model_idx))
        except FileNotFoundError:
            ret_rms = None

        returns = run_policy(config, policy, ret_rms, encoder, num_episodes, rollouts_per_seed)
        print(returns)

        # add the returns of the current experiment!
        results.append(returns.cpu())

    # shape: num_seeds * num_episodes
    results = np.stack(results)

    # save the results so we don't have to recompute them every time
    np.save(precomputed_results_file, results)

    return results


def run_policy(config, policy, ret_rms, encoder, num_episodes, rollouts_per_seed):
    avg_return_per_episode = 0
    for i in range(rollouts_per_seed):
        returns_per_episode = evaluate(config, policy, ret_rms, iter_idx=i, tasks=None, num_episodes=num_episodes, encoder=encoder)
        avg_return_per_episode += returns_per_episode.mean(dim=0)
    avg_return_per_episode /= rollouts_per_seed

    return avg_return_per_episode



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--method', type=str, default='varibad')
    parser.add_argument('--num_episodes', type=int, default=3)
    parser.add_argument('--num_evaluation', type=int, default=8)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    evaluate_varibad(env_name=args.env,
                     method_name=args.method,
                     num_episodes=args.num_episodes,
                     rollouts_per_seed=args.num_evaluation,
                     recompute_results=True, 
                     model_path=args.model_path, 
                     config_path=args.config_path
                     )