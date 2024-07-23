import torch
import numpy as np
import os
import sys
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy, MLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def main(args):
    ckpt_dir = args['ckpt_dir']

    policy_config = {'lr': args['lr'], 'num_queries': 1,}

    config = {
        'ckpt_dir': ckpt_dir,
        'real_robot': True,
        'policy_config': policy_config
    }

    ckpt_names = [f'policy_best.ckpt']
    results = []
    print(ckpt_names)
    for ckpt_name in ckpt_names:
        success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
        results.append([ckpt_name, success_rate, avg_return])
    
    for ckpt_name, success_rate, avg_return in results:
        print(f'{ckpt_name}: {success_rate=} {avg_return=}')
    print()
    exit()


def eval_bc(config, ckpt_name, save_episode=True):
    # load policy and stats
    ckpt_dir = config['ckpt_dir']
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy_config = config['policy_config']
    policy = make_policy(policy_class="MLP", policy_config=policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    #input = torch.Tensor([0.4755, -0.7302, 0.9956, 0.02915, 0.4464, -0.4694, 0.1614]) # 76
    input = torch.Tensor([1.077, 0.003068, 1.195, 0.1948, 0.2746, -0.5875, 0.01614]) # 664
    # input = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Random
    input = input.cuda()
    out = policy(qpos=input)
    print('output: ', out)
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    success_rate = 0
    avg_return=0

    return success_rate, avg_return

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'MLP':
        policy = MLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    # parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    # parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    # parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    # parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    # parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    # parser.add_argument('--temporal_agg', action='store_true')

    # my modifications
    # parser.add_argument('--no_cameras', action='store', type=bool, help='no_cameras', required=False)
    
    main(vars(parser.parse_args()))
