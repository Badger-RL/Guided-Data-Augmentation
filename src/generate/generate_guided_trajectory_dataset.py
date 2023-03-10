import os

import gym
import numpy as np
import h5py
import argparse

from augment.rotate_reflect_trajectory import rotate_reflect_traj
from augment.translate_reflect_trajectory import translate_reflect_traj_y
from augment.utils import check_valid
import custom_envs

def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def load_observed_data(dataset_path):
    observed_data_hdf5 = h5py.File(f"{dataset_path}", "r")

    observed_dataset = {}
    for key in observed_data_hdf5.keys():
        observed_dataset[key] = np.array(observed_data_hdf5[key])

    return observed_dataset

def gen_aug_dataset(env, observed_dataset, aug_ratio=1):

    obs = observed_dataset['observations']
    action = observed_dataset['actions']
    reward = observed_dataset['rewards']
    next_obs = observed_dataset['next_observations']
    done = observed_dataset['terminals']

    aug_obs_list, aug_action_list, aug_reward_list, aug_next_obs_list, aug_done_list = [], [], [], [], []

    count = 0
    while count < aug_ratio:
        for aug_function in [rotate_reflect_traj, translate_reflect_traj_y]:
            aug_obs, aug_action, aug_reward, aug_next_obs, aug_done = aug_function(obs, action, next_obs, reward, done)
            is_valid = check_valid(env, aug_obs, aug_action, aug_reward, aug_next_obs)

            if is_valid:
                count += 1
                aug_obs_list.append(aug_obs)
                aug_action_list.append(aug_action)
                aug_reward_list.append(aug_reward)
                aug_next_obs_list.append(aug_next_obs)
                aug_done_list.append(aug_done)
            else:
                print('Invalid augmented trajectory')

    aug_obs = np.concatenate(aug_obs_list)
    aug_action = np.concatenate(aug_action_list)
    aug_reward = np.concatenate(aug_reward_list)
    aug_next_obs = np.concatenate(aug_next_obs_list)
    aug_done = np.concatenate(aug_done_list)

    aug_dataset = {
        'observations': aug_obs,
        'actions': aug_action,
        'terminals': aug_done,
        'rewards': aug_reward,
        'next_observations': aug_next_obs,
    }
    npify(aug_dataset)

    return aug_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--aug-ratio', type=int, default=10, help='Number of augmented trajectories to generate')
    parser.add_argument('--observed-dataset-path', type=str, default=f'../datasets/expert/trajectories/1.hdf5', help='path to observed trajectory dataset')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save augmented dataset')
    parser.add_argument('--save-name', type=str, default=None, help='Name of augmented dataset')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    env = gym.make('PushBallToGoal-v0')

    observed_dataset = load_observed_data(dataset_path=args.observed_dataset_path)
    aug_dataset = gen_aug_dataset(env, observed_dataset, aug_ratio=args.aug_ratio)

    os.makedirs(args.save_dir, exist_ok=True)
    fname = f'{args.save_dir}/{args.save_name}'
    new_dataset = h5py.File(fname, 'w')

    for k in aug_dataset:
        observed = observed_dataset[k]
        aug = np.array(aug_dataset[k])
        data = np.concatenate([observed, aug])
        new_dataset.create_dataset(k, data=data, compression='gzip')



