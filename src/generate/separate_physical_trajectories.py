import os

import gym, custom_envs
import numpy as np
import h5py
import argparse

from stable_baselines3.common.utils import set_random_seed

from generate.utils import npify


def load_observed_data(dataset_path):
    observed_data_hdf5 = h5py.File(f"{dataset_path}", "r")

    observed_dataset = {}
    for key in observed_data_hdf5.keys():
        observed_dataset[key] = np.array(observed_data_hdf5[key])

    return observed_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--aug-ratio', type=int, default=10, help='Number of augmented trajectories to generate')
    parser.add_argument('--observed-dataset-path', type=str, default=f'../export/trajectories.hdf5', help='path to observed trajectory dataset')
    parser.add_argument('--save-dir', type=str, default='../datasets/expert/trajectories_physical/', help='Directory to save augmented dataset')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_random_seed(args.seed)

    env = gym.make('PushBallToGoal-v0')

    observed_dataset = load_observed_data(dataset_path=args.observed_dataset_path)

    obs = observed_dataset['observations']
    action = observed_dataset['actions']
    reward = observed_dataset['rewards']
    next_obs = observed_dataset['next_observations']
    done = observed_dataset['terminals']

    terminal_indices = np.where(done)[0]

    prev_terminal_idx = 0
    i = 0
    for terminal_idx in terminal_indices:
        obs_i = obs[prev_terminal_idx:terminal_idx]
        action_i = action[prev_terminal_idx:terminal_idx]
        reward_i = reward[prev_terminal_idx:terminal_idx]
        next_obs_i = next_obs[prev_terminal_idx:terminal_idx]
        done_i = done[prev_terminal_idx:terminal_idx]

        print(reward_i)

        new_dataset = {
            'observations': obs_i,
            'actions': action_i,
            'rewards': reward_i,
            'next_observations': next_obs_i,
            'terminals': done_i,
        }
        npify(new_dataset)

        os.makedirs(args.save_dir, exist_ok=True)
        fname = f'{args.save_dir}/{i}.hdf5'
        new_dataset_hdf5 = h5py.File(fname, 'w')
        for k in new_dataset:
            data = np.array(new_dataset[k])
            new_dataset_hdf5.create_dataset(k, data=data, compression='gzip')

        print(f"New dataset size: {new_dataset_hdf5['observations'].shape[0]}")
        prev_terminal_idx = terminal_idx
        i += 1


