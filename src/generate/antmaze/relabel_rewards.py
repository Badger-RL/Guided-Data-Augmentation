# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import argparse
import os
import time
from collections import defaultdict

import gym, d4rl
import h5py
import numpy as np
import random

from augment.maze.point_maze_aug_function import PointMazeAugmentationFunction
from generate.utils import reset_data, append_data, npify, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='antmaze-umaze-diverse-v1')
    # parser.add_argument('--dataset-path', type=str, default='../../datasets/antmaze-umaze-diverse-v1/no_aug.hdf5')
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--save-dir', '-fd', type=str, default='../../datasets/antmaze-umaze-diverse-v1')
    parser.add_argument('--save-name', '-fn', type=str, default='no_aug.hdf5')
    args = parser.parse_args()

    if args.dataset_path:
        dataset = load_dataset(args.dataset_path)
    else:
        env = gym.make(args.env_id)
        dataset = d4rl.qlearning_dataset(env)

    dataset_obs = dataset['observations']
    dataset_action = dataset['actions']
    dataset_next_obs = dataset['next_observations']
    dataset_reward = dataset['rewards']
    dataset_done = dataset['terminals']

    goal = np.array([0.75, 8.5])
    goal = np.array([20.5, 20.5])
    goal = np.array([32.5, 24.5])

    print(np.sum(dataset_done))

    for t in range(len(dataset['observations'])):
        next_obs = dataset['next_observations'][t]
        dist = np.linalg.norm(next_obs[:2] - goal)

        if dist < 0.5:
            reward = 1
            done = True
        else:
            reward = 0

            if not(next_obs[2] >= 0.2 and next_obs[2] <= 1.0):
                done = True
            else:
                done = False


        dataset['rewards'][t] = reward
        dataset['terminals'][t] = done

        if t % 1000 == 0:
            print(f'{t}')



    os.makedirs(args.save_dir, exist_ok=True)

    save_path = f'{args.save_dir}/{args.save_name}'
    new_dataset = h5py.File(save_path, 'w')
    npify(dataset)
    for key, data, in dataset.items():
        new_dataset.create_dataset(key, data=data, compression='gzip')
    print(f"New dataset size: {len(new_dataset['observations'])}")
    print(f"Reward sum: {np.sum(new_dataset['rewards'])}")


if __name__ == '__main__':
    main()