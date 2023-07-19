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


def xy_to_rowcol(xy):
    size_scaling = 4
    return (int(np.round(1 + (xy[0]) / size_scaling)),
            int(np.round(1 + (xy[1]) / size_scaling)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', '-fd', type=str, default='../../datasets/antmaze-umaze-diverse-v1')
    parser.add_argument('--save-name', '-fn', type=str, default='no_aug_no_collisions_relabeled_clean.hdf5')
    args = parser.parse_args()

    # f = RotateReflectTranslate(env=None)


    # env = gym.make('maze2d-umaze-v0')
    env = gym.make('antmaze-umaze-diverse-v1')
    # dataset = d4rl.qlearning_dataset(env)
    dataset = load_dataset('../../datasets/antmaze-umaze-diverse-v1/no_aug_no_collisions_relabeled.hdf5')
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    env.reset()

    dataset_obs = dataset['observations']
    dataset_action = dataset['actions']
    dataset_next_obs = dataset['next_observations']
    dataset_reward = dataset['rewards']
    dataset_done = dataset['terminals']

    dataset_dict = reset_data()
    count = 0

    for t in range(len(dataset['observations'])):
        # if t > 10000: break
        obs = dataset_obs[t]
        next_obs = dataset_next_obs[t]

        delta_obs = next_obs - obs
        phi = 2*np.arctan2(delta_obs[3], delta_obs[6])
        theta = 2*np.arctan2(delta_obs[1], delta_obs[0])
        if phi < 0:
            phi += 2*np.pi
        if theta < 0:
            theta += 2*np.pi

        if np.abs(phi-theta) < np.pi/6:
            append_data(dataset_dict,
                        dataset_obs[t],
                        dataset_action[t],
                        dataset_reward[t],
                        dataset_next_obs[t],
                        dataset_done[t])
            count += 1
        if t % 1000 == 0:
            print(f'{count}/{t}')



    os.makedirs(args.save_dir, exist_ok=True)

    save_path = f'{args.save_dir}/{args.save_name}'
    new_dataset = h5py.File(save_path, 'w')
    npify(dataset_dict)
    for key, data in dataset_dict.items():
        new_dataset.create_dataset(key, data=data, compression='gzip')
    print(f"New dataset size: {len(new_dataset['observations'])}")

if __name__ == '__main__':
    main()