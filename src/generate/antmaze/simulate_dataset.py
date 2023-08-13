# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import argparse
import os
import time

import gym, d4rl
import h5py
import numpy as np
import random

from augment.maze.point_maze_aug_function import PointMazeAugmentationFunction
from generate.utils import reset_data, append_data, npify, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', '-fd', type=str, default='tmp')
    parser.add_argument('--save-name', '-fn', type=str, default='tmp.hdf5')
    args = parser.parse_args()

    # f = RotateReflectTranslate(env=None)


    # env = gym.make('maze2d-umaze-v0')
    env = gym.make('antmaze-umaze-diverse-v2')
    dataset = d4rl.qlearning_dataset(env)
    # dataset = load_dataset('../../datasets/antmaze-umaze-diverse-v1/no_aug_no_collisions_relabeled_1k.hdf5')
    # dataset = load_dataset('tmp/tmp.hdf5')

    print('terminals', dataset['terminals'].sum())
    t = dataset['terminals'][:10000000]
    o = dataset['observations'][:10000000]
    r = dataset['rewards'][:10000000]

    overlap = t & (r > 0)
    print('overlap', overlap.sum())

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    env.reset()
    t = 0
    obs = dataset['observations'][t]
    action = dataset['actions'][t]

    qpos = obs[:15]
    qvel = obs[15:]
    env.set_state(qpos, qvel)
    mask = dataset['rewards'] > 0
    print(mask.sum())

    count = 0
    # for t in range(12100, len(dataset['observations'])):
    for t in range(0, len(dataset['observations'])):

        print(t)
        # env.reset()
        # if obs[3] < 0.25:
        #     continue
        obs = dataset['observations'][t]
        action = dataset['actions'][t]


        qpos = obs[:15]
        qvel = obs[15:]
        env.set_state(qpos, qvel)
        next_obs, reward, done, info = env.step(action)
        env.render()

        true_next_obs = dataset['next_observations'][t]
        # print(next_obs-true_next_obs)
        # print(dataset['truncateds'][t])

        # print(rowcol)
        if reward > 0:
            count += 1
            # print(count)
            stop = 0

if __name__ == '__main__':
    main()