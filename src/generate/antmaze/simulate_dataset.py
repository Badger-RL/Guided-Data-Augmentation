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
    env = gym.make('antmaze-medium-diverse-v1')
    # dataset = load_dataset('../../datasets/antmaze-umaze-diverse-v1/guided/m_1.hdf5')
    # dataset = load_dataset('tmp/tmp.hdf5')

    dataset = d4rl.qlearning_dataset(env)

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
    for t in range(len(dataset['observations'])):
        # if obs[3] < 0.25:
        #     continue
        obs = dataset['observations'][t]
        action = dataset['actions'][t]


        qpos = obs[:15]
        qvel = obs[15:]
        env.set_state(qpos, qvel)
        next_obs, reward, done, info = env.step(action)
        env.render()

        # print(rowcol)
        if reward > 0:
            count += 1
            print(count)
            stop = 0

if __name__ == '__main__':
    main()