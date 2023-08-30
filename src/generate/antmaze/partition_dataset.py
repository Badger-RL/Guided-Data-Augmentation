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
from generate.utils import reset_data, append_data, npify


def xy_to_rowcol(xy):
    size_scaling = 4
    return (int(np.round(1 + (xy[0]) / size_scaling)),
            int(np.round(1 + (xy[1]) / size_scaling)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', '-fd', type=str, default='tmp')
    parser.add_argument('--save-name', '-fn', type=str, default='tmp.hdf5')
    args = parser.parse_args()

    # f = RotateReflectTranslate(env=None)


    # env = gym.make('maze2d-umaze-v0')
    env = gym.make('antmaze-umaze-diverse-v1')
    dataset = d4rl.qlearning_dataset(env)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    env.reset()

    partitions = defaultdict(lambda: reset_data())
    # for location in [
    #     (1,3), (2,3), (3,3),
    #     (1,2),
    #     (1,1), (2,1), (3,1),
    # ]:
    #     partitions
    #     parition = reset_data()

    dataset_obs = dataset['observations']
    dataset_action = dataset['actions']
    dataset_next_obs = dataset['next_observations']
    dataset_reward = dataset['rewards']
    dataset_done = dataset['terminals']

    count = 0

    for t in range(len(dataset['observations'])):
        obs = dataset['observations'][t]
        action = dataset['actions'][t]

        qpos = obs[:15]
        qvel = obs[15:]
        env.set_state(qpos, qvel)
        next_obs, reward, done, info = env.step(action)
        # env.render()

        rowcol = xy_to_rowcol(obs[:2])
        if rowcol == (3,2):
            append_data(partitions[rowcol],
                        dataset_obs[t],
                        dataset_action[t],
                        dataset_reward[t],
                        dataset_next_obs[t],
                        dataset_done[t])
            count += 1
            if count == 5000:
                break
        if t % 1000 == 0:
            print(f'{count}/{t}')



    os.makedirs(args.save_dir, exist_ok=True)

    for location, partition in partitions.items():
        save_path = f'{args.save_dir}/{location[0]}_{location[1]}.hdf5'
        new_dataset = h5py.File(save_path, 'w')
        npify(partition)
        for k in partition:
            new_dataset.create_dataset(k, data=partition[k], compression='gzip')
        print(f"{location} dataset size: {len(new_dataset['observations'])}")

if __name__ == '__main__':
    main()