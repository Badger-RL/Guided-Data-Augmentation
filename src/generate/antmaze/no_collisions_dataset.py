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
from generate.utils import reset_data, append_data, npify, append_data2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='antmaze-medium-diverse-v1')
    parser.add_argument('--save-dir', '-fd', type=str, default='../../datasets/antmaze-medium-diverse-v1')
    parser.add_argument('--save-name', '-fn', type=str, default='no_aug_no_collisions.hdf5')
    args = parser.parse_args()

    # f = RotateReflectTranslate(env=None)


    # env = gym.make('maze2d-medium-v0')
    env = gym.make('antmaze-umaze-diverse-v1')
    env_empty = gym.make('antmaze-open-umaze-diverse-v1')

    dataset = d4rl.qlearning_dataset(env)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    env.reset()
    env_empty.reset()

    clean_dataset = reset_data()

    dataset_obs = dataset['observations']
    dataset_action = dataset['actions']
    dataset_next_obs = dataset['next_observations']
    dataset_reward = dataset['rewards']
    dataset_done = dataset['terminals']

    clean_count = 0

    for t in range(0,len(dataset['observations'])):
        # if t == 10000: break
        env.reset()
        env_empty.reset()

        obs = dataset['observations'][t]
        action = dataset['actions'][t]

        qpos = obs[:15]
        qvel = obs[15:]
        env.set_state(qpos, qvel)
        next_obs, reward, done, info = env.step(action)
        # env.render()

        env_empty.set_state(qpos, qvel)
        next_obs_empty, reward, done, info = env_empty.step(action)
        # env_empty.render()

        if np.allclose(next_obs, next_obs_empty) and next_obs[2] > 0.35:
            append_data2(clean_dataset,
                        dataset_obs[t],
                        dataset_action[t],
                        dataset_reward[t],
                        dataset_next_obs[t],
                        dataset_done[t],
                        truncated=False)

            clean_count +=1
            if clean_count % 100 == 0:
                print(f'{clean_count} / {t+1}')
        else:
            clean_dataset['truncateds'][clean_count-1] = True
        # if reward > 0:
        #     clean_dataset['truncateds'][clean_count-1] = True


        # diff = next_obs - next_obs_empty
        # print(diff)
        # assert np.allclose(next_obs, dataset['next_observations'][t], atol=1e-2)
        # assert np.allclose(reward, dataset['rewards'][t])


    os.makedirs(args.save_dir, exist_ok=True)
    save_path = f'{args.save_dir}/{args.save_name}'
    new_dataset = h5py.File(save_path, 'w')
    npify(clean_dataset)
    for k in clean_dataset:
        new_dataset.create_dataset(k, data=clean_dataset[k], compression='gzip')
    print(f"New dataset size: {len(new_dataset['observations'])}")

if __name__ == '__main__':
    main()