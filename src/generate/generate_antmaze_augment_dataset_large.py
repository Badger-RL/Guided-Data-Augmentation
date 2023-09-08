import os

import numpy as np
import pandas as pd

import d4rl
from src.generate.utils import reset_data, append_data, load_dataset, npify
# import gym
# from src.augment.maze.point_maze_aug_function import PointMazeTrajectoryAugmentationFunction, PointMazeGuidedTrajectoryAugmentationFunction
from src.augment.antmaze.antmaze_aug_function import AntMazeTrajectoryGuidedAugmentationFunction, \
    AntMazeTrajectoryRandomAugmentationFunction
import gym
import h5py
import gzip
from src.generate.utils import npify

timestamps = {
    'antmaze-umaze-diverse-v1': {
        'n': int(1e6),
    },
    'antmaze-medium-diverse-v1': {
        'n': int(1e6)
    },
    'antmaze-large-diverse-v1': {
        'n': int(1e6)
    }
}

for maze in ['umaze',]:
    for aug in ['guided']:
        for demo_size in [10e3]:
            aug_size = int(100e3)
            demo_size = int(demo_size)

            env_id = f'antmaze-{maze}-diverse-v1'
            d4rl_dataset_path = f"../datasets/antmaze-{maze}-diverse-v1/no_aug_relabeled.hdf5"
            to_aug_dataset_path = f"../datasets/antmaze-{maze}-diverse-v1/no_aug_no_collisions_relabeled.hdf5"

            save_dir = f'../datasets/{env_id}/s_{int(demo_size / 1e3)}k'
            aug_save_path = f"{save_dir}/{aug}.hdf5"
            os.makedirs(save_dir, exist_ok=True)

            env = gym.make(env_id)
            d4rl_dataset = load_dataset(d4rl_dataset_path)
            to_aug_dataset = load_dataset(to_aug_dataset_path)
            for k, v in to_aug_dataset.items():
                to_aug_dataset[k] = v[:demo_size]

            n = len(to_aug_dataset['rewards'])

            subtraj_len = 30
            start_timestamps = np.arange(0, n - subtraj_len, subtraj_len)
            end_timestamps = np.arange(subtraj_len, n, subtraj_len)
            directions = np.zeros_like(start_timestamps)

            num_of_trajectories = len(start_timestamps)

            ## save generated dataset
            if aug == 'guided':
                f = AntMazeTrajectoryGuidedAugmentationFunction(env=env)
            if aug == 'random':
                f = AntMazeTrajectoryRandomAugmentationFunction(env=env)
            env.reset()

            augmented_trajectories = {
                'observations': [],
                'actions': [],
                'next_observations': [],
                'rewards': [],
                'terminals': []
            }

            size = 0
            while True:
                for i in range(num_of_trajectories):
                    start_timestamp = start_timestamps[i]
                    end_timestamp = end_timestamps[i]
                    trajectory = {
                        'observations': to_aug_dataset['observations'][start_timestamp:end_timestamp],
                        'actions': to_aug_dataset['actions'][start_timestamp:end_timestamp],
                        'next_observations': to_aug_dataset['next_observations'][start_timestamp:end_timestamp],
                        'rewards': to_aug_dataset['rewards'][start_timestamp:end_timestamp],
                        'terminals': to_aug_dataset['terminals'][start_timestamp:end_timestamp]
                    }
                    augmented_trajectory = f.augment_trajectory(trajectory, -1)
                    for key in augmented_trajectory:
                        for j in range(len(augmented_trajectory[key])):
                            augmented_trajectories[key].append(augmented_trajectory[key][j])
                    size += len(augmented_trajectory['observations'])
                    print(size)

                    if size > aug_size:
                        break
                if size > aug_size:
                    break
                print(size)

            augmented_trajectory_dataset = h5py.File(aug_save_path, 'w')

            npify(augmented_trajectories)
            for k in augmented_trajectories:
                data = np.concatenate([d4rl_dataset[k][:demo_size], augmented_trajectories[k]])
                augmented_trajectory_dataset.create_dataset(k, data=data, compression='gzip')
            augmented_trajectory_dataset.close()
