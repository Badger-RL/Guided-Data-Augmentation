import numpy as np
import pandas as pd
from src.generate.utils import reset_data, append_data, load_dataset, npify
# import gym
# from src.augment.maze.point_maze_aug_function import PointMazeTrajectoryAugmentationFunction, PointMazeGuidedTrajectoryAugmentationFunction
from src.augment.antmaze.antmaze_aug_function import AntMazeTrajectoryGuidedAugmentationFunction
import gym
import h5py
import gzip
from src.generate.utils import npify

env_id = 'antmaze-large-diverse-v1'
dataset_path = "/Users/yxqu/Desktop/Research/GuDA/Antmaze_Dataset/antmaze-large-diverse-v1/no_aug_no_collisions_relabeled.hdf5"
select_trajectories_save_path = f"{env_id}_reward.hdf5"
observed_dataset = load_dataset(dataset_path)

## save original trajectories
select_trajectories = {
    'observations': [],
    'actions': [],
    'next_observations': [],
    'rewards': [],
    'terminals': []
}

for i in range(len(observed_dataset['observations'])):
    if observed_dataset['rewards'][i] > 0:
        select_trajectories['observations'].append(observed_dataset['observations'][i])
        select_trajectories['actions'].append(observed_dataset['actions'][i])
        select_trajectories['next_observations'].append(observed_dataset['next_observations'][i])
        select_trajectories['rewards'].append(observed_dataset['rewards'][i])
        select_trajectories['terminals'].append(observed_dataset['terminals'][i])

print("number of transitions: ", len(select_trajectories['observations']))
select_trajectory_dataset = h5py.File(select_trajectories_save_path, 'w')

npify(select_trajectories)
for k in select_trajectories:
    select_trajectory_dataset.create_dataset(k, data=select_trajectories[k], compression='gzip')
select_trajectory_dataset.close()