import numpy as np
import pandas as pd
from src.generate.utils import reset_data, append_data, load_dataset, npify
# import gym
from src.augment.maze.point_maze_aug_function import PointMazeGuidedTrajectoryAugmentationFunction
# from src.augment.antmaze.antmaze_aug_function import AntMazeTrajectoryAugmentationFunction, AntMazeGuidedTrajectoryAugmentationFunction
import gym
import h5py
import gzip
from src.generate.utils import npify

dataset_path = "/Users/yxqu/Desktop/Research/GuDA/D4RL/scripts/generation/pointmaze_small/maze2d-umaze-v1-sparse.hdf5"
env_id = 'maze2d-umaze-v1'
select_trajectories_save_path = "/Users/yxqu/Desktop/Research/GuDA/GuidedDataAugmentationForRobotics_new/src/datasets/maze2d/original.hdf5"
generate_trajectories_save_path = "/Users/yxqu/Desktop/Research/GuDA/GuidedDataAugmentationForRobotics_new/src/datasets/maze2d/generated.hdf5"
generate_num_of_transitions = 1000000
start_timestamps = [490, 2870, 6100, 9750]
end_timestamps = [670, 3050, 6280, 9930]

observed_dataset = load_dataset(dataset_path)
num_of_trajectories = len(start_timestamps)

## save original trajectories
select_trajectories = {
    'observations': [],
    'actions': [],
    'next_observations': [],
    'rewards': [],
    'terminals': []
}

for i in range(num_of_trajectories):
    start_timestamp = start_timestamps[i]
    end_timestamp = end_timestamps[i]
    trajectory = {
        'observations': observed_dataset['observations'][start_timestamp:end_timestamp],
        'actions': observed_dataset['actions'][start_timestamp:end_timestamp],
        'next_observations': observed_dataset['next_observations'][start_timestamp:end_timestamp],
        'rewards': observed_dataset['rewards'][start_timestamp:end_timestamp],
        'terminals': observed_dataset['terminals'][start_timestamp:end_timestamp]
    }

    for key in trajectory:
        for j in range(len(trajectory[key])):
            select_trajectories[key].append(trajectory[key][j])

select_trajectory_dataset = h5py.File(select_trajectories_save_path, 'w')

npify(select_trajectories)
for k in select_trajectories:
    select_trajectory_dataset.create_dataset(k, data=select_trajectories[k], compression='gzip')
select_trajectory_dataset.close()


## save generated dataset
env = gym.make(env_id)
f = PointMazeGuidedTrajectoryAugmentationFunction(env=env)
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
            'observations': observed_dataset['observations'][start_timestamp:end_timestamp],
            'actions': observed_dataset['actions'][start_timestamp:end_timestamp],
            'next_observations': observed_dataset['next_observations'][start_timestamp:end_timestamp],
            'rewards': observed_dataset['rewards'][start_timestamp:end_timestamp],
            'terminals': observed_dataset['terminals'][start_timestamp:end_timestamp]
        }

        augmented_trajectory = f.augment_trajectory1(trajectory)
        for key in augmented_trajectory:
            for j in range(len(augmented_trajectory[key])):
                augmented_trajectories[key].append(augmented_trajectory[key][j])
        size += len(augmented_trajectory['observations'])

    if size > generate_num_of_transitions:
        break

augmented_trajectory_dataset = h5py.File(generate_trajectories_save_path, 'w')

npify(augmented_trajectories)
for k in augmented_trajectories:
    data = np.concatenate([select_trajectories[k], augmented_trajectories[k]])
    augmented_trajectory_dataset.create_dataset(k, data=data, compression='gzip')
augmented_trajectory_dataset.close()
