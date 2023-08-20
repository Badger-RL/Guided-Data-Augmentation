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

dataset_path = "../datasets/antmaze-umaze-diverse-v1/no_aug_no_collisions_relabeled.hdf5"
env_id = 'antmaze-umaze-diverse-v1'
select_trajectories_save_path = "original.hdf5"
generate_trajectories_save_path = "generated.hdf5"
generate_num_of_transitions = 100000
start_timestamps = [560, 1135, 1570, 3185]
end_timestamps = [600, 1240, 1650, 3290]
# start_timestamps = []
# end_timestamps = []
# add 500 more subtrajectories. I think most of these show the ant walk around randomly near the goal.
# I didn't render these, so I don't know for sure what they look like.
start_timestamps += [5000 + 100*i for i in range(5)]
end_timestamps += [5100 + 100*i for i in range(5)]

# start_timestamps += [101000 + 100*i for i in range(5)]
# end_timestamps += [10100 + 100*i for i in range(5)]

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
# f = AntMazeTrajectoryRandomAugmentationFunction(env=env)
f = AntMazeTrajectoryGuidedAugmentationFunction(env=env)
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

        augmented_trajectory = f.augment_trajectory(trajectory)
        for key in augmented_trajectory:
            for j in range(len(augmented_trajectory[key])):
                augmented_trajectories[key].append(augmented_trajectory[key][j])
        size += len(augmented_trajectory['observations'])

    if size > generate_num_of_transitions:
        break
    print(size)

augmented_trajectory_dataset = h5py.File(generate_trajectories_save_path, 'w')

npify(augmented_trajectories)
for k in augmented_trajectories:
    data = np.concatenate([select_trajectories[k], augmented_trajectories[k]])
    augmented_trajectory_dataset.create_dataset(k, data=data, compression='gzip')
augmented_trajectory_dataset.close()
