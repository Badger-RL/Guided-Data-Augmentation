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

timestamps = {
    # 'antmaze-umaze-diverse-v1': {
    #     'start': [560, 1135, 1570, 3185] + [5000 + 100*i for i in range(5)],
    #     'end': [600, 1240, 1650, 3290] + [5100 + 100*i for i in range(5)],
    #     'n': int(10e3)
    # },
    'antmaze-umaze-diverse-v1': {
        'start': [560, 1135, 1570, 3185] + [5000 + 100 * i for i in range(5)],
        'end': [600, 1240, 1650, 3290] + [5100 + 100 * i for i in range(5)],
        'n': int(100e3)
    },
    'antmaze-medium-diverse-v1': {
        "start": [0, 350, 710, 2085, 2360, 4500, 10620, 15460, 23450, 23740, 25100, 26080, 27000, 27770, 28750, 29610],
        "end": [50, 600, 810, 2250, 2500, 4900, 10820, 15520, 23600, 23850, 25350, 26300, 27200, 27850, 29000, 29780],
        'n': int(400e3)
    },
    'antmaze-large-diverse-v1': {
        "start": [3000, 4000, 4990, 5305, 6320, 7310, 8310, 9305, 9720, 10720, 11110, 13100, 14110, 15110, 18100, 19100, 20150, 21180, 24300, 38050, 39040],
        "end": [3050, 4050, 5100, 5500, 6500, 7710, 8450, 9450, 10000, 10870, 11300, 13500, 14350, 15450, 18250, 19700, 20400, 21340, 24470, 38250, 39180],
        'n': int(800e3)
    }
}
maze = 'umaze'
env_id = f'antmaze-{maze}-diverse-v1'
dataset_path = f"../datasets/antmaze-{maze}-diverse-v1/no_aug_no_collisions_relabeled.hdf5"
select_trajectories_save_path = f"../datasets/{env_id}/no_aug.hdf5"
generate_trajectories_save_path = f"../datasets/{env_id}/guided.hdf5"
generate_num_of_transitions = timestamps[env_id]["n"]

start_timestamps = timestamps[env_id]['start']
end_timestamps = timestamps[env_id]['end']

if len(start_timestamps) != len(end_timestamps):
    print("Error: start_timestamps and end_timestamps must have the same length")
    exit()

for i in range(len(start_timestamps)):
    if start_timestamps[i] >= end_timestamps[i]:
        print("Error: start_timestamps must be less than end_timestamps")
        exit()

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
print("number of transitions: ", len(select_trajectories['observations']))
select_trajectory_dataset = h5py.File(select_trajectories_save_path, 'w')

npify(select_trajectories)
for k in select_trajectories:
    select_trajectory_dataset.create_dataset(k, data=select_trajectories[k], compression='gzip')
select_trajectory_dataset.close()
if generate_num_of_transitions == 0:
    exit()


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
