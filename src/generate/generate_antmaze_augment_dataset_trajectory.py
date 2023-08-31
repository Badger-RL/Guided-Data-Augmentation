import numpy as np
import pandas as pd
from src.generate.utils import reset_data, append_data, load_dataset, npify
# import gym
# from src.augment.maze.point_maze_aug_function import PointMazeTrajectoryAugmentationFunction, PointMazeGuidedTrajectoryAugmentationFunction
from src.augment.antmaze.antmaze_aug_function import AntMazeTrajectoryGuidedAugmentationFunction, \
    AntMazeTrajectoryRandomAugmentationFunction
import gym
import h5py
import gzip
from src.generate.utils import npify

# 0: move randomly
# 1: →
# 2: ↑
# 3: ←
# 4: ↓

timestamps = {
    'antmaze-umaze-diverse-v1': {
        'start': [560, 599, 1135, 1181, 1570, 1620, 3185, 3216, 5108, 5140, 6075, 6990, 7080, 7125, 7181, 7231, 7820, 8500, 8531, 8580, 8631, 9480, 11260, 11301, 11960, 13015, 13100, 13151, 13920, 13956, 14001, 14129, 14171, 14230, 14256, 14311, 15085, 15106, 17880, 19515, 19541, 19601, 19660],
        'end': [598, 670, 1180, 1240, 1620, 1650, 3215, 3290, 5140, 5190, 6100, 7020, 7120, 7180, 7230, 7300, 7860, 8530, 8579, 8630, 8700, 9509, 11300, 11350, 12000, 13075, 13150, 13210, 13955, 14000, 14100, 14170, 14205, 14255, 14310,14400, 15105, 15160, 17945, 19540, 19600, 19659, 19800],
        "direction": [3, 0, 1, 2, 3, 0, 1, 2, 2, 3, 3, 2, 1, 2, 3, 0, 3, 1, 2, 3, 0, 2, 2, 3, 3, 1, 2, 3, 2, 3, 0, 1, 2, 2, 3, 0, 2, 3, 2, 1, 2, 3, 0],
        'n': int(1e6),
    },
    'antmaze-medium-diverse-v1': {
        "start": [0, 350, 406, 471, 510, 551, 710, 731, 751, 781, 2085, 2141, 2171, 2201, 2360, 2396, 2436, 2461, 4500, 4531, 4551, 4586, 4611, 4644, 4686, 4718, 4751, 4771, 4801, 4846, 4871, 10620, 10641, 10671, 10731, 10781, 15460, 15476, 15501, 23450, 23506, 23526, 23740, 23761, 23816, 25100, 25126, 25206, 25271, 26080, 26101, 26181, 26191, 26206, 26246, 26276, 27000, 27016, 27040, 27116, 27146, 27186, 27774, 27806, 28750, 28775, 28811, 28851, 28901, 28926, 28966, 29610, 29631, 29701, 29756],
        "end": [50, 405, 470, 510, 550, 600, 730, 750, 780, 830, 2140, 2170, 2200, 2250, 2395, 2435, 2460, 2500, 4530, 4550, 4585, 4610, 4643, 4685, 4717, 4750, 4770, 4800, 4845, 4870, 4900, 10640, 10670, 10730, 10780, 10820, 15475, 15500, 15520, 23505, 23525, 23600, 23760, 23815, 23850, 25125, 25205, 25270, 25350, 26100, 26180, 26190, 26205, 26245, 26275, 26300, 27015, 27040, 27115, 27145, 27185, 27200, 27805, 27850, 28774, 28810, 28850, 28900, 28925, 28965, 29000, 29630, 29700, 29755, 29780],
        "direction": [1, 1, 0, 2, 3, 0, 2, 2, 1, 2, 1, 0, 2, 1, 1, 2, 3, 0, 1, 2, 1, 0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 1, 4, 1, 2, 3, 2, 1, 2, 0, 1, 0, 2, 3, 0, 2, 1, 2, 3, 4, 1, 2, 4, 2, 3, 0, 2, 1, 2, 3, 2, 0, 4, 0, 1, 2, 0, 1, 4, 1, 0, 2, 1, 2, 3],
        'n': int(1e6)
    },
    'antmaze-large-diverse-v1': {
        "start": [3000, 3026, 4000, 4995, 5046, 5305, 5335, 6320, 6460, 7310, 7376, 7451, 7531, 7581, 7651, 8310, 8351, 8421, 9305, 9331, 9720, 9741, 9791, 9846, 9916, 9951, 10720, 10771, 10826, 11110, 11231, 13100, 13151, 13216, 13351, 13410, 13436, 14110, 14281, 14321, 15110, 15151, 15211, 15271, 15371, 18100, 18211, 19100, 19201, 19241, 19261, 19301, 19476, 19526, 19576, 19616, 19651, 19671, 20150, 20196, 20251, 20371, 21180, 21286, 24300, 24351, 24406, 24436, 38050, 38116, 38236, 39040, 39061, 39106],
        "end": [3025, 3050, 4050, 5045, 5100, 5335, 5500, 6460, 6500, 7375, 7450, 7530, 7580, 7650, 7710, 8350, 8420, 8450, 9330, 9450, 9740, 9790, 9845, 9915, 9950, 10000, 10770, 10825, 10870, 11230, 11300, 13150, 13215, 13350, 13410, 13435, 13500, 14280, 14320, 14350, 15150, 15210, 15270, 15370, 15450, 18210, 18250, 19200, 19240, 19260, 19300, 19475, 19525, 19575, 19615, 19650, 19670, 19800, 20195, 20250, 20370, 20450, 21285, 21340, 24350, 24405, 24435, 24470, 38115, 38235, 38300, 39060, 39105, 39180],
        "direction": [2, 3, 4, 3, 2, 1, 0, 3, 2, 1, 2, 1, 4, 1, 2, 3, 2, 3, 2, 3, 3, 4, 3, 2, 3, 0, 2, 1, 4, 0, 1, 3, 2, 3, 2, 1, 0, 1, 4, 0, 1, 2, 1, 2, 3, 1, 0, 1, 0, 1, 2, 0, 1, 0, 2, 0, 1, 0, 2, 1, 4, 0, 2, 1, 3, 2, 1, 0, 4, 3, 0, 2, 0, 1],
        'n': int(1e6)
    }
}

for maze in ['umaze']:
    for aug in ['guided', 'random']:
        env_id = f'antmaze-{maze}-diverse-v1'
        generate_num_of_transitions = timestamps[env_id]["n"]
        dataset_path = f"../datasets/antmaze-{maze}-diverse-v1/no_aug_no_collisions_relabeled.hdf5"
        select_trajectories_save_path = f"../datasets/{env_id}/no_aug.hdf5"
        generate_trajectories_save_path = f"../datasets/{env_id}/{aug}.hdf5"

        start_timestamps = timestamps[env_id]['start']
        end_timestamps = timestamps[env_id]['end']
        directions = timestamps[env_id]['direction']

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
                    'observations': observed_dataset['observations'][start_timestamp:end_timestamp],
                    'actions': observed_dataset['actions'][start_timestamp:end_timestamp],
                    'next_observations': observed_dataset['next_observations'][start_timestamp:end_timestamp],
                    'rewards': observed_dataset['rewards'][start_timestamp:end_timestamp],
                    'terminals': observed_dataset['terminals'][start_timestamp:end_timestamp]
                }
                augmented_trajectory = f.augment_trajectory(trajectory, directions[i])
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
