import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

# from augment.highway.parking_augment_transition import aug_middle
from src.generate.utils import reset_data, append_data, load_dataset, npify
import time
import h5py
from src.augment.highway.highway import ChangeLaneAllVehicles

def init_trajectory():
    trajectory = {
        'actions': [],
        # 'desired_goal': [],
        'next_observations': [],
        'observations': [],
        'rewards': [],
        'terminals': []
    }

    return trajectory

env_name = 'highway-v0'

dataset_path = f"../../datasets/{env_name}/no_aug.hdf5"
env = gym.make(f'{env_name}', render_mode='rgb_array')
observed_dataset = load_dataset(dataset_path)

n = len(observed_dataset['observations'])

aug_transitions= init_trajectory()

F = ChangeLaneAllVehicles(env)


max_aug = 1e2
aug_count = 0

while aug_count < max_aug:
    start_timestamp = 0
    for i in range(n):
        obs = observed_dataset['observations'][i]
        action = observed_dataset['actions'][i]
        next_obs = observed_dataset['next_observations'][i]
        reward = observed_dataset['rewards'][i]
        done = observed_dataset['terminals'][i]
        aug_obs, aug_action, aug_reward, aug_next_obs, aug_done = F.augment(obs, action, next_obs, reward, done)
        aug_transitions['observations'].append(aug_obs)
        aug_transitions['actions'].append(aug_action)
        aug_transitions['rewards'].append(aug_reward)
        aug_transitions['next_observations'].append(aug_next_obs)
        aug_transitions['terminals'].append(aug_done)
        aug_count += 1
        if aug_count >= max_aug:
            break
    

dataset = h5py.File(f"../../datasets/{env_name}/aug.hdf5", 'w')
for k in aug_transitions:
    aug_transitions[k] = np.concatenate(aug_transitions[k])
    print(f'{k}: {aug_transitions[k].shape}')
    dataset.create_dataset(k, data=np.array(aug_transitions[k]), compression='gzip')



