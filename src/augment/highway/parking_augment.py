import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from src.generate.utils import reset_data, append_data, load_dataset, npify
import time
import h5py


def init_trajectory():
    trajectory = {
        'actions': [],
        'desired_goal': [],
        'next_observations': [],
        'observations': [],
        'rewards': [],
        'terminals': []
    }

    return trajectory

def append_trajectory(trajectories, trajectory):
    for key in trajectory:
        for i in range(len(trajectory[key])):
            trajectories[key].append(trajectory[key][i])

def get_trajectories(dataset, start_timestamp, end_timestamp):
    trajectory = init_trajectory()

    for i in range(start_timestamp, end_timestamp + 1):
        for key in dataset:
            trajectory[key].append(dataset[key][i])
    return trajectory

def generate_aug_trajectory(trajectory):
    env = gym.make('parking-v0', render_mode='rgb_array')
    aug_trajectory = init_trajectory()
    n = len(trajectory['observations'])
    original_desired_goal = trajectory['desired_goal'][0]
    obs, _ = env.reset()
    new_desired_goal = trajectory['desired_goal'][0]

    for i in range(n):
        original_obs = trajectory['observations'][i]
        delta_x = original_obs[0] - original_desired_goal[0]
        delta_y = original_obs[1] - original_desired_goal[1]

        aug_obs = original_obs.copy()
        aug_obs[6:] = new_desired_goal.copy() 
        aug_obs[0] = new_desired_goal[0] + delta_x
        aug_obs[1] = new_desired_goal[1] + delta_y

        original_next_obs = trajectory['next_observations'][i].copy()    
        delta_x = original_next_obs[0] - original_desired_goal[0]
        delta_y = original_next_obs[1] - original_desired_goal[1]

        aug_next_obs = original_next_obs.copy()
        aug_next_obs[6:] = new_desired_goal.copy()
        aug_next_obs[0] = new_desired_goal[0] + delta_x
        aug_next_obs[1] = new_desired_goal[1] + delta_y
        
        ## TODO how to get achieved goal, reward, and terminal:
        achieved_goal = aug_obs[6:].copy()
        aug_reward = env.compute_reward(achieved_goal, new_desired_goal, {})
        aug_action = trajectory['actions'][i]
        aug_terminal = trajectory['terminals'][i]

        aug_trajectory['observations'].append(aug_obs)
        aug_trajectory['actions'].append(aug_action)
        aug_trajectory['next_observations'].append(aug_next_obs)
        aug_trajectory['rewards'].append(aug_reward)
        aug_trajectory['terminals'].append(aug_terminal)
        aug_trajectory['desired_goal'].append(new_desired_goal)
        aug_state = {
            'observation': aug_obs,
            'desired_goal': new_desired_goal,
            'achieved_goal': achieved_goal
        }
        env.set_state(aug_state)
        new_state, _, _, _, _ = env.step(aug_action)
        true_obs = new_state['observation']
        print(f"difference: {true_obs - aug_next_obs[:6]}")

    return aug_trajectory

dataset_path = f"/Users/yxqu/Desktop/Research/GuDA/GuidedDataAugmentationForRobotics/src/datasets/parking-v0/no_aug.hdf5"
observed_dataset = load_dataset(dataset_path)

n = len(observed_dataset['observations'])

start_timestamp = 0
aug_trajectories = init_trajectory()

for i in range(n):
    if observed_dataset['terminals'][i]:
        trajectory = get_trajectories(observed_dataset, start_timestamp, i)
        start_timestamp = i + 1
        aug_trajectory = generate_aug_trajectory(trajectory)
        append_trajectory(aug_trajectories, aug_trajectory)

dataset = h5py.File("tmp.hdf5", 'w')
for k in aug_trajectories:
    dataset.create_dataset(k, data=np.array(aug_trajectories[k]), compression='gzip')
