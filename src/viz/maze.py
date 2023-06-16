import d4rl
import gym
import h5py
import numpy as np
from matplotlib import pyplot as plt

from algorithms.utils import load_dataset


env_id = 'maze2d-umaze-v1'
env_id = 'maze2d-medium-v1'

env = gym.make(env_id)

dataset_name = None
# dataset_name = f"../datasets/{env_id}/no_aug.hdf5"
dataset_name = f'../datasets/{env_id}/random/m_1.hdf5'


for env_id in ['maze2d-umaze-v1', 'maze2d-medium-v1']:
    plt.figure(figsize=(12,12))
    for aug in ['no_aug', 'guided']:
        dataset_name = f'../datasets/{env_id}/guided/m_1.hdf5'

        dataset = {}
        if dataset_name:
            # local dataset
            data_hdf5 = h5py.File(dataset_name, "r")
            for key in data_hdf5.keys():
                dataset[key] = np.array(data_hdf5[key])
        else:
            # remote dataset
            dataset = d4rl.qlearning_dataset(env)
        # dataset = d4rl.qlearning_dataset(env, dataset=dataset)

        if aug == 'no_aug':
            start = int(0e6)
            end = start + int(5e3)
        else:
            start = len(dataset['observations']) - int(5e3)
            end = start + int(5e3)
        observations = dataset['observations'][start:end]
        next_observations = dataset['next_observations'][start:end]
        rewards = dataset['rewards'][start:end]
        at_goal = rewards > 0

        x = observations[:, 0]
        y = observations[:, 1]
        next_x = next_observations[:, 0]
        next_y = next_observations[:, 1]

        if aug == 'no_aug':
            plt.scatter(x, y, alpha=0.5)
        else:
            u = next_x - x
            v = next_y - y
            norm = np.sqrt(u**2 + v**2)
            u /= norm
            v /= norm

            plt.quiver(x, y, u, v)
        plt.scatter(x[at_goal],y[at_goal], alpha=0.3, color='g')

    plt.title(f'{env_id}')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.show()

    plt.figure(figsize=(12, 12))
    for aug in ['no_aug', 'random']:
        dataset_name = f'../datasets/{env_id}/random/m_1.hdf5'

        dataset = {}
        if dataset_name:
            # local dataset
            data_hdf5 = h5py.File(dataset_name, "r")
            for key in data_hdf5.keys():
                dataset[key] = np.array(data_hdf5[key])
        else:
            # remote dataset
            dataset = d4rl.qlearning_dataset(env)
        # dataset = d4rl.qlearning_dataset(env, dataset=dataset)

        if aug == 'no_aug':
            start = int(0e6)
            end = start + int(5e3)
        else:
            start = len(dataset['observations']) - int(5e3)
            end = start + int(5e3)
        observations = dataset['observations'][start:end]
        next_observations = dataset['next_observations'][start:end]
        rewards = dataset['rewards'][start:end]
        at_goal = rewards > 0


        x = observations[:, 0]
        y = observations[:, 1]
        next_x = next_observations[:, 0]
        next_y = next_observations[:, 1]

        if aug == 'no_aug':
            plt.scatter(x, y, alpha=0.5)
        else:
            u = next_x - x
            v = next_y - y
            norm = np.sqrt(u ** 2 + v ** 2)
            u /= norm
            v /= norm

            plt.quiver(x, y, u, v)
        plt.scatter(x[at_goal], y[at_goal], alpha=0.3, color='g')

    plt.title(f'{env_id}')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.show()