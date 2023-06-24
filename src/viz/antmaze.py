import d4rl
import gym
import h5py
import numpy as np
from matplotlib import pyplot as plt

import os

import d4rl
import gym
import h5py
import numpy as np
from matplotlib import pyplot as plt

from algorithms.utils import load_dataset


for env_id in ['antmaze-umaze-diverse-v1', 'antmaze-medium-diverse-v1', 'antmaze-large-diverse-v1']:
    for aug in ['guided']:
        plt.figure(figsize=(12, 12))

        dataset_name = f'../datasets/{env_id}/{aug}/m_1.hdf5'
        # dataset_name = None
        # local dataset
        dataset = {}
        if dataset_name:
            data_hdf5 = h5py.File(dataset_name, "r")
            for key in data_hdf5.keys():
                dataset[key] = np.array(data_hdf5[key])
        else:
            env = gym.make(env_id)
            dataset = d4rl.qlearning_dataset(env)

        # plot no_aug
        start = int(0e6)
        end = start + int(20e3)

        observations = dataset['observations'][start:end]
        next_observations = dataset['next_observations'][start:end]
        rewards = dataset['rewards'][start:end]
        at_goal = rewards > 0

        x = observations[:, 0]
        y = observations[:, 1]
        plt.scatter(x, y, alpha=0.5)
        plt.scatter(x[at_goal], y[at_goal], alpha=0.5, color='g')

        # plot aug
        start = len(dataset['observations']) - int(3e3)
        end = start + int(3e3)
        observations = dataset['observations'][start:end] + 0
        next_observations = dataset['next_observations'][start:end] + 0
        rewards = dataset['rewards'][start:end]
        at_goal = rewards > 0

        x = observations[:, 0]
        y = observations[:, 1]
        next_x = next_observations[:, 0]
        next_y = next_observations[:, 1]

        u = next_x - x
        v = next_y - y
        norm = np.sqrt(u**2 + v**2)
        u /= norm
        v /= norm

        plt.quiver(x, y, u, v)
        plt.scatter(next_x[at_goal],next_y[at_goal], alpha=0.5, color='g')

        for k in range(12):
            plt.axhline(y=k*4-2)
            plt.axvline(x=k*4-2)

        plt.title(f'{env_id}: {aug}', fontsize=24)
        plt.xlabel('x position', fontsize=24)
        plt.ylabel('y position', fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.show()
        # os.makedirs(f'figures/{env_id}', exist_ok=True)
        # plt.savefig(f'figures/{env_id}/{aug}.png')
