import os

import d4rl
import gym
import h5py
import numpy as np
from matplotlib import pyplot as plt

from algorithms.utils import load_dataset


# for env_id in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']:
for env_id in ['maze2d-umaze-v1']:

    for aug in ['random', 'guided']:
        plt.figure(figsize=(12, 12))

        dataset_name = f'../datasets_good/{env_id}/{aug}/m_1.hdf5'

        # local dataset
        dataset = {}
        data_hdf5 = h5py.File(dataset_name, "r")
        for key in data_hdf5.keys():
            dataset[key] = np.array(data_hdf5[key])

        # # plot no_aug
        # start = int(0e6)
        # end = start + int(5e3)
        #
        # observations = dataset['observations'][start:end]
        # next_observations = dataset['next_observations'][start:end]
        # rewards = dataset['rewards'][start:end]
        # at_goal = rewards > 0
        #
        # x = observations[:, 0]
        # y = observations[:, 1]
        # plt.scatter(x, y, alpha=0.5)
        # plt.scatter(x[at_goal], y[at_goal], alpha=0.5, color='g')

        # plot aug
        start = len(dataset['observations']) - int(1e3)
        end = start + int(1e3)
        observations = dataset['observations'][start:end]
        next_observations = dataset['next_observations'][start:end]
        rewards = dataset['rewards'][start:end]
        at_goal = rewards > 0

        x = observations[:, 0]
        y = observations[:, 1]
        next_x = next_observations[:, 0]
        next_y = next_observations[:, 1]

        u = next_x - x
        v = next_y - y
        # u *= 2
        # v *= 2
        # norm = np.sqrt(u**2 + v**2)
        # u /= norm
        # v /= norm

        plt.quiver(x, y, u, v, scale=3, width=0.005)
        # plt.scatter(next_x[at_goal],next_y[at_goal], alpha=0.5, color='g')
        plt.title(f'{env_id}\n{aug.capitalize()} Augmentation', fontsize=56)
        plt.xlabel('x position', fontsize=56)
        plt.ylabel('y position', fontsize=56)
        plt.xticks(fontsize=48)
        plt.yticks(fontsize=48)
        plt.tight_layout()
        os.makedirs(f'figures/{env_id}', exist_ok=True)
        plt.savefig(f'figures/{env_id}/{aug}.png')
        plt.show()
