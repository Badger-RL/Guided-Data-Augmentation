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


for env_id in ['antmaze-medium-diverse-v1']:
    plt.figure(figsize=(12, 12))

    dataset_name = f'../datasets/{env_id}/no_aug_relabeled.hdf5'
    dataset = {}
    data_hdf5 = h5py.File(dataset_name, "r")
    for key in data_hdf5.keys():
        dataset[key] = np.array(data_hdf5[key])

    # env = gym.make(env_id)
    # dataset = d4rl.qlearning_dataset(env)

    n = int(100e3)

        # plot no_aug
    start = int(0e6)
    end = start + n

    observations = dataset['observations'][start:end]
    next_observations = dataset['next_observations'][start:end]
    rewards = dataset['rewards'][start:end]
    at_goal = rewards > 0
    print(at_goal.sum())

    x = observations[:, 0]
    y = observations[:, 1]
    plt.scatter(x, y, alpha=0.5)
    plt.scatter(x[at_goal], y[at_goal], alpha=0.5, color='g')

    sin = observations[:, 3]
    cos = observations[:, 6]
    orientation = 2*np.arctan2(sin, cos)
    delta = next_observations[:,:2] - observations[:,:2]

    for cellx in range(6):
        for celly in range(6):
            xlo = 4*cellx-2
            xhi = 4*cellx+2
            ylo = 4*celly-2
            yhi = 4*celly+2

            mask = (x > xlo) & (x < xhi) & ( y > ylo) & (y < yhi)
            # cell = observations[:, mask]

            partition_orientation = orientation[mask]
            partition_delta = delta[mask]

            # print(cellx, celly, np.average(parition_orientation)*180/np.pi)
            v = np.average(partition_delta, axis=0)
            phi = np.average(partition_orientation)*180/np.pi
            print(cellx, celly, v, phi)
            plt.quiver(4*cellx, 4*celly, v[0], v[1])


    if 'umaze' in env_id:
        r = 4
        c = 4
    elif 'medium' in env_id:
        r = 7
        c = 7
    elif 'large' in env_id:
        r = 8
        c = 10

    for k in range(r):
        plt.axhline(y=k*4-2)
    for k in range(c):
        plt.axvline(x=k*4-2)

    plt.show()
