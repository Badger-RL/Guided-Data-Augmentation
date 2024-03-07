import os

import d4rl
import gym
import h5py
import numpy as np
from matplotlib import pyplot as plt



dataset_name = f'/Users/nicholascorrado/code/mocoda/datasets/maze2d-umaze-v1/mocoda.hdf5'
# dataset_name = f'/Users/nicholascorrado/code/mocoda/datasets/antmaze-large-diverse-v1/mocoda.hdf5'
dataset_name = f'maze2d-large-v1/mocoda.hdf5'

# local dataset
dataset = {}
data_hdf5 = h5py.File(dataset_name, "r")
for key in data_hdf5.keys():
    dataset[key] = np.array(data_hdf5[key])

# plot no_aug
start = int(0)
end = start + int(40000)

observations = dataset['observations'][start:end]
next_observations = dataset['next_observations'][start:end]
actions = dataset['actions'][start:end]
# rewards = dataset['rewards'][start:end]

env = gym.make('maze2d-umaze-v1')
n = len(observations)

err_sum = 0
for i in range(n):
    _ = env.reset()
    qpos = observations[i, :2]
    qvel = observations[i, 2:]
    env.set_state(qpos, qvel)
    env.set_marker()

    next_obs, reward, done, info = env.step(actions[i])

    err_sum += np.linalg.norm(next_obs - next_observations[i])
    print(err_sum/(i+1))
