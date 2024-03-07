import os

import d4rl
import gym
import h5py
import numpy as np
from matplotlib import pyplot as plt



for env_id in ['antmaze-umaze-diverse-v1']:
    dataset_name = f'{env_id}/mocoda.hdf5'

    # local dataset
    dataset = {}
    data_hdf5 = h5py.File(dataset_name, "r")
    for key in data_hdf5.keys():
        dataset[key] = np.array(data_hdf5[key])

    # plot no_aug


    observations = dataset['observations']
    next_observations = dataset['next_observations']
    actions = dataset['actions']
    # rewards = dataset['rewards'][start:end]

    env = gym.make(env_id)
    n = len(observations)

    err_sum = 0
    err_pos_sum = 0

    for i in range(n):
        _ = env.reset()
        qpos = observations[i, :15]
        qvel = observations[i, 15:]
        env.set_state(qpos, qvel)
        # env.set_marker()

        next_obs, reward, done, info = env.step(actions[i])

        err_sum += np.linalg.norm(next_obs - next_observations[i])
        err_pos_sum += np.linalg.norm(next_obs[:2] - next_observations[i, :2])

        if i % 10000 == 0:
            print(f'{i}: {err_sum/(i+1)}')
            print(f'{i}: {err_pos_sum/(i+1)}')
            print()
