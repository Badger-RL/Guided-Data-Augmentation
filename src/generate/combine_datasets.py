import os
import gym
import h5py
import numpy as np

import d4rl
from generate.augment_dataset import AUG_FUNCTIONS
from generate.utils import reset_data, load_dataset
from simulate_cql import append_data


def fetch_dataset(
        env_id,
        save_dir=None,
        save_name=None
):
    if save_dir is None:
        save_dir = f'../datasets/{env_id}'
    if save_name is None:
        save_name = f'no_aug.hdf5'

    env = gym.make(env_id)
    dataset1 = d4rl.qlearning_dataset(env)
    dataset2 = d4rl.qlearning_dataset(env)


    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{save_name}'
    new_dataset = h5py.File(save_path, 'w')
    for k in new_dataset:
        data = np.concatenate([dataset1[k], dataset2[k]])
        new_dataset.create_dataset(k, data=data, compression='gzip')
    print(f"Dataset size: {len(new_dataset['observations'])}")

if __name__ == '__main__':

    env_id = 'PushBallToGoal-v0'
    save_dir = '../datasets/PushBallToGoal-v0/simrobot'
    save_name = 'no_aug.hdf5'
    # if save_dir is None:
    #     save_dir = f'../datasets/{env_id}'
    # if save_name is None:
    #     save_name = f'no_aug.hdf5'

    env = gym.make(env_id)
    dataset1 = load_dataset(f'{save_dir}/no_aug_1.hdf5')
    dataset2 = load_dataset(f'{save_dir}/no_aug_3.hdf5')

    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{save_name}'
    new_dataset = h5py.File(save_path, 'w')
    for k in dataset1.keys():
        data = np.concatenate([dataset1[k], dataset2[k]])
        new_dataset.create_dataset(k, data=data, compression='gzip')
    print(f"Dataset size: {len(new_dataset['observations'])}")


