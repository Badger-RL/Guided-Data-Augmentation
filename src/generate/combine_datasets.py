import os
import gym
import h5py
import numpy as np

import d4rl
from generate.utils import load_dataset


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
    observed_dataset = d4rl.qlearning_dataset(env)

    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{save_name}'
    new_dataset = h5py.File(save_path, 'w')
    for k in observed_dataset:
        new_dataset.create_dataset(k, data=observed_dataset[k], compression='gzip')
    print(f"Dataset size: {len(new_dataset['observations'])}")

if __name__ == '__main__':

    env_id = 'antmaze-umaze-diverse-v1'

    dataset1 = load_dataset('../datasets/antmaze-umaze-diverse-v1/no_aug_no_collisions_relabeled.hdf5')
    dataset2 = load_dataset('../datasets/antmaze-umaze-diverse-v1/guided.hdf5')

    save_dir = f'../datasets/{env_id}'
    save_name = f'guided_extra_data.hdf5'

    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{save_name}'
    new_dataset = h5py.File(save_path, 'w')
    for k in dataset2:
        data = np.concatenate([dataset1[k][10000:11000], dataset2[k]])
        new_dataset.create_dataset(k, data=data, compression='gzip')
    print(f"Dataset size: {len(new_dataset['observations'])}")



