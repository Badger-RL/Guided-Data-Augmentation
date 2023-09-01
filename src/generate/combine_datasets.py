import os
import gym
import h5py
import numpy as np

import d4rl
from generate.augment_dataset import AUG_FUNCTIONS
from generate.utils import reset_data
from simulate_cql import append_data


def fetch_dataset(
        env_id,
        save_dir=None,
        save_name=None
):
    if save_dir is None:
        save_dir = f'../datasets/{env_id}'
    if save_name is None:
        save_name = f'no_aug_clean.hdf5'

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

    for env_id in [
        # 'maze2d-umaze-v1',
        # 'maze2d-medium-v1',
        # 'maze2d-large-v1',
        # 'antmaze-umaze-diverse-v1',
        'antmaze-medium-diverse-v1',
        'antmaze-large-diverse-v1',
    ]:
        print(env_id)
        fetch_dataset(env_id)



