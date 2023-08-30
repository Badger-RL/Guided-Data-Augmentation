import os

import gym
import h5py

import d4rl

if __name__ == "__main__":

    for env_id in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']:
        env = gym.make(env_id)
        dataset = d4rl.qlearning_dataset(env)


        if 'umaze' in env_id:
            start = 0
            end = 1500
        if 'medium' in env_id:
            start = 40000
            end = start + 3000
        if 'large' in env_id:
            start = 0
            end = 4000

        # start = 0
        # end = 1000000
        save_dir = f'../../datasets/{env_id}/small'
        os.makedirs(save_dir, exist_ok=True)
        save_path = f'{save_dir}/no_aug.hdf5'
        new_dataset = h5py.File(save_path, 'w')
        for k,v in dataset.items():
            data = dataset[k][start:end]
            new_dataset.create_dataset(k, data=data, compression='gzip')
