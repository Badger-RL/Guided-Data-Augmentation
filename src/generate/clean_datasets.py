import os
import gym
import h5py
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
    observed_dataset = d4rl.qlearning_dataset(env)
    observed_dataset_obs = observed_dataset['observations']
    observed_dataset_action = observed_dataset['actions']
    observed_dataset_next_obs = observed_dataset['next_observations']
    observed_dataset_reward = observed_dataset['rewards']
    observed_dataset_done = observed_dataset['terminals']

    f = AUG_FUNCTIONS[env_id]['random'](env=env)

    clean_dataset = reset_data()
    aug_count = 0 # number of valid augmentations produced
    for i in range(len(observed_dataset_obs)):
        is_valid_input = f.is_valid_input(
            obs=observed_dataset_obs[i],
            next_obs=observed_dataset_next_obs[i],
        )

        if is_valid_input:
            aug_count += 1
            if aug_count % 1000 == 0: print('aug_count:', aug_count)
            append_data(
                clean_dataset,
                observed_dataset_obs[i],
                observed_dataset_action[i],
                observed_dataset_reward[i],
                observed_dataset_next_obs[i],
                observed_dataset_done[i])

    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{save_name}'
    new_dataset = h5py.File(save_path, 'w')
    for k in clean_dataset:
        new_dataset.create_dataset(k, data=clean_dataset[k], compression='gzip')
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



