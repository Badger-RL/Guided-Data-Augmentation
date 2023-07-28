import argparse

import gym, custom_envs

from augment.utils import check_valid
from generate.utils import load_dataset, unpack_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset-path', type=str, default=f'../../datasets/PushBallToGoal-v0/no_aug_72_10k.hdf5', help='path to observed trajectory dataset')
    parser.add_argument('--dataset-path', type=str, default=f'tmp.hdf5', help='path to observed trajectory dataset')

    args = parser.parse_args()

    env = gym.make('PushBallToGoal-v0')

    aug_dataset = load_dataset(dataset_path=args.dataset_path)
    
    n = aug_dataset['observations'].shape[0]
    n = 10000
    is_valid = check_valid(
        env,
        aug_dataset['absolute_observations'][n:],
        aug_dataset['actions'][n:],
        aug_dataset['rewards'][n:],
        aug_dataset['absolute_next_observations'][n:],
        aug_dataset['terminals'][n:],
        verbose=True,
        render=1,
    )
    if not is_valid:
        print(f'Dataset {args.dataset_path} is NOT valid.')
        # break



