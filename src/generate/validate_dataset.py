import argparse

import gym, custom_envs

from augment.utils import check_valid
from generate.utils import load_dataset, unpack_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default=f'../datasets/expert/translate_and_rotate/10000_1.hdf5', help='path to observed trajectory dataset')
    # parser.add_argument('--dataset-path', type=str, default=f'../datasets/expert/trajectories/0.hdf5', help='path to observed trajectory dataset')

    args = parser.parse_args()

    env = gym.make('PushBallToGoal-v0')

    aug_dataset = load_dataset(dataset_path=args.dataset_path)
    
    n = aug_dataset['observations'].shape[0]
    is_valid = check_valid(
        env,
        aug_dataset['observations'],
        aug_dataset['actions'],
        aug_dataset['rewards'],
        aug_dataset['next_observations'],
        render=False,
        verbose=True,
    )
    if not is_valid:
        print(f'Dataset {args.dataset_path} is NOT valid.')
        # break



