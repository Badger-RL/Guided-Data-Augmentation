# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import os

import gym
import numpy as np
import h5py
import argparse

import d4rl
from src.augment.antmaze.antmaze_aug_function import AntMazeAugmentationFunction, AntMazeGuidedAugmentationFunction
from src.augment.maze.point_maze_aug_function import PointMazeAugmentationFunction, PointMazeGuidedAugmentationFunction
from src.generate.utils import reset_data, append_data, load_dataset, npify

AUG_FUNCTIONS = {
    'maze2d-umaze-v1': {
        'random': PointMazeAugmentationFunction,
        'guided': PointMazeGuidedAugmentationFunction,
        'mixed': PointMazeAugmentationFunction,
    },
    'maze2d-medium-v1': {
        'random': PointMazeAugmentationFunction,
        'guided': PointMazeGuidedAugmentationFunction,
        'mixed': PointMazeAugmentationFunction,
    },
    'maze2d-large-v1': {
        'random': PointMazeAugmentationFunction,
        'guided': PointMazeGuidedAugmentationFunction,
        'mixed': PointMazeAugmentationFunction,
    },
    'antmaze-umaze-v1': {
        'random': AntMazeAugmentationFunction,
        'guided': AntMazeGuidedAugmentationFunction,
        # 'mixed': AntMazeAugmentationFunction,
    },
    'antmaze-umaze-diverse-v1': {
        'random': AntMazeAugmentationFunction,
        'guided': AntMazeGuidedAugmentationFunction,
        'mixed': AntMazeAugmentationFunction,
    },
    'antmaze-medium-diverse-v1': {
        'random': AntMazeAugmentationFunction,
        'guided': AntMazeGuidedAugmentationFunction,
        'mixed': AntMazeAugmentationFunction,
    },
    'antmaze-large-diverse-v1': {
        'random': AntMazeAugmentationFunction,
        'guided': AntMazeGuidedAugmentationFunction,
        'mixed': AntMazeAugmentationFunction,
    },
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='antmaze-umaze-v1')
    parser.add_argument('--observed-dataset-path', type=str, default=None)
    parser.add_argument('--observed-dataset-frac', '-frac', type=float, default=None)
    parser.add_argument('--observed-dataset-size', '-size', type=int, default=None)

    parser.add_argument('--aug-func', type=str, default='random')
    parser.add_argument('--aug-ratio', '-m', type=int, default=1, help='Number of augmentations per observed transition')
    parser.add_argument('--save-dir', '-fd', type=str, default=None)
    parser.add_argument('--save-name', '-fn', type=str, default=None)

    parser.add_argument('--check-valid', type=int, default=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f'../datasets/{args.env_id}/{args.aug_func}'
    if args.save_name is None:
        args.save_name = f'm_{args.aug_ratio}.hdf5'

    env = gym.make(args.env_id)
    m = args.aug_ratio
    np.random.seed(seed=args.seed)

    if args.observed_dataset_path:
        observed_dataset = load_dataset(args.observed_dataset_path)
    else:
        observed_dataset = d4rl.qlearning_dataset(env)

    if args.observed_dataset_frac:
        n = observed_dataset['observations'].shape[0]
        end = int(n * args.observed_dataset_frac)
    elif args.observed_dataset_size:
        end = args.observed_dataset_size
    else:
        end = observed_dataset['observations'].shape[0]
    for key in observed_dataset:
        observed_dataset[key] = observed_dataset[key][:end]
    n = observed_dataset['observations'].shape[0]

    observed_dataset_obs = observed_dataset['observations']
    observed_dataset_action = observed_dataset['actions']
    observed_dataset_next_obs = observed_dataset['next_observations']
    observed_dataset_reward = observed_dataset['rewards']
    observed_dataset_done = observed_dataset['terminals']

    f = AUG_FUNCTIONS[args.env_id][args.aug_func](env=env)

    aug_dataset = reset_data()
    aug_count = 0 # number of valid augmentations produced
    i = 0
    while aug_count < n*m:
        if args.aug_func == 'mixed' and aug_count == n//5:
            print('Switching to guided aug')
            f = AUG_FUNCTIONS[args.env_id]['guided'](env=env)

        idx = i % n
        obs, action, reward, next_obs, done = f.augment(
            obs=observed_dataset_obs[idx],
            action=observed_dataset_action[idx],
            next_obs=observed_dataset_next_obs[idx],
            reward=observed_dataset_reward[idx],
            done=observed_dataset_done[idx]
        )

        i += 1
        if obs is not None:
            aug_count += 1
            if aug_count % 10000 == 0: print('aug_count:', aug_count)
            append_data(aug_dataset, obs, action, reward, next_obs, done)
        if aug_count >= n * m:
            break

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = f'{args.save_dir}/{args.save_name}'
    new_dataset = h5py.File(save_path, 'w')
    npify(aug_dataset)
    for k in aug_dataset:
        data = np.concatenate([observed_dataset[k], aug_dataset[k]])
        new_dataset.create_dataset(k, data=data, compression='gzip')
    new_dataset.create_dataset('original_size', data=n)
    new_dataset.create_dataset('aug_size', data=aug_count)

    print(f"New dataset size: {len(new_dataset['observations'])}")

