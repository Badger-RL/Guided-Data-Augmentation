import copy
import os

import gym
import numpy as np
import h5py
import argparse

from stable_baselines3.common.utils import set_random_seed

from src.augment.abstractsim.rotate_reflect_trajectory import rotate_reflect_traj, random_traj
from src.augment.abstractsim.translate_reflect_trajectory import translate_reflect_traj_y
from src.augment.utils import check_valid, convert_to_absolute_obs, calculate_reward, convert_to_relative_obs
import custom_envs

def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def load_observed_data(dataset_path):
    observed_data_hdf5 = h5py.File(f"{dataset_path}", "r")

    observed_dataset = {}
    for key in observed_data_hdf5.keys():
        observed_dataset[key] = np.array(observed_data_hdf5[key])

    return observed_dataset



def gen_aug_dataset(env, observed_dataset, aug, check_goal_post, validate=True, aug_size=100000,):

    obs = observed_dataset['observations'][:201500]
    action = observed_dataset['actions'][:201500]
    next_obs = observed_dataset['next_observations'][:201500]
    aug_obs_list, aug_action_list, aug_reward_list, aug_next_obs_list, aug_done_list = [], [], [], [], []

    absolute_next_obs = convert_to_absolute_obs(next_obs)
    aug_reward, _ = calculate_reward(absolute_next_obs)
    aug_done = np.zeros_like(aug_reward)
    aug_done[::500] = 1

    aug_obs = obs
    aug_action = action
    aug_next_obs = next_obs

    aug_dataset = {
        'observations': aug_obs,
        'actions': aug_action,
        'terminals': aug_done,
        'rewards': aug_reward,
        'next_observations': aug_next_obs,
    }
    npify(aug_dataset)

    return aug_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', type=str, default='random')
    parser.add_argument('--neg', type=int, default=0)
    parser.add_argument('--aug-size', type=int, default=int(200e3), help='Number of augmented trajectories to generate')
    parser.add_argument('--observed-dataset-path', type=str, default=f'../../datasets/PushBallToGoal-v0/mocoda_unlabeled.hdf5', help='path to observed trajectory dataset')
    parser.add_argument('--save-dir', type=str, default='../../datasets/PushBallToGoal-v0', help='Directory to save augmented dataset')
    parser.add_argument('--save-name', type=str, default='mocoda.hdf5', help='Name of augmented dataset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--check-goal-post', type=int, default=False, help='Verify that augmented trajectories do not pass through the goal post')
    parser.add_argument('--validate', type=int, default=False, help='Verify augmented transitions agree with simulation')


    args = parser.parse_args()

    set_random_seed(args.seed)

    env = gym.make('PushBallToGoal-v0')

    observed_dataset = load_observed_data(dataset_path=args.observed_dataset_path)
    aug_dataset = gen_aug_dataset(env, observed_dataset, aug_size=args.aug_size, aug=args.aug, validate=args.validate, check_goal_post=args.check_goal_post)

    os.makedirs(args.save_dir, exist_ok=True)
    fname = f'{args.save_dir}/{args.save_name}'
    new_dataset = h5py.File(fname, 'w')

    for k in aug_dataset:
        data = np.array(aug_dataset[k])
        new_dataset.create_dataset(k, data=data, compression='gzip')

    print(f"Aug dataset size: {new_dataset['observations'].shape[0]}")

