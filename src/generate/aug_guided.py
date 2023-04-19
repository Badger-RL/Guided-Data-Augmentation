import os

import gym
import numpy as np
import h5py
import argparse

from stable_baselines3.common.utils import set_random_seed

from augment.rotate_reflect_trajectory import rotate_reflect_traj
from augment.translate_reflect_trajectory import translate_reflect_traj_y
from augment.utils import check_valid
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

def gen_aug_dataset(env, observed_dataset, check_goal_post, validate=True, aug_size=100000,):

    obs = observed_dataset['observations']
    action = observed_dataset['actions']
    reward = observed_dataset['rewards']
    next_obs = observed_dataset['next_observations']
    done = observed_dataset['terminals']

    terminal_indices = np.where(done)[0]
    num_episodes = np.sum(done)
    episode_length = terminal_indices[0]+1

    obs = obs.reshape(num_episodes, -1, 8)
    action = action.reshape(num_episodes, -1, 3)
    reward = reward.reshape(num_episodes, -1)
    next_obs = next_obs.reshape(num_episodes, -1, 8)
    done = done.reshape(num_episodes, -1)


    aug_obs_list, aug_action_list, aug_reward_list, aug_next_obs_list, aug_done_list = [], [], [], [], []



    num_guided_traj_to_generate = aug_size//episode_length - num_episodes
    aug_count = 0
    invalid_count = 0
    while aug_count < num_guided_traj_to_generate:

        for episode_i in range(num_episodes):
            aug_function = rotate_reflect_traj

            aug_obs, aug_action, aug_reward, aug_next_obs, aug_done = aug_function(
                obs[episode_i], action[episode_i], next_obs[episode_i], reward[episode_i], done[episode_i], check_goal_post)
            # episode_obs, episode_action, episode_next_obs, episode_reward, episode_done, check_goal_post)
            if aug_obs is None:
                continue

            if validate:
                is_valid = check_valid(env, aug_obs, aug_action, aug_reward, aug_next_obs)
            else:
                is_valid = True

            if is_valid and not(aug_obs is None):
                aug_count += 1
                print(aug_count)
                aug_obs_list.append(aug_obs)
                aug_action_list.append(aug_action)
                aug_reward_list.append(aug_reward)
                aug_next_obs_list.append(aug_next_obs)
                aug_done_list.append(aug_done)
            else:
                invalid_count += 1

            if aug_count >= num_guided_traj_to_generate:
                break

    aug_obs = np.concatenate(aug_obs_list)
    aug_action = np.concatenate(aug_action_list)
    aug_reward = np.concatenate(aug_reward_list)
    aug_next_obs = np.concatenate(aug_next_obs_list)
    aug_done = np.concatenate(aug_done_list)

    aug_dataset = {
        'observations': aug_obs,
        'actions': aug_action,
        'terminals': aug_done,
        'rewards': aug_reward,
        'next_observations': aug_next_obs,
    }
    npify(aug_dataset)

    print(f'Invalid count: {invalid_count}')
    return aug_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--aug-size', type=int, default=int(100e3), help='Number of augmented trajectories to generate')
    parser.add_argument('--observed-dataset-path', type=str, default=f'../datasets/expert/no_aug/50k.hdf5', help='path to observed trajectory dataset')
    parser.add_argument('--save-dir', type=str, default='../datasets/physical/aug_guided', help='Directory to save augmented dataset')
    parser.add_argument('--save-name', type=str, default='10_episodes_200k.hdf5', help='Name of augmented dataset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--check-goal-post', type=int, default=False, help='Verify that augmented trajectories do not pass through the goal post')
    parser.add_argument('--validate', type=int, default=True, help='Verify augmented transitions agree with simulation')


    args = parser.parse_args()

    set_random_seed(args.seed)

    env = gym.make('PushBallToGoal-v0')

    observed_dataset = load_observed_data(dataset_path=args.observed_dataset_path)
    aug_dataset = gen_aug_dataset(env, observed_dataset, aug_size=args.aug_size, validate=args.validate, check_goal_post=args.check_goal_post)

    os.makedirs(args.save_dir, exist_ok=True)
    fname = f'{args.save_dir}/{args.save_name}'
    new_dataset = h5py.File(fname, 'w')

    for k in aug_dataset:
        observed = observed_dataset[k]
        aug = np.array(aug_dataset[k])
        data = np.concatenate([observed, aug])
        new_dataset.create_dataset(k, data=data, compression='gzip')

    print(f"Aug dataset size: {new_dataset['observations'].shape[0]}")

