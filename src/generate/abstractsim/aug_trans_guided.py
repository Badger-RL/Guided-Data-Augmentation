# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import os

import gym, custom_envs
import numpy as np
import h5py
import argparse

from augment.abstractsim.guided import RotateReflectTranslateGuided
from generate.abstractsim.expert import reset_data, append_data
from src.augment.abstractsim.random import RotateReflectTranslate
from src.augment.utils import check_valid, is_at_goal
from custom_envs.push_ball_to_goal import PushBallToGoalEnv

models = {"push_ball_to_goal": {"env": PushBallToGoalEnv}}

AUG_FUNCTIONS = {
    'random': RotateReflectTranslate,
    'guided': RotateReflectTranslateGuided
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--observed-dataset-path', type=str, default='../../datasets/PushBallToGoal-v1/no_aug.hdf5')
    parser.add_argument('--augmentation-ratio', '-aug-ratio', type=int, default=1, help='Number of augmentations per observed transition')
    parser.add_argument('--aug-size', '-aug-size', type=int, default=1, help='Number of augmentations per observed transition')
    parser.add_argument('--save-dir', type=str, default='.')
    parser.add_argument('--save-name', type=str, default='tmp.hdf5')
    parser.add_argument('--aug', type=str, default='guided')
    parser.add_argument('--check-valid', type=int, default=False)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    np.random.seed(seed=args.seed)

    aug_ratio = args.augmentation_ratio

    observed_data_hdf5 = h5py.File(f"{args.observed_dataset_path}", "r")

    observed_dataset = {}
    for key in observed_data_hdf5.keys():
        observed_dataset[key] = np.array(observed_data_hdf5[key])
    n = observed_dataset['observations'].shape[0]

    env = gym.make('PushBallToGoal-v1')
    aug_func = AUG_FUNCTIONS[args.aug]
    f = aug_func(env=env)

    aug_dataset = reset_data()
    aug_count = 0 # number of valid augmentations produced
    invalid_count = 0 # keep track of the number of invalid augmentations we skip
    i = 0
    goal_count = 0
    new_pos = None
    new_pos = np.array([4400,0])
    while aug_count < args.aug_size: #n*aug_ratio:
        for _ in range(aug_ratio):
            idx = i % n

            obs, next_obs, action, reward, done, abs_obs, abs_next_obs = f.augment(
                abs_obs=observed_dataset['absolute_observations'][idx],
                abs_next_obs=observed_dataset['absolute_next_observations'][idx],
                action=observed_dataset['actions'][idx],
                reward=observed_dataset['rewards'][idx],
                done=observed_dataset['terminals'][idx],
                new_pos=new_pos,
            )

            i += 1
            if obs is not None:
                new_pos = abs_next_obs[2:4].copy() #+ np.random.uniform(-10, +10, size=(2,))
                if is_at_goal(abs_next_obs[2], abs_next_obs[3]):
                    goal_count += 1
                    new_pos = None
                # print(new_pos)
                is_valid = True
                if args.check_valid:
                    is_valid = check_valid(
                        env=env,
                        aug_obs=[abs_obs],
                        aug_action=[action],
                        aug_reward=[reward],
                        aug_next_obs=[abs_next_obs]
                    )
                if is_valid:
                    aug_count += 1
                    if aug_count % 10000 == 0:
                        print(aug_count, goal_count)
                    append_data(aug_dataset, obs, action, reward, next_obs, done, done, abs_obs, abs_next_obs)
                else:
                    invalid_count += 1
            # else:
            #     new_pos = None

            # if aug_count >= 10:
            #     break
            if aug_count == 11:
                stop = 0

    print(f'Invalid count: {invalid_count}')
    os.makedirs(args.save_dir, exist_ok=True)
    fname = f'{args.save_dir}/{args.save_name}'
    new_dataset = h5py.File(fname, 'w')
    for k in aug_dataset:
        observed = observed_dataset[k]
        aug = np.array(aug_dataset[k])
        data = np.concatenate([observed, aug])

        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
        else:
            dtype = np.float32
        data = data.astype(dtype)

        new_dataset.create_dataset(k, data=data, compression='gzip')

    print(f"Aug dataset size: {len(new_dataset['observations'])}")
