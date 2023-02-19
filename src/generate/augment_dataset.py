# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import os

import numpy as np
import h5py
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.env_util import make_vec_env

from src.envs.push_ball_to_goal import PushBallToGoalEnv

from GuidedDataAugmentationForRobotics.src.augment.augmentation_function import AbstractSimAugmentationFunction, \
    TranslateRobotAndBall

models = {"push_ball_to_goal": {"env": PushBallToGoalEnv}}


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'next_observations': [],
            }


def append_data(data, s, a, r, ns, done):
    data['observations'].append(s)
    data['next_observations'].append(ns)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)


def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=int(1e4), help='Num samples to collect')
    parser.add_argument('--path', type=str, default='push_ball_to_goal', help='file_name')
    parser.add_argument('--max_episode_steps', default=1000, type=int)
    parser.add_argument('--use_policy', default=True, type=bool)

    parser.add_argument('--random_actions', action='store_true')
    parser.set_defaults(feature=False)
    parser.add_argument('--render', action='store_true')
    parser.set_defaults(render=True)

    args = parser.parse_args()

    policy_path = f"../expert_policies/{args.path}/policy"
    normalization_path = f"../expert_policies/{args.path}/vector_normalize"

    # model.save(f"./Models/{params['path']}/policy")
    # env.save(f"./Models/{params['path']}/vector_normalize")
    env = VecNormalize.load(
        normalization_path, make_vec_env(models[args.path]["env"], n_envs=1)
    )
    env.norm_obs = True
    env.norm_reward = False
    env.clip_obs = 1.0
    env.training = False

    s = env.reset()
    s_o = env.get_original_obs()
    act = env.action_space.sample()
    done = False

    custom_objects = {
        "lr_schedule": lambda x: .003,
        "clip_range": lambda x: .02
    }
    policy = PPO.load(policy_path, custom_objects=custom_objects, env=env)

    data = reset_data()

    ts = 0
    num_episodes = 0
    for _ in range(args.num_samples):

        act = None
        if not args.random_actions:
            act = policy.predict(s)[0]
            # print(act)
        else:
            act = [env.action_space.sample()]

        ns, r, done, info = env.step(act)
        ns_o = env.get_original_obs()
        if args.render:
            env.render()
        timeout = False
        if ts >= args.max_episode_steps:
            timeout = True

        append_data(data, s_o[0], act[0], r[0], ns_o[0], done[0])

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1

        if done or timeout:
            done = False
            ts = 0
            s = env.reset()
            s_o = env.get_original_obs()

            num_episodes += 1
            frames = []
        else:
            s = ns
            s_o = ns_o

    fname = f'dataset_{"expert" if args.use_policy else "random"}_{args.num_samples}.hdf5'
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == '__main__':
    dataset = {}
    dataset_size = 100000
    dataset_name = f'../datasets/expert/{dataset_size}.hdf5'
    data_hdf5 = h5py.File(f"{dataset_name}", "r")
    for key in data_hdf5.keys():
        dataset[key] = np.array(data_hdf5[key])

    f = TranslateRobotAndBall(env=None)

    n = dataset['observations'].shape[0]

    aug_data = reset_data()
    aug_n = 7
    for i in range(n):
        for _ in range(aug_n):
            obs, next_obs, action, reward, done = f.augment(
                obs=dataset['observations'][i],
                next_obs=dataset['next_observations'][i],
                action=dataset['actions'][i],
                reward=dataset['rewards'][i],
                done=dataset['terminals'][i]
            )

            if obs is not None:
                append_data(aug_data, obs, action, reward, next_obs, done)

    save_dir = f'../datasets/expert/translate_robot_and_ball/'
    os.makedirs(save_dir, exist_ok=True)
    fname = f'{save_dir}/{dataset_size}_{aug_n}.hdf5'
    new_dataset = h5py.File(fname, 'w')
    # npify(aug_data)
    for k in aug_data:
        observed = dataset[k]
        aug = np.array(aug_data[k])
        data = np.concatenate([observed, aug])
        new_dataset.create_dataset(k, data=data, compression='gzip')

    x = 0
    # main()