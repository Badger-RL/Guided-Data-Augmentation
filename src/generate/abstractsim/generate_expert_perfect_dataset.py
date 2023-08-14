#Derived from D4RL
#https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
#https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import os

import numpy as np
import h5py
import argparse

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.env_util import make_vec_env

from custom_envs.push_ball_to_goal import PushBallToGoalEnv
import custom_envs
from generate.utils import reset_data, npify, append_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=int(1e4), help='Num samples to collect')
    parser.add_argument('--path', type=str, default='push_ball_to_goal', help='file_name')
    parser.add_argument('--seed', type=int, default=0)
    parser.set_defaults(use_policy = False)
    parser.add_argument('--random_actions', type=int, default=0)
    parser.add_argument('--render', type=bool, default=False)


    args = parser.parse_args()

    policy_path = f"../expert_policies/{args.path}/policy_100"
    normalization_path = f"../expert_policies/{args.path}/vector_normalize_100"

    env = VecNormalize.load(
        normalization_path, make_vec_env('PushBallToGoal-v0', n_envs=1)
    )
    env.norm_obs = True
    env.norm_reward = False
    env.clip_obs = np.inf
    env.training = False
    env.epsilon = 1e-16

    set_random_seed(args.seed)
    s = env.reset()
    s_o = env.get_original_obs()

    custom_objects = {
    "lr_schedule": lambda x: .003,
    "clip_range": lambda x: .02
    }   
    policy = PPO.load(policy_path, custom_objects = custom_objects, env= env)

    data = reset_data()

    ts = 0
    num_episodes = 0
    ret = 0

    ep_transitions = []

    sample_count = 0
    while sample_count < args.num_samples:
        if args.random_actions:
            act = [env.action_space.sample()]
        else:
            act = policy.predict(s)[0]
          
        ns, r, done, info = env.step(act)
        ret += r
        ns_o = env.get_original_obs()
        if args.render:
            env.render()
       
        if 'terminal_observation' in info[0]:
            ns = np.array([info[0]['terminal_observation']])
            ns_o = env.unnormalize_obs(ns)

        ep_transitions.append((s_o[0], act[0], r[0], ns_o[0], done[0]))

        ts += 1

        if done:
            ts = 0
            s = env.reset()
            s_o = env.get_original_obs()
            num_episodes += 1
            print(ret)
            ret = 0

            if info[0]['is_success']:
                for transition in ep_transitions:
                    append_data(data, *transition)
                sample_count += len(ep_transitions)
                if len(data['observations']) % 10000 == 0:
                    print(len(data['observations']))

            ep_transitions = []

        else:
            s = ns
            s_o = ns_o

    save_dir = f'../datasets/expert_perfect/no_aug'
    os.makedirs(save_dir, exist_ok=True)
    fname = f'{save_dir}/{int(args.num_samples//1e3)}k.hdf5'
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')
        print(len(data[k]))

if __name__ == '__main__':
    main()