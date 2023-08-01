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
    parser.add_argument('--policy-path', type=str, help='file_name')
    parser.add_argument('--norm-path', type=str, help='file_name')
    parser.add_argument('--save-dir', type=str, help='file_name')
    parser.add_argument('--save-name', type=str, help='file_name')

    # 37 = no success
    parser.add_argument('--seed', type=int, default=1)
    parser.set_defaults(use_policy = False)
    parser.add_argument('--random_actions', type=int, default=0)
    parser.add_argument('--render', type=bool, default=False)


    args = parser.parse_args()

    env = VecNormalize.load(
        args.norm_path, make_vec_env('PushBallToGoal-v0', n_envs=1)
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
    policy = PPO.load(args.policy_path, custom_objects = custom_objects, env= env)

    data = reset_data()

    ts = 0
    num_episodes = 0
    ret = 0
    for _ in range(args.num_samples):
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

        append_data(data, s_o[0], act[0], r[0], ns_o[0], done[0])

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1

        if done:
            ts = 0
            s = env.reset()
            s_o = env.get_original_obs()
            num_episodes += 1
            print(ret)
            ret = 0
        else:
            s = ns
            s_o = ns_o

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    fname = f'{save_dir}/{args.save_name}'
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')

if __name__ == '__main__':
    main()