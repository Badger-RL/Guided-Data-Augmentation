#Derived from D4RL
#https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
#https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import os

import numpy as np
import h5py
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

import gym, custom_envs



def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'next_observations': [],

            'absolute_observations': [],
            'absolute_next_observations': []
            }


def append_data(data, s, a, r, ns, done, abs_s, abs_ns):
    data['observations'].append(s)
    data['next_observations'].append(ns)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)

    data['absolute_observations'].append(abs_s)
    data['absolute_next_observations'].append(abs_ns)

def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=int(100e3), help='Num samples to collect')
    parser.add_argument('--policy-path', type=str, default='../../../src/results/PushBallToGoal-v0/rl_model_3200000_steps.zip')
    parser.add_argument('--norm-path', type=str, default='')
    parser.add_argument('--save-dir', type=str, help='file_name')
    parser.add_argument('--save-name', type=str, help='file_name')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--random_actions', type=int, default=0)
    parser.add_argument('--render', type=bool, default=False)

    args = parser.parse_args()

    env = gym.make('PushBallToGoal-v0')

    set_random_seed(args.seed)
    s = env.reset()
    policy = PPO.load(args.policy_path)

    data = reset_data()

    ts = 0
    num_episodes = 0
    ret = 0
    successes = []
    rets =[]
    for _ in range(args.num_samples):
        if args.random_actions:
            act = env.action_space.sample()
        else:
            act = policy.predict(s)[0]
        act[-1] = 0
          
        ns, r, done, info = env.step(act)
        ret += r
        if args.render:
            env.render()

        append_data(data, s, act, r, ns, info['terminated'], info['absolute_obs'], info['absolute_next_obs'])

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1
        if done:
            s = env.reset()
            num_episodes += 1


            if info['is_success']:
                successes.append(1)

            else:
                successes.append(0)

            rets.append(ret)
            print(ts, ret)
            ret = 0
            ts = 0
        else:
            s = ns

    print(np.average(successes), np.std(successes)/np.sqrt(len(rets)), np.average(rets), np.std(rets)/np.sqrt(len(rets)))
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    fname = f'{save_dir}/{args.save_name}'
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')

if __name__ == '__main__':
    main()