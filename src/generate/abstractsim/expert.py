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
            'dones': [],

            'absolute_observations': [],
            'absolute_next_observations': []
            }


def append_data(data, s, a, r, ns, terminal, done, abs_s, abs_ns):
    data['observations'].append(s)
    data['next_observations'].append(ns)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(terminal)
    data['dones'].append(done)

    data['absolute_observations'].append(abs_s)
    data['absolute_next_observations'].append(abs_ns)

def append_dataset(data, append_data):
    data['observations'].append(append_data['observations'])
    data['next_observations'].append(append_data['next_observations'])
    data['actions'].append(append_data['actions'])
    data['rewards'].append(append_data['rewards'])
    data['terminals'].append(append_data['terminals'])
    data['absolute_observations'].append(append_data['absolute_observations'])
    data['absolute_next_observations'].append(append_data['absolute_next_observations'])

def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts', 'dones']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='PushBallToGoal-v2')
    parser.add_argument('--num_samples', type=int, default=int(100e3), help='Num samples to collect')
    parser.add_argument('--num_traj', type=int, default=5, help='Num trajectories to collect. Overrides num_samples')
    parser.add_argument('--num_traj_success', type=int, default=5, help='Num trajectories to collect. Overrides num_samples')
    parser.add_argument('--policy-path', type=str, default='../../../src/policies/PushBallToGoal-v2/policy_72.zip')
    parser.add_argument('--norm-path', type=str, default='')
    parser.add_argument('--save-dir', type=str, help='file_name')
    parser.add_argument('--save-name', type=str, help='file_name')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--random_actions', type=int, default=0)
    parser.add_argument('--render', type=bool, default=0)

    args = parser.parse_args()

    env = gym.make(args.env_id,
                   # init_robot_x_range=(4400, 4400),
                   # init_robot_y_range=(-3000, -3000),
                   # init_ball_x_range=(-4400, -4400),
                   # init_ball_y_range=(3000, 3000)

                   # init_robot_x_range=(4400, 4400),
                   # init_robot_y_range=(-3000, -3000),
                   init_ball_x_range=(-1000,+1000),
                   init_ball_y_range=(3000,3000)
                   )

    set_random_seed(args.seed)
    s = env.reset()
    policy = PPO.load(args.policy_path)

    data = reset_data()

    ts = 0
    num_episodes = 0
    ret = 0
    successes = []
    rets =[]

    # num_traj_success = 0
    # num_traj = 0
    # while num_traj < args.num_traj and num_traj_success < args.num_traj_success:
    for episode_i in range(args.num_traj):
        done = False
        while not done:
            if args.random_actions:
                act = env.action_space.sample()
            else:
                act = policy.predict(s)[0]
            act[-1] = 0

            ns, r, done, info = env.step(act)
            ret += r
            if args.render:
                env.render()

            # print(info['is_success'], info['terminated'],  done)
            append_data(data, s, act, r, ns, done, done, info['absolute_obs'], info['absolute_next_obs'])

            if len(data['observations']) % 10000 == 0:
                print(len(data['observations']))
            # print(r)
            ts += 1
            if ts % 50 == 0:
                print(f't = {ts}')
            if done:
                s = env.reset()
                num_episodes += 1

                if info['is_success']:
                    successes.append(1)

                else:
                    successes.append(0)

                rets.append(ret)
                print(ts, ret, successes[-1])
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