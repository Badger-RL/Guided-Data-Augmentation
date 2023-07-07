# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import os

import numpy as np
import h5py
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.env_util import make_vec_env

from custom_envs.push_ball_to_goal import PushBallToGoalEnv
import custom_envs
from generate.utils import reset_data, append_data, npify

models = {"push_ball_to_goal": {"env": PushBallToGoalEnv}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='push_ball_to_goal', help='file_name')
    parser.add_argument('--save-dir', type=str, default='../../datasets/expert/trajectories/', help='Directory to save the dataset')
    parser.add_argument('--save-name', type=str, default='1.hdf5', help='Name of dataset')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_random_seed(args.seed)

    policy_path = f"../../policies/PushBallToGoal-v0/policy_100"
    normalization_path = f"../../policies/PushBallToGoal-v0/vector_normalize_100"

    env = VecNormalize.load(
        normalization_path, make_vec_env('PushBallToGoal-v0', n_envs=1)
    )
    env.norm_obs = True
    env.norm_reward = False
    env.clip_obs = 1.0
    env.epsilon = 1e-16
    env.training = False

    s = env.reset()
    s_o = env.get_original_obs()

    custom_objects = {
        "lr_schedule": lambda x: .003,
        "clip_range": lambda x: .02
    }
    policy = PPO.load(policy_path, custom_objects=custom_objects, env=env)

    data = reset_data()

    ret = 0
    done = False
    while not done:
        act = policy.predict(s)[0]

        ns, r, done, info = env.step(act)
        ret += r
        ns_o = env.get_original_obs()

        if 'terminal_observation' in info[0]:
            ns = [info[0]['terminal_observation']]
            ns_o = env.unnormalize_obs(ns)

        append_data(data, s_o[0], act[0], r[0], ns_o[0], done[0])
        s = ns
        s_o = ns_o
        # env.render()

    print(ret)


    os.makedirs(args.save_dir, exist_ok=True)
    save_path = f'{args.save_dir}/{args.save_name}'
    dataset = h5py.File(save_path, 'w')
    npify(data)

    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == '__main__':
    main()