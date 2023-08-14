import gym
import numpy as np
import h5py
import argparse
from src.generate.utils import load_dataset
import time
import os
import d4rl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='antmaze-umaze-diverse-v1', help='Maze type')
    parser.add_argument('--observed-dataset-path', type=str, default="/Users/yxqu/Desktop/Research/GuDA/Antmaze_Dataset/antmaze-umaze-diverse-v1/no_aug_no_collisions_relabeled.hdf5")
    parser.add_argument('--start-timestamp', type=int, default=0)
    parser.add_argument('--end-timestamp', type=int, default=-1)
    parser.add_argument('--freq', type=float, default=0.01)
    parser.add_argument('--render', action='store_true', help='Render trajectories')

    args = parser.parse_args()
    print(args)
    env = gym.make(args.env_name)
    env.reset()

    dataset = load_dataset(args.observed_dataset_path)
    print(dataset.keys())
    print("observations: ", dataset['observations'].shape)
    print("actions: ", dataset['actions'].shape)
    print("rewards: ", dataset['rewards'].shape)
    print("terminals: ", dataset['terminals'].shape)
    print("next_observations: ", dataset['next_observations'].shape)
    
    i = args.start_timestamp
    target = np.array([0.0,0.0])
    env.set_target(target)
    
    if args.end_timestamp == -1:
        args.end_timestamp = len(dataset['observations']) - 1
    num_of_trajectory = 1
    length = 0
    for i in range(args.start_timestamp, args.end_timestamp + 1):
        qpos_prev = dataset['next_observations'][i-1][:15]
        qpos = dataset['observations'][i][:15]
        if not np.allclose(qpos, qpos_prev):
            length = 1
            num_of_trajectory += 1
        else:
            length += 1
        qvel = dataset['observations'][i][15:]
        act = dataset['actions'][i]
        env.set_state(qpos, qvel)
        # env.set_marker()
        ns, reward, _, _ = env.step(act)

        print(f"Timestamp: {i}, ns - dataset['next_observations'][i]: {ns - dataset['next_observations'][i]}")
        # if not np.allclose(ns[:2], dataset['next_observations'][i][:2]) or not np.allclose(reward, dataset['rewards'][i]):
        #     num_of_invalid += 1
        # if (i + 1) % 10000 == 0:
        #     print(f"Fraction of invalid: {num_of_invalid}/{i+1}")
        #     print("Number of trajectories: ", num_of_trajectory)
        if args.render:
            env.render()
            time.sleep(args.freq)
        # env.reset()

if __name__ == "__main__":
    main()


