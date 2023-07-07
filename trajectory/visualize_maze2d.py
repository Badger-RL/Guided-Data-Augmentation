import gym
from d4rl.pointmaze import maze_model
import numpy as np
import h5py
import argparse
from src.generate.utils import load_dataset
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v1', help='Maze type')
    parser.add_argument('--observed-dataset-path', type=str, default="/Users/yxqu/Desktop/Research/GuDA/GuidedDataAugmentationForRobotics/src/datasets/maze2d-umaze-v1/maze2d-umaze-v1-observed.hdf5")
    parser.add_argument('--start-timestamp', type=int, default=0)
    parser.add_argument('--end-timestamp', type=int, default=-1)
    parser.add_argument('--freq', type=float, default=0.01)
    parser.add_argument('--render', action='store_true', help='Render trajectories')

    args = parser.parse_args()

    env = gym.make(args.env_name)
    maze = env.str_maze_spec
    env = maze_model.MazeEnv(maze)

    dataset = load_dataset(args.observed_dataset_path)
    i = args.start_timestamp
    target = np.array([0.0,0.0])
    env.set_target(target)
    
    if args.end_timestamp == -1:
        args.end_timestamp = len(dataset['observations']) - 1
    for i in range(args.start_timestamp, args.end_timestamp + 1):
        qpos = dataset['observations'][i][:2]
        qvel = dataset['observations'][i][2:]
        act = dataset['actions'][i]
        env.set_state(qpos, qvel)
        env.set_marker()
        ns, _, _, _ = env.step(act)
        
        print(f"{i}: {ns - dataset['next_observations'][i]}")
        if args.render:
            env.render()
        time.sleep(args.freq)
        # env.reset()

if __name__ == "__main__":
    main()
