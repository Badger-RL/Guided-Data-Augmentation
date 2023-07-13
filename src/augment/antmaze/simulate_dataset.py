# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import time

import gym, d4rl
import numpy as np
import random

from augment.antmaze.antmaze_aug_function import AntMazeAugmentationFunction
from augment.maze.point_maze_aug_function import PointMazeAugmentationFunction


def main():

    # f = RotateReflectTranslate(env=None)


    # env = gym.make('maze2d-umaze-v0')
    env = gym.make('antmaze-umaze-diverse-v1')
    dataset = d4rl.qlearning_dataset(env)

    observations = dataset['observations']
    actions = dataset['actions']
    next_observations = dataset['next_observations']
    # rewards = dataset['rewards']
    dones = dataset['terminals']


    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    num_steps = int(1e6)

    f = AntMazeAugmentationFunction(env)

    obs = env.reset()

    for t in range(num_steps):
        # obs = env.reset()
        qpos = observations[t, :15]
        qvel = observations[t, 15:]
        env.set_state(qpos, qvel)
        # time.sleep(3)
        action = actions[t]
        # action = np.zeros_like(action)
        next_obs, reward, done, info = env.step(action)
        obs = next_obs

        env.render()
        print(next_obs[:3])

        # if reward > 0:

        # if dones[t]:
        #     qpos = observations[t,:15]
        #     qvel = observations[t,15:]
        #     env.set_state(qpos, qvel)


if __name__ == '__main__':
    main()