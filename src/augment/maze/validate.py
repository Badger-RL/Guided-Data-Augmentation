# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import time

import gym, d4rl
import numpy as np
import random

from augment.maze.point_maze_aug_function import PointMazeAugmentationFunction


def main():

    # f = RotateReflectTranslate(env=None)


    # env = gym.make('maze2d-umaze-v0')
    env = gym.make('maze2d-open-v0')

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    num_steps = int(1e6)

    f = PointMazeAugmentationFunction(env)

    for t in range(num_steps):
        obs = env.reset()
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)

        aug_obs, aug_action, aug_reward, aug_next_obs, aug_done = f.augment(obs, action, next_obs, reward, done)
        aug_qpos = aug_obs[:2]
        aug_qvel = aug_obs[2:]
        env.set_state(aug_qpos, aug_qvel)
        env.set_marker()
        true_next_obs, true_reward, true_done, true_info = env.step(aug_action)

        assert np.allclose(aug_next_obs, true_next_obs)
        assert np.allclose(aug_reward, true_reward)

if __name__ == '__main__':
    main()