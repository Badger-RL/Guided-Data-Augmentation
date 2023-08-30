# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import time

import gym, d4rl
import numpy as np
import random

from augment.antmaze.antmaze_aug_function import AntMazeAugmentationFunction, AntMazeGuidedAugmentationFunction
from augment.maze.point_maze_aug_function import PointMazeAugmentationFunction


def main():

    # f = RotateReflectTranslate(env=None)


    # env = gym.make('maze2d-umaze-v0')
    env = gym.make('antmaze-umaze-diverse-v0')

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    num_steps = int(1e6)

    f = AntMazeGuidedAugmentationFunction(env)

    obs = env.reset()
    print(obs[3:7])

    obs = np.array(
        [-3.89104816e-02,  1.81146354e-01, 3.69904601e-01, 7.08710177e-01,
         - 9.16952844e-05, - 6.09752189e-05, 7.05499732e-01, 5.63445599e-02,
         5.23425730e-01,  3.33113692e-01, - 5.23432674e-01,  1.38210047e-01,
         - 5.23450078e-01,  9.25164954e-02,  5.23444314e-01,  0.00000000e+00,
         0.00000000e+00 , 2.08342598e-07, - 2.41997265e-07, - 7.32552344e-07,
         - 5.15603939e-03 , 6.83734421e-03,  3.64341480e-07, 6.90505922e-03,
         - 5.43488758e-07,  7.25657042e-03, - 2.01560456e-07,  7.14697111e-03,
         6.12681682e-07]
    )

    for t in range(num_steps):
        qpos = obs[:15]
        qvel = obs[15:]
        # alpha = np.pi*3/2
        # qpos[3:6+1] = np.array([np.cos(alpha/2),0,0,np.sin(alpha/2)])
        # qpos[3] = np.cos(alpha/2)
        # qpos[6] = np.sin(alpha/2)
        # qvel[0] = 0
        # qvel[1] = 0
        env.set_state(qpos, qvel)
        # obs = env.reset()

        # env.render()
        # time.sleep(3)
        action = env.action_space.sample()
        # action = np.zeros_like(action)
        next_obs, reward, done, info = env.step(action)
        # env.render()
        # print(obs)


        aug_obs, aug_action, aug_reward, aug_next_obs, aug_done = f.augment(obs, action, next_obs, reward, done)

        aug_qpos = aug_obs[:15]
        aug_qvel = aug_obs[15:]
        env.set_state(aug_qpos, aug_qvel)
        # env.set_xy(aug_qpos[:2])
        env.render()

        true_next_obs, true_reward, true_done, true_info = env.step(aug_action)


        print((aug_next_obs-aug_obs)[:2])
        # print((true_next_obs-aug_obs)[:2])

        if not np.allclose(aug_next_obs, true_next_obs):
            stop = 0
        print(aug_next_obs-true_next_obs)
        # print(obs[15:16+1])
        # print(next_obs[15:16+1])
        # print(aug_next_obs[15:16+1])
        # print(true_next_obs[15:16+1])

        assert np.allclose(aug_next_obs, true_next_obs)
        # assert np.allclose(augs_reward, true_reward)

        next_obs = aug_next_obs.copy()

        obs = next_obs.copy()


'''
# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import time

import gym, d4rl
import numpy as np
import random

from augment.antmaze.antmaze_aug_function import AntMazeAugmentationFunction, AntMazeGuidedAugmentationFunction
from augment.maze.point_maze_aug_function import PointMazeAugmentationFunction


def main():

    # f = RotateReflectTranslate(env=None)


    # env = gym.make('maze2d-umaze-v0')
    env = gym.make('antmaze-umaze-diverse-v0')

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    num_steps = int(1e6)

    f = AntMazeGuidedAugmentationFunction(env)

    obs = env.reset()

    for t in range(num_steps):
        # qpos = obs[:15]
        # qvel = obs[15:]
        # alpha = np.pi*3/2
        # qpos[3:6+1] = np.array([np.cos(alpha/2),0,0,np.sin(alpha/2)])
        # qpos[3] = np.cos(alpha/2)
        # qpos[6] = np.sin(alpha/2)

        # env.set_state(qpos, qvel)
        # obs = env.reset()

        # env.render()
        # time.sleep(3)
        action = env.action_space.sample()
        # action = np.zeros_like(action)
        next_obs, reward, done, info = env.step(action)
        # env.render()
        # print(obs[3:6+1])

        if t % 1 == 0:

            aug_obs, aug_action, aug_reward, aug_next_obs, aug_done = f.augment(obs, action, next_obs, reward, done)

            aug_qpos = aug_obs[:15]
            aug_qvel = aug_obs[15:]
            env.set_state(aug_qpos, aug_qvel)
            # env.set_xy(aug_qpos[:2])
            env.render()

            true_next_obs, true_reward, true_done, true_info = env.step(aug_action)

            if not np.allclose(aug_next_obs, true_next_obs):
                stop = 0
            print(aug_next_obs-true_next_obs)
            # assert np.allclose(aug_next_obs, true_next_obs)
            # assert np.allclose(aug_reward, true_reward)

        obs = next_obs.copy()

if __name__ == '__main__':
    main()
'''

if __name__ == '__main__':
    main()