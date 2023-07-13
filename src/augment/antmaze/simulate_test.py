# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import time

import gym, d4rl
import numpy as np
import random

def main():

    # f = RotateReflectTranslate(env=None)
    seed = 0
    # np.random.seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)

    env = gym.make('antmaze-medium-diverse-v1')
    # env.seed(42)

    print(env.target_goal)
    obs = env.reset()
    print(obs.shape)
    num_steps = 10000
    # print(env.empty_and_goal_locations)
    qpos = np.zeros(15)
    qpos[:2] = np.array([0, 0]) + np.array([0, 0])
    qpos[2:15] = obs[2:15]
    qvel = obs[15:]

    env.set_state(qpos, qvel)

    for t in range(num_steps):
        print(obs[:2])
        action = env.action_space.sample()
        # print(action)
        action = np.ones_like(action)
        # action[0] = -1
        next_obs, reward, done, info = env.step(action)
        # qpos = np.zeros(15)
        # qpos[:2] = np.array([0, 9.5]) + np.array([0,0])
        # qpos[2:15] = obs[2:15]
        # qvel = obs[15:]
        # env.set_state(qpos, qvel)
        # env.set_target((8,-0.09))
        # print(reward)

        env.render()
        obs = next_obs
        # if done:
        #     env.reset()
            # qpos = obs[:15]
            # qvel = obs[15:]
            #
            # env.set_state(qpos, qvel)
            # env.set_marker()

    #
    # for _ in range(1000):
    #
    #     aug_s, aug_a, aug_ns, aug_r, aug_done = f.augment(obs, next_obs, action, reward, done)
    #     env.envs[0].set_abstract_state(aug_s)
    #     env.render()
    #     time.sleep(0.5)

if __name__ == '__main__':
    main()