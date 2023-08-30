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
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make('maze2d-medium-v0')
    env.reset()

    num_steps = 10000
    print(env.empty_and_goal_locations)

    for t in range(num_steps):
        action = env.action_space.sample()
        # print(action)
        action = np.zeros(2)
        # action[0] = -1
        next_state, reward, done, info = env.step(action)
        print(next_state)

        env.render()
        print(reward)
        if done:
            env.reset()
            idx = env.np_random.choice(len(env.empty_and_goal_locations))
            reset_location = np.array(env.empty_and_goal_locations[idx]).astype(env.observation_space.dtype)
            reset_location = np.array((3,3)).astype(env.observation_space.dtype)
            qpos = reset_location + env.np_random.uniform(low=-0.75, high=.25, size=env.model.nq)
            qpos = reset_location + np.array([-0.597, -0.597])
            reset_location = np.array((4,1))
            qpos = reset_location + np.array([-0.2,-0.2])
            qpos = reset_location + np.array([0,0.])
            # env.reset_to_location(qpos)
            print(env.get_target())

            qvel = np.zeros(2) #env.init_qvel + env.np_random.randn(env.model.nv) * .1

            env.set_state(qpos, qvel)
            env.set_marker()

    #
    # for _ in range(1000):
    #
    #     aug_s, aug_a, aug_ns, aug_r, aug_done = f.augment(obs, next_obs, action, reward, done)
    #     env.envs[0].set_abstract_state(aug_s)
    #     env.render()
    #     time.sleep(0.5)

if __name__ == '__main__':
    main()