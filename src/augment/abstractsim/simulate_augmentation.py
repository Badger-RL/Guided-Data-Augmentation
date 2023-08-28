# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import time

import gym, custom_envs

def main():

    env = gym.make('PushBallToGoalHard-v0')
    env.reset()
    aug = False
    for _ in range(100000):
        env.render()
        # act, _ = policy.predict(s)
        # ns, r, done, info = env.step(act)
        # ns_o = env.get_original_obs()
        #
        # if np.linalg.norm(ns_o[0,2:3] - s_o[0,2:3]) > 0:
        #     aug = True
        #     aug_s, aug_a, aug_ns, aug_r, aug_done = f.augment(s_o[0], ns_o[0], act[0], r, done)
        #     env.envs[0].set_abstract_state(aug_s)
        #
        #     env.render()
        #     if aug:
        #         time.sleep(0.25)
        #
        #     s = env.reset()
        #     s_o = env.get_original_obs()
        #
        # else:
        #     s = ns
        #     s_o = ns_o



if __name__ == '__main__':
    main()