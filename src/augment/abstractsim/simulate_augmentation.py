# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import time

import gym, custom_envs
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.env_util import make_vec_env

from src.augment.abstractsim.rotate_reflect_translate import RotateReflectTranslate, RotateReflectTranslateGuided

def main():

    normalization_path = f"../../policies/PushBallToGoal-v0/vector_normalize"
    env = VecNormalize.load(
        normalization_path, make_vec_env('PushBallToGoal-v0', n_envs=1,
                                         )
    )
    policy = PPO.load('../../policies/PushBallToGoal-v0/policy_100.zip')
    env.norm_reward=False

    s = env.reset()
    # s_o = env.get_original_obs()
    # act = env.action_space.sample()
    #
    # ns, r, done, info = env.step([act])
    # ns_o = env.get_original_obs()
    f = RotateReflectTranslateGuided(env=None)

    for _ in range(1000):
        s_o = env.get_original_obs()
        act, _ = policy.predict(s)
        ns, r, done, info = env.step(act)
        env.render()

        ns_o = env.get_original_obs()

        if np.linalg.norm(ns_o[0,2:3] - s_o[0,2:3]) > 0:
            aug_s, aug_a, aug_ns, aug_r, aug_done = f.augment(s_o[0], ns_o[0], act[0], r, done)
            env.envs[0].set_abstract_state(aug_s)
            env.render()
            time.sleep(0.05)

        s = ns



if __name__ == '__main__':
    main()