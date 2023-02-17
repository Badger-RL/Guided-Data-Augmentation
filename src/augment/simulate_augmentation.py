# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import time

import numpy as np
import h5py
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.env_util import make_vec_env

from src.envs.push_ball_to_goal import PushBallToGoalEnv

from GuidedDataAugmentationForRobotics.src.augment.augmentation_function import AbstractSimAugmentationFunction

models = {"push_ball_to_goal": {"env": PushBallToGoalEnv}}

def main():

    path = 'push_ball_to_goal'
    normalization_path = f"../expert_policies/{path}/vector_normalize"
    env = VecNormalize.load(
        normalization_path, make_vec_env(models[path]["env"], n_envs=1)
    )

    env.reset()
    s_o = env.get_original_obs()
    act = env.action_space.sample()

    ns, r, done, info = env.step([act])
    ns_o = env.get_original_obs()

    f = AbstractSimAugmentationFunction(env=None)

    for _ in range(1000):

        s, ns, a, r, terminated, truncated = f.augment(s_o[0], ns_o[0], act, r, done, done)
        env.envs[0].set_abstract_state(s)
        env.render()
        time.sleep(0.5)

if __name__ == '__main__':
    main()