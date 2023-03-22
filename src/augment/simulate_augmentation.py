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


from GuidedDataAugmentationForRobotics.src.augment.translate_robot_and_ball import TranslateRobotAndBall
from augment.translate_and_rotate import TranslateAndRotate
from custom_envs.push_ball_to_goal import PushBallToGoalEnv

models = {"push_ball_to_goal": {"env": PushBallToGoalEnv}}

def main():

    path = 'push_ball_to_goal'
    normalization_path = f"../expert_policies/{path}/vector_normalize"
    env = VecNormalize.load(
        normalization_path, make_vec_env(models[path]["env"], n_envs=1)
    )
    env.norm_reward=False

    env.reset()
    s_o = env.get_original_obs()
    act = env.action_space.sample()

    ns, r, done, info = env.step([act])
    ns_o = env.get_original_obs()
    f = TranslateAndRotate(env=None)

    for _ in range(1000):

        aug_s, aug_a, aug_ns, aug_r, aug_done = f.augment(s_o[0], ns_o[0], act, r, done)
        env.envs[0].set_abstract_state(aug_s)
        env.render()
        time.sleep(0.5)

if __name__ == '__main__':
    main()