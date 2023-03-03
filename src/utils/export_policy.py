


from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import uuid
import sys
from pytorch2keras import pytorch_to_keras

import d4rl
import gym
import h5py
import numpy as np
# import pyrallis
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution
import torch.nn as nn
import torch.nn.functional as F
import wandb
from algorithms.cql import ContinuousCQL, TanhGaussianPolicy, ReplayBuffer, ReparameterizedTanhGaussian, FullyConnectedQFunction, Scalar

#from envs.push_ball_to_goal import PushBallToGoalEnv


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

checkpoint = torch.load(Path(sys.argv[2]))

dataset = {}
data_hdf5 = h5py.File(sys.argv[1], "r")
for key in data_hdf5.keys():
    dataset[key] = np.array(data_hdf5[key])


state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)

#env = gym.make("maze2d-umaze-v1" )
env = gym.make("PushBalltoGoal-v0")
env = wrap_env(env, state_mean=state_mean, state_std=state_std)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
orthogonal_init = True

actor = TanhGaussianPolicy(
    state_dim, action_dim, max_action, orthogonal_init=orthogonal_init
).to("cpu")


actor.load_state_dict(state_dict = checkpoint["actor"])


keras_actor  = pytorch_to_keras(
        actor, torch.zeros((1, env.observation_space.shape[0])), verbose=True, name_policy="renumerate"
)

print(keras_actor)

obs = env.reset()
done = False
while True:
    action = actor.act(obs, "cpu")
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()


