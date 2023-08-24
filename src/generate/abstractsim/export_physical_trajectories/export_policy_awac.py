from typing import Tuple, Union
from pathlib import Path
import json
import sys

import supersuit

import gym
import h5py
import numpy as np
import torch

import algorithms.awac
from src.AbstractSim.multi_agent.multi_rew.multi_rew import parallel_env
from src.algorithms.cql import ContinuousCQL, TanhGaussianPolicy, ReparameterizedTanhGaussian, FullyConnectedQFunction, Scalar

#from envs.push_ball_to_goal import PushBallToGoalEnv
from src.pytorch2keras.pytorch2keras import pytorch_to_keras
from stable_baselines3.ppo import PPO

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

# checkpoint = torch.load(Path(sys.argv[2]))

dataset = {}
data_hdf5 = h5py.File('../../../datasets/PushBallToGoalEasy-v0/physical/guided.hdf5', "r")
for key in data_hdf5.keys():
    dataset[key] = np.array(data_hdf5[key])


eps = 1e-3
state_mean, state_std = compute_mean_std(dataset["observations"], eps=eps)

# env = gym.make("maze2d-umaze-v1" )
env = gym.make("PushBallToGoalEasy-v0")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
orthogonal_init = True


metadata = {
    "observation_length": sum(env.observation_space.shape),
    "action_length": sum(env.action_space.shape),
    "policy_type": "CORL_CQL"
}
metadata["mean"] = [float(entry) for entry in state_mean]
metadata["var"] = [float(std) for std in state_std]
metadata["clip"] = 10000000000
metadata["epsilon"] = eps
print(metadata)
with open("metadata.json", "w") as metadata_file:
    json.dump(metadata, metadata_file)

actor = algorithms.awac.Actor(
    state_dim, action_dim, max_action=max_action, hidden_dim=128, n_layers=1,
).to("cpu")


path = '/Users/nicholascorrado/code/offlinerl/GuidedDataAugmentationForRobotics/src/results/PushBallToGoalEasy-v0/no_aug/awac/nl_1/hd_128/lr_3e-05/l_2/run_34/model.pt'
checkpoint = torch.load(path)
actor.load_state_dict(state_dict = checkpoint["actor"])


keras_actor = pytorch_to_keras(
        actor.mlp, torch.zeros((1, env.observation_space.shape[0])), verbose=True, name_policy="renumerate"
)

keras_actor.save("policy.h5", save_format="h5")

print(keras_actor)


#obs = np.array([-1.3579835,   0.27951971,  0.95481869,  0.41910834 , 2.60238981 ,-7.56153716,
#  2.02996028 , 1.04410311])

# obs = env.reset()
# done = False
# while True:
#     keras_obs = np.array(obs).reshape((1,8))
#     print(obs)
#     print(keras_obs)
#     keras_output = keras_actor.predict(keras_obs)
#
#     keras_action = keras_output[0,:3]
#     print("KERAS ACTION")
#     print(keras_action)
#     keras_log_stds = keras_output[0,3:]
#     keras_log_stds -= 1
#     keras_log_stds = np.clip(keras_log_stds, -20.0, 2.0)
#     keras_stds = np.exp(keras_log_stds)
#     keras_action = np.tanh(keras_action) * float(env.action_space.high[0])
#
#
#
#
#
#     action = actor.act(obs, "cpu")
#     print(action)
#     print(keras_action)
#     print(keras_log_stds)
#
#
#     obs, reward, done, info = env.step(keras_action)
#     env.render()
#     if done:
#         obs = env.reset()


