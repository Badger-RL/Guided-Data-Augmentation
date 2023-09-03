import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from src.generate.utils import reset_data, append_data, load_dataset, npify
import time

dataset_path = f"/Users/yxqu/Desktop/Research/GuDA/GuidedDataAugmentationForRobotics/src/algorithms/discrete_BCQ/buffers/parking-v0_no_aug.hdf5"
observed_dataset = load_dataset(dataset_path)
n = len(observed_dataset['observations'])
print("keys: ", observed_dataset.keys())
env = gym.make('parking-v0', render_mode='rgb_array')

# Visualize initial state
obs, _ = env.reset()


# set_state() should work with dict observations
for i in range(n):
    observation = observed_dataset['observations'][i]
    obs["observation"]= observation
    env.set_state(obs)
    env.render()