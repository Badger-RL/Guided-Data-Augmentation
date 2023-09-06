import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from src.generate.utils import reset_data, append_data, load_dataset, npify
import time
import h5py


dataset_path = f"/Users/yxqu/Desktop/Research/GuDA/GuidedDataAugmentationForRobotics/src/datasets/parking-v0/no_aug.hdf5"
observed_dataset = load_dataset(dataset_path)

n = len(observed_dataset['observations'])
env = gym.make('parking-v0', render_mode='rgb_array')
env.reset()
for i in range(n):
    obs = observed_dataset['observations'][i]
    obs[0] *= 100
    obs[1] *= 100
    obs[2] *= 5
    obs[3] *= 5
    env.set_state(observed_dataset['observations'][i])
    env.render()
    time.sleep(0.1)