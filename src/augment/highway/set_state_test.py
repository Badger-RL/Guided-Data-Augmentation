import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('parking-v0', render_mode='rgb_array')

# Visualize initial state
obs, _ = env.reset()
plt.imshow(env.render())
plt.show()

# set_state() should work with dict observations
theta = np.pi/4
obs["observation"][0] = 26
obs["observation"][1] = 15
obs["observation"][3] = np.cos(theta)
obs["observation"][4] = np.sin(theta)
env.set_state(obs)

plt.imshow(env.render())
plt.show()

# set_state() should also work with np.array observations
theta = -np.pi/4
obs = obs["observation"]
obs[0] = -26
obs[1] = -15
obs[3] = np.cos(theta)
obs[4] = np.sin(theta)
env.set_state(obs)

plt.imshow(env.render())
plt.show()




