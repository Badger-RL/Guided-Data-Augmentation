import gymnasium as gym
from matplotlib import pyplot as plt

env = gym.make('parking-v0', render_mode='rgb_array')

# Visualize initial state
obs, _ = env.reset()
plt.imshow(env.render())
plt.show()

# set_state() should work with dict observations
obs["observation"][0] = 26
obs["observation"][1] = 15
env.set_state(obs)

plt.imshow(env.render())
plt.show()

# set_state() should also work with np.array observations
obs = obs["observation"]
obs[0] = -26
obs[1] = -15
env.set_state(obs)

plt.imshow(env.render())
plt.show()




