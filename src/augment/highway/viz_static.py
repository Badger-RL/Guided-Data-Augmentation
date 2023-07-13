import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from augment.highway.highway import ChangeLaneGuided, CreateCollision

env = gym.make('highway-v0', render_mode='rgb_array')
obs, _ = env.reset()
for _ in range(1):
    action = env.action_space.sample()
    action = np.zeros_like(action)
    next_obs, reward, done, truncated, info = env.step(action)
    env.render()

next_obs = obs.copy()
action = env.action_space.sample()
reward = 1
done = True
plt.imshow(env.render())
plt.show()
obs_prev = env.observation_type.observe()

f = ChangeLaneGuided(env)
f = CreateCollision(env)
aug_obs, aug_action, aug_reward, aug_next_obs, aug_done = f.augment(obs, action, next_obs, reward, done)
env.set_state(aug_obs)
# true_next_obs, true_reward, true_done, true_truncated, true_info = env.step(aug_action)

plt.imshow(env.render())
plt.show()


obs_after = env.observation_type.observe()

print(obs_prev[0:2])
print(obs_after[0:2])
aug_obs = aug_obs.reshape(8,-1)
print(aug_obs[0:2])

