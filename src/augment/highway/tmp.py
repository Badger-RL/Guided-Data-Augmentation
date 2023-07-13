import numpy as np
from stable_baselines3 import PPO, TD3, SAC, DQN, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym

env_id = 'highway-v0'
env_kwargs={'render_mode': 'human',}
env = gym.make(env_id, **env_kwargs)

obs, _ = env.reset()
env.set_state(obs)

ret = 0
H = 0
while True:
    action = env.action_space.sample()
    action = np.zeros_like(action)
    next_obs, rewards, terminated, truncated, info = env.step(action)

    obs = obs.reshape(8,5)
    print(obs[:,1:3])
    print(rewards)
    ret += rewards
    H += 1
    obs = next_obs
    if terminated or truncated:
        env.set_state(obs)

        # obs, _ = env.reset()
