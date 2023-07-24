import copy

import gym, custom_envs
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

env_id = "PushBallToGoal-v0"

# vec_env = DummyVecEnv([lambda: gym.make(env_id)])
# vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)
# vec_env = VecMonitor(vec_env)

vec_env = make_vec_env(env_id, n_envs=8)
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

eval_env = make_vec_env(env_id, n_envs=1)


torch.set_num_threads(1)
model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=2048, learning_rate=3e-4, batch_size=64,  ent_coef=0.01,
            policy_kwargs={'net_arch': [256]})
wandb_callback = WandbCallback(verbose=2)
eval_callback = EvalCallback(eval_env=eval_env, n_eval_episodes=10, eval_freq=100000,
                             best_model_save_path=f'results/{env_id}'
                             )
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=f'results/{env_id}', save_vecnormalize=True, verbose=2)
model.learn(total_timesteps=10000000, callback=[checkpoint_callback],)
model.save(f"ppo_{env_id}")