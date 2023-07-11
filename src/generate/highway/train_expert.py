import torch
from stable_baselines3 import TD3, SAC, DDPG
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

env_id = "intersection-v0"
vec_env = make_vec_env(env_id, n_envs=1)
eval_env = make_vec_env(env_id, n_envs=1)

torch.set_num_threads(1)
model = DDPG("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4, learning_starts=100, batch_size=256, gamma=0.9, gradient_steps=-1,
            policy_kwargs={'net_arch': [256]})
eval_callback = EvalCallback(eval_env=eval_env, n_eval_episodes=10, eval_freq=10000,
                             best_model_save_path=f'results/{env_id}'
                             )
checkpoint_callback = CheckpointCallback(save_freq=500, save_path=f'results/{env_id}', save_vecnormalize=True, verbose=2)
model.learn(total_timesteps=1000000, callback=[checkpoint_callback, eval_callback],)
