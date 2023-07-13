import argparse
import os.path

import gymnasium as gym
import h5py
import numpy as np
from stable_baselines3 import SAC, DDPG
from stable_baselines3.common.utils import set_random_seed


def simulate(env, model, num_samples, num_episodes=None,  seed=0, render=False, flatten=True, verbose=1, skip_terminated_episodes=False):
    set_random_seed(seed)

    observations = []
    next_observations = []
    actions = []
    rewards = []
    returns = []
    dones = []
    infos = []


    episode_count = 0
    step_count = 0
    # while episode_count < num_episodes:
    while step_count < num_samples:
        episode_count += 1
        ep_observations, ep_next_observations, ep_actions, ep_rewards, ep_dones, ep_infos,  = [], [], [], [], [], []
        ep_step_count = 0
        obs, _ = env.reset()
        done = False

        while not done:

            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample() # np.random.uniform(-1, +1, size=env.action_space.shape)

            ep_actions.append(action)
            ep_observations.append(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_next_observations.append(obs)
            ep_rewards.append(reward)
            ep_dones.append(terminated)
            ep_infos.append(info)
            ep_step_count += 1

            if render: env.render()

        if skip_terminated_episodes and info['crashed']:
            episode_count -= 1
            continue

        step_count += ep_step_count

        returns.append(sum(ep_rewards))
        if flatten:
            observations.extend(ep_observations)
            next_observations.extend(ep_next_observations)
            actions.extend(ep_actions)
            rewards.extend(ep_rewards)
            dones.extend(ep_dones)
            infos.extend(ep_infos)
        else:
            observations.append(ep_observations)
            next_observations.append(ep_next_observations)
            actions.append(ep_actions)
            rewards.append(ep_rewards)
            dones.append(ep_dones)
            infos.append(ep_infos)
        if verbose:
            print(f'num_steps: {step_count}, episode {episode_count}, return: {returns[-1]}',)

    print(f'average return: {np.mean(returns)}')
    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'next_observations': np.array(next_observations),
        'terminals': np.array(dones),
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", help="RL Algorithm", default='sac', type=str)
    parser.add_argument("--env_id", type=str, default="highway-v0", help="environment ID")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=int(10e3), help='Num samples to collect')
    parser.add_argument('--policy_path', type=str, default='file_name')
    parser.add_argument('--save_dir', type=str, default='tmp_dir')
    parser.add_argument('--save_name', type=str, default='tmp_name')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--skip_terminated_episodes', type=int, default=True)

    args = parser.parse_args()

    env_kwargs = {'render_mode': 'human'}
    env_kwargs = {}
    env = gym.make(args.env_id, **env_kwargs)

    model = DDPG.load(args.policy_path)

    data = simulate(env=env, model=model, num_samples=args.num_samples, render=False, skip_terminated_episodes=args.skip_terminated_episodes)


    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    fname = f'{save_dir}/{args.save_name}'
    dataset = h5py.File(fname, 'w')
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')

