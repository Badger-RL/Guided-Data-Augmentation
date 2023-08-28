import argparse
import os.path

import gymnasium as gym
import highway_env
import h5py
import numpy as np
from stable_baselines3 import SAC, DDPG, DQN, PPO
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

    desired_goal = []
    achieved_goal = []


    episode_count = 0
    step_count = 0
    # while episode_count < num_episodes:
    while step_count < num_samples:
        episode_count += 1
        ep_observations, ep_next_observations, ep_actions, ep_rewards, ep_dones, ep_infos,  = [], [], [], [], [], []
        ep_desired_goal, ep_achieved_goal = [], []
        ep_step_count = 0
        obs, _ = env.reset()
        done = False

        while not done:

            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample() # np.random.uniform(-1, +1, size=env.action_space.shape)

            ep_observations.append(obs['observation'])
            ep_desired_goal.append(obs['desired_goal'])

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated


            # action_dict = env.controlled_vehicles[0].action
            # action = np.array([action_dict['steering'], action_dict['acceleration']])
            # action[0] += np.random.uniform(-0.05, +0.05)
            # action[1] += np.random.uniform(-0.05, +0.05)

            ep_actions.append(action)
            ep_next_observations.append(obs['observation'])
            ep_rewards.append(reward)
            ep_dones.append(done)
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
            desired_goal.extend(ep_desired_goal)
            next_observations.extend(ep_next_observations)
            actions.extend(ep_actions)
            rewards.extend(ep_rewards)
            dones.extend(ep_dones)
            infos.extend(ep_infos)
        else:
            observations.append(ep_observations)
            desired_goal.append(ep_desired_goal)
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
        'desired_goal': np.array(desired_goal)
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", help="RL Algorithm", default='ddpg', type=str)
    parser.add_argument("--env_id", type=str, default="parking-v0", help="environment ID")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=int(100), help='Num samples to collect')
    parser.add_argument('--policy_path', type=str, default='file_name')
    parser.add_argument('--save_dir', type=str, default='tmp_dir')
    parser.add_argument('--save_name', type=str, default='tmp_name.hdf5')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--skip_terminated_episodes', type=int, default=False)

    args = parser.parse_args()

    env_kwargs = {'render_mode': 'human'}
    # env_kwargs = {}
    env = gym.make(args.env_id, **env_kwargs)


    args.policy_path = f'../../policies/{args.env_id}/{args.algo}/best_model.zip'
    # model = DQN.load(args.policy_path)
    model = DDPG.load(args.policy_path, env)

    # args.policy_path = f'../../results/{args.env_id}/rl_model_3000_steps.zip'
    # model = PPO.load(args.policy_path)

    data = simulate(env=env, model=model, num_samples=args.num_samples, render=False, skip_terminated_episodes=args.skip_terminated_episodes)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    fname = f'{save_dir}/{args.save_name}'
    dataset = h5py.File(fname, 'w')
    for k in data:
        print(k, data[k].shape)
        dataset.create_dataset(k, data=data[k], compression='gzip')

