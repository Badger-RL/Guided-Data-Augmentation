import argparse
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
import h5py
import numpy as np


# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -200, 200)
    g_clip = np.clip(g, -200, 200)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -5, 5)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -5, 5)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs


def simulate(env, policy, num_episodes, seed=0, render=False, flatten=True, verbose=1, skip_terminated_episodes=False):
    observations = []
    next_observations = []
    actions = []
    rewards = []
    returns = []
    dones = []
    infos = []

    episode_count = 0
    step_count = 0
    while episode_count < num_episodes:
        # while step_count < num_samples:
        episode_count += 1
        ep_observations, ep_next_observations, ep_actions, ep_rewards, ep_dones, ep_infos, = [], [], [], [], [], []
        ep_step_count = 0
        obs, _ = env.reset()
        done = False

        g = obs['desired_goal']

        while not done:

            # print(obs["observation"])
            inputs = process_inputs(obs["observation"], g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = policy(inputs)
            action = pi.detach().numpy().squeeze()

            # the next line concatenates creates a new variable called ob_to_save by concatenating together obs["observation"]

            ep_observations.append(np.concatenate([obs["observation"], obs["desired_goal"]]))

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_actions.append(action)
            ep_next_observations.append(np.concatenate([obs["observation"], obs["desired_goal"]]))
            ep_rewards.append(reward)
            ep_dones.append(terminated)
            # print(info["is_success"])
            ep_infos.append(int(info["is_success"]))
            ep_step_count += 1

            if render: env.render()

        ep_dones[-1] = True
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
            print(f'num_steps: {step_count}, episode {episode_count}, return: {returns[-1]}', )

    print(f'average return: {np.mean(returns)}')
    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'next_observations': np.array(next_observations),
        'terminals': np.array(dones),
        'infos': np.array(infos),
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", help="RL Algorithm", default='dqn', type=str)
    parser.add_argument("--env_id", type=str, default="FetchSlide-v2", help="environment ID")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=int(150), help='Num samples to collect')
    parser.add_argument('--policy_path', type=str, default='file_name')
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--save_name', type=str, default="dataset.hdf5")
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--skip_terminated_episodes', type=int, default=False)

    args = parser.parse_args()

    policy_path = args.policy_path

    o_mean, o_std, g_mean, g_std, model = torch.load(policy_path, map_location=lambda storage, loc: storage)
    # create the environment
    # env = gym.make(args.env_id, render_mode = "human")
    env = gym.make(args.env_id)

    _, _ = env.reset(seed=0)

    env_params = {'obs': 25, 'goal': 3, 'action': 4, 'action_max': 1.0}

    # create the actor network
    policy = actor(env_params)
    policy.load_state_dict(model)
    policy.eval()

    data = simulate(env=env, policy=policy, num_episodes=args.num_episodes, render=False,
                    skip_terminated_episodes=args.skip_terminated_episodes)

    env.close()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    fname = f'{args.save_dir}/{args.save_name}'
    dataset = h5py.File(fname, 'w')
    for k in data:
        print(k, data[k].shape)
        dataset.create_dataset(k, data=data[k], compression='gzip')
