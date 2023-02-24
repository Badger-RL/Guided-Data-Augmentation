#Derived from D4RL
#https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
#https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE

import numpy as np
import h5py
import argparse


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.env_util import make_vec_env

# from src.envs.push_ball_to_goal import PushBallToGoalEnv

# from custom_envs.push_ball_to_goal import PushBallToGoalEnv
from custom_envs.push_ball_to_goal import PushBallToGoalEnv

models = {"push_ball_to_goal": {"env": PushBallToGoalEnv}}

def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'next_observations': [],
            }

def append_data(data, s, a, r, ns, done):
    data['observations'].append(s)
    data['next_observations'].append(ns)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)

def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=int(10e4), help='Num samples to collect')
    parser.add_argument('--render', type=bool, default=False)

    args = parser.parse_args()

    policy_path = f"../expert_policies/push_ball_to_goal/policy"
    normalization_path = f"../expert_policies/push_ball_to_goal/vector_normalize"

    env = VecNormalize.load(
    normalization_path, make_vec_env(models["push_ball_to_goal"]["env"], n_envs=1)
    )
    # env = PushBallToGoalEnv()
    env.norm_obs = True
    env.norm_reward = False
    env.clip_obs = 1.0
    env.training = False

    s = env.reset()


    custom_objects = {
    "lr_schedule": lambda x: .003,
    "clip_range": lambda x: .02
    }
    policy = PPO.load(policy_path, custom_objects = custom_objects, env= env)

    ts = 0
    num_episodes = 0
    rets = []
    succeses = []
    ret = 0
    for _ in range(args.num_samples):
        act = policy.predict(s)[0]


        ns, r, done, info = env.step(act)
        ret += r
        if args.render:
            env.render()
        timeout = False

        ts += 1

        if done or timeout:
            print(ts)
            if ts < 500:
                succeses.append(1)
            else:
                succeses.append(0)
            ts = 0
            s = env.reset()

            num_episodes += 1
            rets.append(ret)
            ret = 0
        else:
            s = ns

    rets = np.array(rets)
    print('avg_return: ', np.average(rets), 'success_rate: ', np.average(succeses))


if __name__ == '__main__':
    main()