import torch
import algorithms.td3_bc
import numpy as np
import argparse

import gym, custom_envs

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
    parser.add_argument('--render', type=bool, default=True)

    args = parser.parse_args()

    env = gym.make('PushBallToGoal-v2',
                   # init_robot_x_range=(4400, 4400),
                   # init_robot_y_range=(-3000, -3000),
                   # init_ball_x_range=(-1000,+1000),
                   # init_ball_y_range=(-1000,1000)
                   )

    state_dict = torch.load('/Users/nicholascorrado/code/offlinerl/GuidedDataAugmentationForRobotics/src/results/run_307/model.pt')
    policy = algorithms.td3_bc.Actor(
        12, 4, 1, 128, 1
    )
    policy.load_state_dict(state_dict['actor'])
    s = env.reset()

    ts = 0
    num_episodes = 0
    rets = []
    succeses = []
    ret = 0
    for _ in range(args.num_samples):
        act = policy(torch.tensor(s)).detach().numpy()


        ns, r, done, info = env.step(act)
        ret += r
        env.render()
        timeout = False

        ts += 1

        if done or timeout:
            print(ts)
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