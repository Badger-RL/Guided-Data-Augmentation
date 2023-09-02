import copy
import json
import sys

import gym, custom_envs
from custom_envs.push_ball_to_goal import PushBallToGoalEnv
import h5py
import os
import numpy as np
import natsort

#this file is for annotating trajectories returned from the physical robots with rewards and terminals.

env = gym.make('PushBallToGoalHard-v0')


EPISODE_BATCH_LENGTH = 2

def annotate_trajectories(paths):

    trajectories = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "terminals": [],
        "dones": [],

        "absolute_observations": [],
        "absolute_next_observations": [],

    }

    for path_i, path in enumerate(paths):
        dataset = None
        with open(path, 'r') as input_file:
            dataset = json.load(input_file)

        npify(dataset)

        # reorder obs to match what I use in abstractim
        observations = dataset["observations"]
        observations = np.concatenate([observations[:, :2], observations[:, 3:5], observations[:, [2]]], axis=-1)
        # chop off last obs; we use it as the last next_obs
        obs = observations[:-1]
        next_obs = observations[1:]

        actions = dataset["actions"]
        actions[:, -1] = 0 #np.random.uniform(-1, 1, size=len(actions))
        actions = np.clip(actions, -1, 1)

        robot_displacement = np.linalg.norm(next_obs[:,:2] - obs[:,:2], axis=-1)
        ball_displacement = np.linalg.norm(next_obs[:,2:4] - obs[:,2:4], axis=-1)

        mask = (ball_displacement > 500) | (robot_displacement > 500)
        # trajectories["terminals"].extend(mask)
        # trajectories["dones"].extend(mask)

        env.reset()
        i = 0
        for obs, next_obs in zip(obs, next_obs):
            env.set_state(obs[:2], obs[2:4], obs[-1])

            # if mask[i] or (path_i == 2 and (i >= 0 and i <= 600)):
            if mask[i]:
                print('i' ,i)
                i += 1
                continue
                # next_obs = obs
            reward, ball_is_at_goal, ball_is_out_of_bounds = env.calculate_reward(next_obs)
            trajectories["rewards"].append(reward)
            if ball_is_at_goal:
                print('goal')
            if ball_is_out_of_bounds:
                print('out of bounds')
            #     print(obs[:2])

            # displacement = np.linalg.norm(next_obs[:2] - obs[:2])
            # trajectories["terminals"].append(False)
            # trajectories["dones"].append(False)


            trajectories["observations"].append(env.get_obs(obs[:2], obs[2:4], obs[-1]))
            trajectories["next_observations"].append(env.get_obs(next_obs[:2], next_obs[2:4], next_obs[-1]))
            trajectories["absolute_observations"].append(np.concatenate([obs[:2], obs[2:4], [obs[-1]]]))
            trajectories["absolute_next_observations"].append(np.concatenate([next_obs[:2], next_obs[2:4], [next_obs[-1]]]))
            trajectories["actions"].append(actions[i])
            trajectories["dones"].append(False)
            # trajectories["terminals"].append(False)
            trajectories["terminals"].append(ball_is_at_goal)


            # env.render()
            i += 1


        print([str(len(trajectories[key]) )for key in trajectories.keys()])


    for i in range(300):
        obs = trajectories['absolute_observations'][-1].copy()
        next_obs = trajectories['absolute_next_observations'][-1].copy()

        noise = np.random.uniform(-10, 10, size=(2,))
        obs[:2] += noise
        next_obs[:2] += noise

        reward, ball_is_at_goal, ball_is_out_of_bounds = env.calculate_reward(next_obs)
        trajectories["rewards"].append(reward)
        trajectories["actions"].append(env.action_space.sample())
        trajectories["terminals"].append(ball_is_out_of_bounds)
        trajectories["dones"].append(False)
        if reward > 1:
            print('goal')
        if reward < 0:
            print('out of bounds')


        trajectories["observations"].append(env.get_obs(obs[:2], obs[2:4], obs[-1]))
        trajectories["next_observations"].append(env.get_obs(next_obs[:2], next_obs[2:4], next_obs[-1]))
        trajectories["absolute_observations"].append(obs)
        trajectories["absolute_next_observations"].append(next_obs)

    return trajectories


def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts', 'dones']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


if __name__ == "__main__":



    # if not len(sys.argv) >= 3:
    #     print("usage: python3 ./annotate_trajectories.py <input_path> <output_path>")
    #     exit()

    argv = [None, None, None]
    # argv[1] = 'simrobot_normal/traj3'
    # argv[2] = '../../../datasets/PushBallToGoal-v0/simrobot/no_aug_3.hdf5'
    argv[1] = 'physical_data_clean/scoring_clean_3'
    argv[1] = 'guda_2'
    argv[2] = '../../../datasets/PushBallToGoalHard-v0/no_aug.hdf5'
    # argv[1] = 'best'
    # argv[2] = 'best.hdf5'
    trajectory_files = []

    for path, subdirs, files in os.walk(argv[1]):
        print(files)
        files = natsort.natsorted(files)
        for name in files:

            if "trajectories_" in name:
                print(name)
                trajectory_files.append(os.path.join(path, name))

    dataset = annotate_trajectories(trajectory_files)

    # print(dataset)
    print(len(dataset["observations"]))
    for i in range(len(dataset["observations"])):
        if dataset["terminals"][i]:
            print('terminal', i)

    npify(dataset)
    obs = dataset['observations']
    next_obs = dataset['next_observations']
    displacement = np.linalg.norm(next_obs[:,:2] - obs[:,:2], axis=-1)
    mask = displacement > 500

    mask = np.ones(len(dataset["observations"]), dtype=bool)
    mask[731] = False
    for k, v in dataset.items():
        dataset[k] = v[mask]

    n = 900
    ball_init = np.empty(shape=(n,2))
    noise = np.random.uniform(-50, 50, size=(n,2))
    ball_init[:] = dataset["absolute_observations"][n, 2:4] + noise
    dataset["absolute_observations"][:n, 2:4] = ball_init
    dataset["absolute_next_observations"][:n, 2:4] = ball_init

    dataset["absolute_observations"][:732, :2] += 300
    dataset["absolute_next_observations"][:732, :2] += 300

    dataset["absolute_observations"][:1170, 2] += -300
    dataset["absolute_next_observations"][:1170, 2] += -300

    # dataset["absolute_observations"][:732, :2] += 300
    # dataset["absolute_next_observations"][:732, :2] += 300

    dataset["dones"][-1] = True
    dataset["terminals"][-1] = True

    if "--json" in argv:
        with open(argv[2], 'w') as output_file:
            json.dump(dataset, output_file)
    else:
        out_file = h5py.File(argv[2], 'w')
        for k in dataset:
            out_file.create_dataset(k, data=dataset[k], compression='gzip')


