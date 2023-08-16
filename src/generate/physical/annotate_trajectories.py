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

    env = gym.make('PushBallToGoalEasy-v0')

    traj_i = 0
    for path in paths:
        traj_i += 1
        dataset = None
        with open(path, 'r') as input_file:
            dataset = json.load(input_file)



        # reorder obs to match what I use in abstractim
        observations = np.array(dataset["observations"])
        observations = np.concatenate([observations[:, :2], observations[:, 3:5], observations[:, [2]]], axis=-1)
        if traj_i == 2:
            final_ball_pos = observations[-1, 2:4]
            goal_pos = np.array([4401, 0])
            delta = goal_pos - final_ball_pos
            observations[:, :2] += delta
            observations[:, 2:4] += delta
        # else:
        #     continue
            # trajectories["absolute_observations"] = np.array(trajectories["absolute_observations"])
            # trajectories["absolute_next_observations"] = np.array(trajectories["absolute_next_observations"])
            #
            # trajectories["absolute_observations"][:, :2] += delta
            # trajectories["absolute_observations"][:, 2:4] += delta
            # trajectories["absolute_next_observations"][:, :2] += delta
            # trajectories["absolute_next_observations"][:, 2:4] += delta

        # chop off last obs; we use it as the last next_obs
        obs = observations[:-1]
        next_obs = observations[1:]
        trajectories["absolute_observations"].extend(obs)
        trajectories["absolute_next_observations"].extend(next_obs)
        actions = np.array(dataset["actions"])[:-1]
        actions[:, -1] = 0
        actions = np.clip(actions, -1, 1)
        # norms = np.linalg.norm(actions, axis=-1)
        # actions /= norms.reshape(-1, 1)
        trajectories["actions"].extend(actions)



        env.reset()
        for o, no in zip(obs, next_obs):
            env.set_state(o[:2], o[2:4], o[-1])
            reward, ball_is_at_goal, ball_is_out_of_bounds = env.calculate_reward(no)
            trajectories["rewards"].append(reward)
            trajectories["terminals"].append(ball_is_at_goal or ball_is_out_of_bounds)
            trajectories["dones"].append(False)
            if reward > 1:
                print('goal')
            # if reward < 0:
            #     print('out of bounds')
            # print('out of bounds:', ball_is_out_of_bounds)

            trajectories["observations"].append(env.get_obs(o[:2], o[2:4], o[-1]))
            trajectories["next_observations"].append(env.get_obs(no[:2], no[2:4], no[-1]))

            # env.render()


        print([str(len(trajectories[key]) )for key in trajectories.keys()])


        for i in range(1500 - len(observations)):
            obs = trajectories['absolute_observations'][-1].copy()
            next_obs = trajectories['absolute_next_observations'][-1].copy()

            noise = np.random.uniform(-10, 10, size=(2,))
            obs[:2] += noise
            next_obs[:2] += noise

            reward, ball_is_at_goal, ball_is_out_of_bounds = env.calculate_reward(next_obs)
            trajectories["rewards"].append(reward)
            trajectories["actions"].append(trajectories["actions"][-1])
            trajectories["terminals"].append(ball_is_at_goal or ball_is_out_of_bounds)
            trajectories["dones"].append(False)
            if reward > 1:
                print('goal')
            # if reward < 0:
            #     print('out of bounds')


            trajectories["observations"].append(env.get_obs(obs[:2], obs[2:4], obs[-1]))
            trajectories["next_observations"].append(env.get_obs(next_obs[:2], next_obs[2:4], next_obs[-1]))
            trajectories["absolute_observations"].append(obs)
            trajectories["absolute_next_observations"].append(next_obs)

        trajectories["dones"][-1] = True
        trajectories["terminals"][-1] = True

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
    argv[1] = 'best'
    argv[2] = '../../datasets/PushBallToGoal-v0/physical/no_aug.hdf5'
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
            print(i)


    if "--json" in argv:
        with open(argv[2], 'w') as output_file:
            json.dump(dataset, output_file)
    else:
        out_file = h5py.File(argv[2], 'w')
        npify(dataset)
        for k in dataset:
            out_file.create_dataset(k, data=dataset[k], compression='gzip')


