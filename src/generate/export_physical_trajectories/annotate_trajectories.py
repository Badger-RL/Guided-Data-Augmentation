import json
import sys
from custom_envs.push_ball_to_goal import PushBallToGoalEnv
import h5py
import os
import numpy as np

#this file is for annotating trajectories returned from the physical robots with rewards and terminals.

env = PushBallToGoalEnv()


EPISODE_BATCH_LENGTH = 2

def annotate_trajectories(paths):

    trajectories = {
        "terminals": [],
        "observations": [],
        "next_observations": [],
        "rewards": [],
        "actions": [],
    }

    for path in paths:
        dataset = None
        with open(path, 'r') as input_file:
            dataset = json.load(input_file)
        new_terminals = [False for i in range(len(dataset["observations"]))]
        if f"{str(EPISODE_BATCH_LENGTH)}.log" in path:
            new_terminals[-1] = True
        trajectories["terminals"] += new_terminals
        trajectories["observations"] += [observation[4:] for observation in dataset["observations"]]
        trajectories["actions"] += [action for action in dataset["actions"]]
        trajectories["next_observations"] += [observation[4:] for observation in dataset["next_observations"]]
        for observation in dataset["next_observations"]:
           # print(observation)
            env.set_state_from_obs(observation[4:])
            env.render()
            print(env.calculate_reward())
            trajectories["rewards"].append(env.calculate_reward())
        print([str(len(trajectories[key]) )for key in trajectories.keys()])

    return trajectories


def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


if __name__ == "__main__":

    

    if not len(sys.argv) >= 3:
        print("usage: python3 ./annotate_trajectories.py <input_path> <output_path>")
        exit()


    trajectory_files = []

    for path, subdirs, files in os.walk(sys.argv[1]):
        print(files)
        files = sorted(files)
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


    if "--json" in sys.argv:
        with open(sys.argv[2], 'w') as output_file:
            json.dump(dataset, output_file)
    else:
        out_file = h5py.File(sys.argv[2], 'w')
        npify(dataset)
        for k in dataset:
            out_file.create_dataset(k, data=dataset[k], compression='gzip')

    
