import json
import sys
from custom_envs.push_ball_to_goal import PushBallToGoalEnv
import h5py
import numpy as np

#this file is for annotating trajectories returned from the physical robots with rewards and terminals.

env = PushBallToGoalEnv()


def annotate_trajectories(trajectories):
    trajectories["terminals"] = [False for i in range(len(trajectories["observations"]))]
    trajectories["terminals"][-1] = True
    trajectories["observations"] = [observation[4:] for observation in trajectories["observations"]]
    trajectories["next_observations"] = [observation[4:] for observation in trajectories["next_observations"]]
    trajectories["rewards"] = []
    for observation in trajectories["next_observations"]:
        env.set_state_from_obs(observation)
        trajectories["rewards"].append(env.calculate_reward())

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


    dataset = None
    with open(sys.argv[1], 'r') as input_file:
        dataset = json.load(input_file)
    dataset = annotate_trajectories(dataset)

    print(dataset)


    if "--json" in sys.argv:
        with open(sys.argv[2], 'w') as output_file:
            json.dump(dataset, output_file)
    else:
        out_file = h5py.File(sys.argv[2], 'w')
        npify(dataset)
        for k in dataset:
            out_file.create_dataset(k, data=dataset[k], compression='gzip')

    
