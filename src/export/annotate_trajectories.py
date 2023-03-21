import json
import sys


#this file is for annotating trajectories returned from the physical robots with rewards and terminals.


def annotate_trajectories(trajectories):
    trajectories["terminals"] = [False for i in range(len(trajectories["observations"]))]
    trajectories["terminals"][-1] = True
    trajectories["observations"] = [observation[4:] for observation in trajectories["observations"]]
    trajectories["next_observations"] = [observation[4:] for observation in trajectories["next_observations"]]
    return trajectories

if __name__ == "__main__":

    if not len(sys.argv) == 3:
        print("usage: python3 ./annotate_trajectories.py <input_path> <output_path>")
        exit()


    dataset = None
    with open(sys.argv[1], 'r') as input_file:
        dataset = json.load(input_file)
    dataset = annotate_trajectories(dataset)

    print(dataset)

    with open(sys.argv[2], 'w') as output_file:
        json.dump(dataset, output_file)

    
