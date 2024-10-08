import argparse
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import json

def visualize_recorded_rollout(dataset, save_path, show_actions=False, single_episode=False):
    agent_x_list = []
    agent_y_list = []
    action_x_list = []
    action_y_list = []
    ball_x_list = []
    ball_y_list = []

    observations = dataset["absolute_observations"]
    next_observations = dataset["absolute_next_observations"]

    fig, ax = plt.subplots()

    robot_pos = observations[:, :2]
    next_robot_pos = next_observations[:, :2]
    ball_pos = observations[:, 2:4]
    next_ball_pos = next_observations[:, 2:4]
    plt.scatter(robot_pos[:, 0], robot_pos[:, 1], c=[i for i in range(len(robot_pos))], cmap="Greens")
    plt.scatter(next_robot_pos[:, 0], next_robot_pos[:, 1], c=[i for i in range(len(robot_pos))], cmap="Greens")

    plt.scatter(ball_pos[:, 0], ball_pos[:, 1], c=[i for i in range(len(ball_pos))], cmap="Reds")
    plt.scatter(next_ball_pos[:, 0], next_ball_pos[:, 1], c=[i for i in range(len(ball_pos))], cmap="Reds")

    # plt.scatter(ball_pos[:, 0], ball_pos[:, 1], c=[i for i in range(len(robot_pos))], cmap="Greens")

    delta = next_ball_pos - ball_pos
    u = delta[:, 0]
    v = delta[:, 1]

    mask = (np.abs(u)>0) | (np.abs(v)>0)
    x = ball_pos[:, 0]
    x = x[mask]
    y = ball_pos[:, 1]
    y = y[mask]
    u = u[mask]
    v = v[mask]

    plt.quiver(x,y, u,v, color='r')

    delta = next_robot_pos - robot_pos
    u = delta[:, 0]
    v = delta[:, 1]
    mask = (np.abs(u)>0) | (np.abs(v)>0)
    x = robot_pos[:, 0]
    x = x[mask]
    y = robot_pos[:, 1]
    y = y[mask]
    u = u[mask]
    v = v[mask]
    plt.quiver(x,y, u,v, color='g')

    # plt.scatter(
    #     x,y, c=[i for i in range(len(x))], cmap="Greens"
    # )
    if show_actions:
        agent_points = [(x, y) for x, y in zip(agent_x_list, agent_y_list)]
        action_points = [(x, y) for x, y in zip(action_x_list, action_y_list)]
        lines = [
            [agent_point, action_point]
            for agent_point, action_point in zip(agent_points, action_points)
        ]

        col = LineCollection(lines)
        ax.add_collection(col)

    # plt.xlim(-6000, 6000)
    # plt.ylim(-4500, 4500)
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='../datasets/PushBallToGoal-v0/guided_neg.hdf5')
    parser.add_argument('--save-dir', type=str, default='.')
    parser.add_argument('--save-name', type=str, default='tmp')
    parser.add_argument('--single-episode', type=bool, default=False)
    parser.add_argument('--show-actions', type=bool, default=False)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = f'{args.save_dir}/{args.save_name}'

    dataset = {}
    data_hdf5 = h5py.File(args.dataset_path, "r")
    for key in data_hdf5.keys():
        start = 000
        end = 100000
        dataset[key] = np.array(data_hdf5[key][start:end])

    visualize_recorded_rollout(dataset, save_path, show_actions=args.show_actions, single_episode=args.single_episode)