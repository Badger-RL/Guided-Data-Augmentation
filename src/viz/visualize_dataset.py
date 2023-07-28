import argparse
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import json


def get_coords(observation, action):
    """Function to extract absolute coordinates from and observation and an action."""
    ball_agent_x_offset = observation[0] * 9000
    ball_agent_y_offset = observation[1] * 6000
    goal_ball_x_offset = observation[2] * 9000
    goal_ball_y_offset = observation[3] * 6000

    ball_robot_angle_offset_sin = observation[4]
    ball_robot_angle_offset_cos = observation[5]
    goal_robot_angle_offset_sin = observation[6]
    goal_robot_angle_offset_cos = observation[7]

    target_x = 4800 - goal_ball_x_offset
    target_y = 0 - goal_ball_y_offset

    robot_x = target_x - ball_agent_x_offset
    robot_y = target_y - ball_agent_y_offset

    ball_robot_angle_offset = np.arctan2(
        ball_robot_angle_offset_sin, ball_robot_angle_offset_cos
    )
    goal_robot_angle_offset = np.arctan2(
        goal_robot_angle_offset_sin, goal_robot_angle_offset_cos
    )

    robot_angle_to_ball = np.arctan2(ball_agent_y_offset, ball_agent_x_offset)

    robot_angle_to_ball = robot_angle_to_ball

    robot_angle = robot_angle_to_ball - ball_robot_angle_offset

    policy_target_x = robot_x + (
            (
                    (np.cos(robot_angle) * np.clip(action[1], -1, 1))
                    + (np.cos(robot_angle + np.pi / 2) * np.clip(action[2], -1, 1))
            )
            * 200
    )  # the x component of the location targeted by the high level action
    policy_target_y = robot_y + (
            (
                    (np.sin(robot_angle) * np.clip(action[1], -1, 1))
                    + (np.sin(robot_angle + np.pi / 2) * np.clip(action[2], -1, 1))
            )
            * 200
    )  # the y component of the location targeted by the high level action

    return robot_x, robot_y, target_x, target_y, policy_target_x, policy_target_y


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

    plt.xlim(-5000, 5000)
    plt.ylim(-3500, 3500)
    plt.xlabel('x position')
    plt.ylabel('y position')
    # plt.show()

    plt.savefig(save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/Users/nicholascorrado/code/offlinerl/GuidedDataAugmentationForRobotics/src/generate/abstractsim/tmp.hdf5')
    # parser.add_argument('--dataset-path', type=str,
    #                     default='/Users/nicholascorrado/code/offlinerl/GuidedDataAugmentationForRobotics/src/datasets/PushBallToGoal-v0/no_aug_40_5k.hdf5')

    parser.add_argument('--save-dir', type=str, default='./figures/PushBallToGoal-v0/')
    parser.add_argument('--save-name', type=str, default='tmp.png')
    parser.add_argument('--single-episode', type=bool, default=False)
    parser.add_argument('--show-actions', type=bool, default=False)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = f'{args.save_dir}/{args.save_name}'

    dataset = {}
    data_hdf5 = h5py.File(args.dataset_path, "r")
    for key in data_hdf5.keys():
        start = 5000
        end = 6000
        dataset[key] = np.array(data_hdf5[key][start:end])

    """
    print(len(dataset["terminals"]))


    for i in range(len(dataset["terminals"])-1):
        assert(dataset["terminals"][i] == 0)
    assert(dataset["terminals"][-1] == 1)
    assert(len(dataset["observations"]) == 2000)
    """

    visualize_recorded_rollout(dataset, save_path, show_actions=args.show_actions, single_episode=args.single_episode)