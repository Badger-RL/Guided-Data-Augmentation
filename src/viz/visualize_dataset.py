

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

def visualize_recorded_rollout(dataset, single_episode = False):

    agent_x_list = []
    agent_y_list = []
    action_x_list = []
    action_y_list = []
    ball_x_list = []
    ball_y_list = []



    for observation, action, done in zip(dataset["observations"],dataset["actions"], dataset["terminals"]):
        print(action)
        print(observation)
        robot_x, robot_y, target_x, target_y, action_x, action_y = get_coords(
            observation, action
        )
        agent_x_list.append(robot_x)
        agent_y_list.append(robot_y)
        action_x_list.append(action_x)
        action_y_list.append(action_y)
        ball_x_list.append(target_x)
        ball_y_list.append(target_y)
        if done and single_episode:
            fig, ax = plt.subplots()

            agent_points = [(x, y) for x, y in zip(agent_x_list, agent_y_list)]
            action_points = [(x, y) for x, y in zip(action_x_list, action_y_list)]
            lines = [
                [agent_point, action_point]
                for agent_point, action_point in zip(agent_points, action_points)
            ]

            col = LineCollection(lines)
            ax.add_collection(col)

            plt.scatter(
                agent_x_list,
                agent_y_list,
                c=[i for i in range(len(agent_x_list))],
                cmap="Blues",
            )
            plt.scatter(
                ball_x_list, ball_y_list, c=[i for i in range(len(ball_x_list))], cmap="Greens"
            )
            plt.scatter(
                action_x_list,
                action_y_list,
                c=[i for i in range(len(action_x_list))],
                cmap="Reds",
            )
            

            plt.show()
            agent_x_list = []
            agent_y_list = []
            action_x_list = []
            action_y_list = []
            ball_x_list = []
            ball_y_list = []
    if not single_episode:
        fig, ax = plt.subplots()

        agent_points = [(x, y) for x, y in zip(agent_x_list, agent_y_list)]
        action_points = [(x, y) for x, y in zip(action_x_list, action_y_list)]
        lines = [
            [agent_point, action_point]
            for agent_point, action_point in zip(agent_points, action_points)
        ]

        col = LineCollection(lines)
        ax.add_collection(col)

        plt.scatter(
            agent_x_list,
            agent_y_list,
            c=[i for i in range(len(agent_x_list))],
            cmap="Blues",
        )
        plt.scatter(
            ball_x_list, ball_y_list, c=[i for i in range(len(ball_x_list))], cmap="Greens"
        )
        plt.scatter(
            action_x_list,
            action_y_list,
            c=[i for i in range(len(action_x_list))],
            cmap="Reds",
        )
        plt.show()

if __name__ == "__main__":


    if len(sys.argv) < 2:
        print("usage: python3 ./viz/visualize_dataset.py <dataset name>")
        print("optional arg --single_episode")
        print("optional arg --json")

        exit()

    dataset = {}
    if "--json" not in sys.argv:
        data_hdf5 = h5py.File(f"./datasets/{sys.argv[1]}", "r")
        for key in data_hdf5.keys():
            dataset[key] = np.array(data_hdf5[key])
    else:
        with open(sys.argv[1],'r') as json_file:
            dataset = json.load(json_file)
    print(dataset)

    visualize_recorded_rollout(dataset, single_episode = "--single_episode" in sys.argv)
