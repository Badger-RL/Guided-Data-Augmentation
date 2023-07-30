import math

import numpy as np


def is_at_goal(target_x, target_y):
    mask = (target_x > 4400 ) & (target_y < 500) & (target_y > -500)
    return mask

def is_in_bounds(target_x, target_y):
    mask = (np.abs(target_x) < 4500 ) & (np.abs(target_y) < 3500)
    return mask

def get_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2), axis=-1)

def looking_at_ball(robot, ball, angle):
    # Convert from radians to degrees
    robot_angle = (angle*180/np.pi) % 360

    # Find the angle between the robot and the ball
    angle_to_ball = (
        np.arctan2(ball[:, 1] - robot[:, 1], ball[:, 0] - robot[:, 0])
    )*180/np.pi
    # Check if the robot is facing the ball
    req_angle = 10
    angle = (robot_angle - angle_to_ball) % 360

    return (angle < req_angle) | (angle > 360 - req_angle)

def calculate_reward(obs, next_obs, ):

    rewards = np.zeros(len(obs))

    robot = obs[:, :2]
    next_robot = next_obs[:, :2]
    ball = obs[:, 2:4]
    next_ball = next_obs[:, 2:4]

    # Goal - Team
    mask = is_at_goal(next_ball[:, 0], next_ball[:, 1])
    rewards[mask] += 1

    # # Ball to goal - Team
    cur_ball_distance = get_distance(next_ball, [4800, 0])
    prev_ball_distance = get_distance(ball, [4800, 0])
    rewards += 1/40 * (prev_ball_distance - cur_ball_distance)

    # reward for stepping towards ball
    cur_distance = get_distance(next_robot, next_ball)
    prev_distance = get_distance(robot, next_ball)
    rewards += 1/10 * (prev_distance - cur_distance)

    mask = looking_at_ball(robot, ball, obs[:,-1])
    rewards[mask] += 1/100

    return rewards


def convert_to_absolute_obs(obs):

    x_scale = 9000
    y_scale = 6000
    goal_x = 4800
    goal_y = 0

    target_x = (goal_x - obs[:, 2]*x_scale)
    target_y = (goal_y - obs[:, 3]*y_scale)
    robot_x = (target_x - obs[:, 0]*x_scale)
    robot_y = (target_y - obs[:, 1]*y_scale)

    relative_x = target_x - robot_x
    relative_y = target_y - robot_y
    relative_angle = np.arctan2(relative_y, relative_x)
    relative_angle[relative_angle < 0] += 2*np.pi

    relative_angle_minus_robot_angle = np.arctan2(obs[:, 4], obs[:, 5])
    relative_angle_minus_robot_angle[relative_angle_minus_robot_angle < 0] += 2*np.pi

    robot_angle = relative_angle - relative_angle_minus_robot_angle
    robot_angle[robot_angle < 0] += 2*np.pi

    return np.array([
        robot_x,
        robot_y,
        target_x,
        target_y,
        robot_angle
    ]).T

def get_relative_observation(agent_loc, object_loc):
    # Get relative position of object to agent, returns x, y, angle
    # Agent loc is x, y, angle
    # Object loc is x, y

    # Get relative position of object to agent
    if len(object_loc.shape) > 1:
        x = object_loc[:, 0] - agent_loc[:, 0]
        y = object_loc[:, 1] - agent_loc[:, 1]
    else:
        x = object_loc[0] - agent_loc[:, 0]
        y = object_loc[1] - agent_loc[:, 1]

    angle = np.arctan2(y, x) - agent_loc[:, 2]

    # Rotate x, y by -agent angle
    xprime = x * np.cos(-agent_loc[:, 2]) - y * np.sin(-agent_loc[:, 2])
    yprime = x * np.sin(-agent_loc[:, 2]) + y * np.cos(-agent_loc[:, 2])

    return [xprime / 10000, yprime / 10000, np.sin(angle), np.cos(angle)]

def convert_to_relative_obs(obs):
    # Get origin position
    relative_obs = []
    agent_loc = np.concatenate([obs[:, :2], obs[:, -1].reshape(-1,1)], axis=-1)
    ball_loc = obs[:, 2:4]
    origin_obs = get_relative_observation(agent_loc, np.array([0, 0]))
    relative_obs.extend(origin_obs)

    # Get goal position
    goal_obs = get_relative_observation(agent_loc, np.array([4800, 0]))
    relative_obs.extend(goal_obs)

    # Get ball position
    ball_obs = get_relative_observation(agent_loc, ball_loc)
    relative_obs.extend(ball_obs)

    return np.array(relative_obs, dtype=np.float32).T

def check_in_bounds(env, absolute_obs):
    is_in_bounds = False
    # print(np.max(np.abs(absolute_obs[:, 0])))
    # print(np.max(np.abs(absolute_obs[:, 1])))

    ball_is_at_goal = (absolute_obs[:,2] > 4400) & (np.abs(absolute_obs[:, 3]) < 500)
    absolute_obs = absolute_obs[~ball_is_at_goal]
    abs_robot_x = np.abs(absolute_obs[:, 0])
    abs_robot_y = np.abs(absolute_obs[:, 1])
    abs_ball_x = np.abs(absolute_obs[:, 2])
    abs_ball_y = np.abs(absolute_obs[:, 3])

    # check robot in bounds
    if np.all(abs_robot_x < 4400):
        if np.all(abs_robot_y < 3500):
            if np.all(abs_ball_x < 4500):
                if np.all(abs_ball_y < 3500):
                    is_in_bounds = True
    # only check y positions, since there's sizeable uncertainty in x localization
    # if  np.all(np.abs(absolute_obs[:, 1]) < 3000) and np.all(np.abs(absolute_obs[:, 3]) < 3000):
    #         is_in_bounds = True
    return is_in_bounds

def check_valid(env, aug_abs_obs, aug_action, aug_reward, aug_abs_next_obs, aug_done, render=False, verbose=False):

    np.set_printoptions(linewidth=np.inf)

    valid = True
    for i in range(len(aug_abs_obs)):
        env.reset()
        aug_abs_obs_i = aug_abs_obs[i]
        env.set_state(robot_pos=aug_abs_obs_i[:2],
                      robot_angle=aug_abs_obs_i[4],
                      ball_pos=aug_abs_obs_i[2:4])

        next_obs, reward, done, info = env.step(aug_action[i])
        next_obs = info['absolute_next_obs']

        if render:
            env.render()
        if i == 10:
            stop = 90
        # Augmented transitions at the goal are surely not valid, but that's fine.
        if not info['is_success']:
            if not np.allclose(next_obs, aug_abs_next_obs[i]):
                valid = False
                if verbose:
                    # print(f'{i}, true next obs - aug next obs\t', aug_abs_next_obs[i]-next_obs)
                    print(f'{i}, true next obs\t', next_obs)
                    print(f'{i}, aug next obs \t', aug_abs_next_obs[i])
                    aug_delta_ball = aug_abs_next_obs[i,2:4] - aug_abs_obs_i[2:4]
                    true_delta_ball = next_obs[2:4] - aug_abs_obs_i[2:4]
                    print(f'{i}, true delta ball\t', true_delta_ball)
                    print(f'{i}, aug delta ball \t', aug_delta_ball)
                    # print(np.linalg.norm(aug_abs_next_obs[i]-next_obs))
                    #
                    # print(aug_next_obs[i, 2:4], next_obs[2:4])

            # if not np.isclose(reward, aug_reward[i]):
            #     valid = False
            #     if verbose:
            #         print(f'{i}, aug reward: {aug_reward[i]}\ttrue reward: {reward}')

            if done != aug_done[i]:
                valid = False
                if verbose:
                    print(f'{i}, aug reward: {aug_reward[i]}\ttrue reward: {reward}')

        # if not valid:
        #     break

    return valid