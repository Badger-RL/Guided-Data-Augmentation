import copy
import numpy as np

from augment.utils import convert_to_absolute_obs, calculate_reward, convert_to_relative_obs, check_valid


def rotate_reflect_traj(obs, action, next_obs, reward, done):

    absolute_obs = convert_to_absolute_obs(obs)
    absolute_next_obs = convert_to_absolute_obs(next_obs)

    aug_absolute_obs = copy.deepcopy(absolute_obs)
    aug_absolute_next_obs = copy.deepcopy(absolute_next_obs)
    aug_action = action.copy()
    aug_done = done.copy()

    ball_at_goal_x = absolute_obs[-1, 2]
    ball_at_goal_y = absolute_obs[-1, 3]

    theta = np.random.uniform(-45, 45)*np.pi/180
    # theta = -20*np.pi/180
    #
    # Translate origin to the final ball position
    aug_absolute_obs[:, 0] -= ball_at_goal_x
    aug_absolute_obs[:, 1] -= ball_at_goal_y
    aug_absolute_obs[:, 2] -= ball_at_goal_x
    aug_absolute_obs[:, 3] -= ball_at_goal_y
    aug_absolute_next_obs[:, 0] -= ball_at_goal_x
    aug_absolute_next_obs[:, 1] -= ball_at_goal_y
    aug_absolute_next_obs[:, 2] -= ball_at_goal_x
    aug_absolute_next_obs[:, 3] -= ball_at_goal_y
    M = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    aug_absolute_obs[:, :2] = M.dot(aug_absolute_obs[:, :2].T).T
    aug_absolute_obs[:, 2:4] = M.dot(aug_absolute_obs[:, 2:4].T).T

    robot_angle = aug_absolute_obs[:, 4] + theta
    robot_angle[robot_angle < 0] += 2*np.pi
    aug_absolute_obs[:, 4] += theta

    aug_absolute_next_obs[:, :2] = M.dot(aug_absolute_next_obs[:, :2].T).T
    aug_absolute_next_obs[:, 2:4] = M.dot(aug_absolute_next_obs[:, 2:4].T).T

    next_robot_angle = aug_absolute_next_obs[:, 4] + theta
    next_robot_angle[next_robot_angle < 0] += 2*np.pi
    aug_absolute_next_obs[:, 4] += theta

    aug_absolute_obs[:, 0] += ball_at_goal_x
    aug_absolute_obs[:, 1] += ball_at_goal_y
    aug_absolute_obs[:, 2] += ball_at_goal_x
    aug_absolute_obs[:, 3] += ball_at_goal_y
    aug_absolute_next_obs[:, 0] += ball_at_goal_x
    aug_absolute_next_obs[:, 1] += ball_at_goal_y
    aug_absolute_next_obs[:, 2] += ball_at_goal_x
    aug_absolute_next_obs[:, 3] += ball_at_goal_y

    if np.random.random() < 0.5:
        aug_absolute_obs[:, 1] *= -1
        aug_absolute_next_obs[:, 1] *= -1
        aug_absolute_obs[:, 3] *= -1
        aug_absolute_next_obs[:, 3] *= -1
        aug_absolute_obs[:, 4] *= -1
        aug_absolute_next_obs[:, 4] *= -1

        aug_action[:, 0] *= -1
        aug_action[:, 1] *= 1
        aug_action[:, 2] *= -1


    aug_reward, _ = calculate_reward(aug_absolute_next_obs)

    aug_obs = convert_to_relative_obs(aug_absolute_obs)
    aug_next_obs = convert_to_relative_obs(aug_absolute_next_obs)

    return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done
