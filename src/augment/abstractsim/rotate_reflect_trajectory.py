import copy
import numpy as np

from src.augment.utils import convert_to_absolute_obs, calculate_reward, convert_to_relative_obs, check_valid, \
    check_in_bounds


def rotate_reflect_traj(obs, action, next_obs, reward, done, check_goal_post, guided=False, neg=False):
    absolute_obs = convert_to_absolute_obs(obs)
    absolute_next_obs = convert_to_absolute_obs(next_obs)

    aug_absolute_obs = copy.deepcopy(absolute_obs)
    aug_absolute_next_obs = copy.deepcopy(absolute_next_obs)
    aug_action = action.copy()
    aug_done = done.copy()

    attempts = 0
    while attempts < 10:
        attempts += 1

        ball_at_goal_x = absolute_obs[-1, 2]
        ball_at_goal_y = absolute_obs[-1, 3]
        new_ball_final_pos_x = np.random.uniform(4501,4501)
        new_ball_final_pos_y = np.random.uniform(-750, 750)
        if neg:
            new_ball_final_pos_y = np.random.uniform(-850, 850)

        traj_delta_x = new_ball_final_pos_x - ball_at_goal_x
        traj_delta_y = new_ball_final_pos_y - ball_at_goal_y

        aug_absolute_obs[:, 0] += traj_delta_x
        aug_absolute_obs[:, 1] += traj_delta_y
        aug_absolute_obs[:, 2] += traj_delta_x
        aug_absolute_obs[:, 3] += traj_delta_y

        absolute_next_obs[:, 0] += traj_delta_x
        absolute_next_obs[:, 1] += traj_delta_y
        absolute_next_obs[:, 2] += traj_delta_x
        absolute_next_obs[:, 3] += traj_delta_y

        ball_at_goal_x = aug_absolute_obs[-1, 2]
        ball_at_goal_y = aug_absolute_obs[-1, 3]

        # Translate origin to the final ball position
        aug_absolute_obs[:, 0] -= ball_at_goal_x
        aug_absolute_obs[:, 1] -= ball_at_goal_y
        aug_absolute_obs[:, 2] -= ball_at_goal_x
        aug_absolute_obs[:, 3] -= ball_at_goal_y
        aug_absolute_next_obs[:, 0] -= ball_at_goal_x
        aug_absolute_next_obs[:, 1] -= ball_at_goal_y
        aug_absolute_next_obs[:, 2] -= ball_at_goal_x
        aug_absolute_next_obs[:, 3] -= ball_at_goal_y

        # rotate robot and ball position about ball's final position
        theta = np.random.uniform(-180, 180) * np.pi / 180
        M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        aug_absolute_obs[:, :2] = M.dot(aug_absolute_obs[:, :2].T).T
        aug_absolute_obs[:, 2:4] = M.dot(aug_absolute_obs[:, 2:4].T).T

        robot_angle = aug_absolute_obs[:, 4] + theta
        robot_angle[robot_angle < 0] += 2 * np.pi
        aug_absolute_obs[:, 4] += theta

        aug_absolute_next_obs[:, :2] = M.dot(aug_absolute_next_obs[:, :2].T).T
        aug_absolute_next_obs[:, 2:4] = M.dot(aug_absolute_next_obs[:, 2:4].T).T

        next_robot_angle = aug_absolute_next_obs[:, 4] + theta
        next_robot_angle[next_robot_angle < 0] += 2 * np.pi
        aug_absolute_next_obs[:, 4] += theta

        aug_absolute_obs[:, 0] += ball_at_goal_x
        aug_absolute_obs[:, 1] += ball_at_goal_y
        aug_absolute_obs[:, 2] += ball_at_goal_x
        aug_absolute_obs[:, 3] += ball_at_goal_y
        aug_absolute_next_obs[:, 0] += ball_at_goal_x
        aug_absolute_next_obs[:, 1] += ball_at_goal_y
        aug_absolute_next_obs[:, 2] += ball_at_goal_x
        aug_absolute_next_obs[:, 3] += ball_at_goal_y

        # Verify that translation doesn't move the agent out of bounds.
        # Only need to check y since we only translate vertically.

        # check if agent and ball are in bounds
        if check_in_bounds(aug_absolute_obs, check_goal_post=check_goal_post) and \
                check_in_bounds(aug_absolute_next_obs, check_goal_post=check_goal_post):
            break
        else:
            aug_absolute_obs = copy.deepcopy(absolute_obs)
            aug_absolute_next_obs = copy.deepcopy(absolute_next_obs)
        # print('theta', is_valid_theta)
    if attempts >= 10:
        print(f'Skipping trajectory after {attempts} augmentation attempts.')
        return None, None, None, None, None

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


def random_traj(obs, action, next_obs, reward, done, check_goal_post, guided=False):
    absolute_obs = convert_to_absolute_obs(obs)
    absolute_next_obs = convert_to_absolute_obs(next_obs)

    aug_absolute_obs = copy.deepcopy(absolute_obs)
    aug_absolute_next_obs = copy.deepcopy(absolute_next_obs)
    aug_action = action.copy()
    aug_done = done.copy()

    at_goal = (aug_absolute_next_obs[:, 2] > 4492) & (np.abs(aug_absolute_next_obs[:, 3]) < 750)


    attempts = 0
    while attempts < 10:
        attempts += 1

        # rotate robot and ball position about ball's final position
        theta = np.random.uniform(-180, 180) * np.pi / 180
        M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        aug_absolute_obs[:, :2] = M.dot(aug_absolute_obs[:, :2].T).T
        aug_absolute_obs[:, 2:4] = M.dot(aug_absolute_obs[:, 2:4].T).T

        robot_angle = aug_absolute_obs[:, 4] + theta
        robot_angle[robot_angle < 0] += 2 * np.pi
        aug_absolute_obs[:, 4] += theta

        aug_absolute_next_obs[:, :2] = M.dot(aug_absolute_next_obs[:, :2].T).T
        aug_absolute_next_obs[:, 2:4] = M.dot(aug_absolute_next_obs[:, 2:4].T).T

        next_robot_angle = aug_absolute_next_obs[:, 4] + theta
        next_robot_angle[next_robot_angle < 0] += 2 * np.pi
        aug_absolute_next_obs[:, 4] += theta

        xmin = np.min([aug_absolute_obs[0], aug_absolute_next_obs[0], aug_absolute_obs[2], aug_absolute_next_obs[2]])
        ymin = np.min([aug_absolute_obs[1], aug_absolute_next_obs[1], aug_absolute_obs[3], aug_absolute_next_obs[3]])
        xmax = np.max([aug_absolute_obs[0], aug_absolute_next_obs[0], aug_absolute_obs[2], aug_absolute_next_obs[2]])
        ymax = np.max([aug_absolute_obs[1], aug_absolute_next_obs[1], aug_absolute_obs[3], aug_absolute_next_obs[3]])

        # Translate bottom left corner of the righter bounding box containing the robot and ball
        new_x = np.random.uniform(-4500, 4500 - (xmax - xmin))
        new_y = np.random.uniform(-3000, 3000 - (ymax - ymin))
        if np.abs(new_y) < 750 and np.random.random() < 1/30:
            new_x = 4501

        delta_x = new_x - xmin
        delta_y = new_y - ymin

        aug_absolute_obs[:, 0] += delta_x
        aug_absolute_obs[:, 1] += delta_y
        aug_absolute_obs[:, 2] += delta_x
        aug_absolute_obs[:, 3] += delta_y

        aug_absolute_next_obs[:, 0] += delta_x
        aug_absolute_next_obs[:, 1] += delta_y
        aug_absolute_next_obs[:, 2] += delta_x
        aug_absolute_next_obs[:, 3] += delta_y

        # Verify that translation doesn't move the agent out of bounds.
        # Only need to check y since we only translate vertically.

        # check if agent and ball are in bounds
        if check_in_bounds(aug_absolute_obs, check_goal_post=check_goal_post) and \
                check_in_bounds(aug_absolute_next_obs, check_goal_post=check_goal_post):
            break
        else:
            aug_absolute_obs = copy.deepcopy(absolute_obs)
            aug_absolute_next_obs = copy.deepcopy(absolute_next_obs)
        # print('theta', is_valid_theta)
    if attempts >= 100:
        print(f'Skipping trajectory after {attempts} augmentation attempts.')
        return None, None, None, None, None

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
    aug_at_goal = (aug_absolute_next_obs[:, 2] > 4492) & (np.abs(aug_absolute_next_obs[:, 3]) < 750)
    if at_goal.sum() > 0 and aug_at_goal.sum() < 1:
        print('here')
        aug_absolute_obs = aug_absolute_obs[~at_goal]
        aug_absolute_next_obs = aug_absolute_next_obs[~at_goal]
        aug_action = aug_action[~at_goal]
        aug_reward = aug_reward[~at_goal]
        aug_done = aug_done[~at_goal]




    aug_obs = convert_to_relative_obs(aug_absolute_obs)
    aug_next_obs = convert_to_relative_obs(aug_absolute_next_obs)

    return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done

