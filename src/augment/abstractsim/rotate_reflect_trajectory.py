import copy
import numpy as np

from src.augment.utils import convert_to_absolute_obs, calculate_reward, convert_to_relative_obs, check_valid, \
    check_in_bounds


def rotate_reflect_traj(env, obs, action, next_obs, reward, done, guided):
    aug_abs_obs = obs.copy()
    aug_abs_next_obs = next_obs.copy()
    aug_action = action.copy()
    aug_done = done.copy()

    ball_at_goal_x = obs[-1, 2].copy()
    ball_at_goal_y = obs[-1, 3].copy()

    attempts = 0
    while attempts < 100:
        attempts += 1

        if guided:
            new_ball_final_pos_x = np.random.uniform(4500, 4700)
            new_ball_final_pos_y = np.random.uniform(-500, 500)
        else:
            new_ball_final_pos_x = np.random.uniform(-4500, 4500)
            new_ball_final_pos_y = np.random.uniform(-3500, 3500)
        # new_ball_final_pos_x = ball_at_goal_x
        # new_ball_final_pos_y = ball_at_goal_y

        traj_delta_x = new_ball_final_pos_x - ball_at_goal_x
        traj_delta_y = new_ball_final_pos_y - ball_at_goal_y

        aug_abs_obs[:, 0] += traj_delta_x
        aug_abs_obs[:, 1] += traj_delta_y
        aug_abs_obs[:, 2] += traj_delta_x
        aug_abs_obs[:, 3] += traj_delta_y

        aug_abs_next_obs[:, 0] += traj_delta_x
        aug_abs_next_obs[:, 1] += traj_delta_y
        aug_abs_next_obs[:, 2] += traj_delta_x
        aug_abs_next_obs[:, 3] += traj_delta_y

        #
        # Translate origin to the final ball position
        aug_abs_obs[:, 0] -= new_ball_final_pos_x
        aug_abs_obs[:, 1] -= new_ball_final_pos_y
        aug_abs_obs[:, 2] -= new_ball_final_pos_x
        aug_abs_obs[:, 3] -= new_ball_final_pos_y
        aug_abs_next_obs[:, 0] -= new_ball_final_pos_x
        aug_abs_next_obs[:, 1] -= new_ball_final_pos_y
        aug_abs_next_obs[:, 2] -= new_ball_final_pos_x
        aug_abs_next_obs[:, 3] -= new_ball_final_pos_y

        # rotate robot and ball position about ball's final position
        theta = np.random.uniform(-180, 180) * np.pi / 180
        M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        aug_abs_obs[:, :2] = M.dot(aug_abs_obs[:, :2].T).T
        aug_abs_obs[:, 2:4] = M.dot(aug_abs_obs[:, 2:4].T).T
        aug_abs_obs[:, 4] += theta

        aug_abs_next_obs[:, :2] = M.dot(aug_abs_next_obs[:, :2].T).T
        aug_abs_next_obs[:, 2:4] = M.dot(aug_abs_next_obs[:, 2:4].T).T
        aug_abs_next_obs[:, 4] += theta


        aug_abs_obs[:, 0] += new_ball_final_pos_x
        aug_abs_obs[:, 1] += new_ball_final_pos_y
        aug_abs_obs[:, 2] += new_ball_final_pos_x
        aug_abs_obs[:, 3] += new_ball_final_pos_y
        aug_abs_next_obs[:, 0] += new_ball_final_pos_x
        aug_abs_next_obs[:, 1] += new_ball_final_pos_y
        aug_abs_next_obs[:, 2] += new_ball_final_pos_x
        aug_abs_next_obs[:, 3] += new_ball_final_pos_y

        # Verify that translation doesn't move the agent out of bounds.
        # Only need to check y since we only translate vertically.
        # break
        # check if agent and ball are in bounds
        if check_in_bounds(env, aug_abs_obs):
            break
        else:
            aug_abs_obs = copy.deepcopy(obs)
            aug_abs_next_obs = copy.deepcopy(next_obs)
        # print('theta', is_valid_theta)
    if attempts >= 100:
        print(f'Skipping trajectory after {attempts} augmentation attempts.')
        return None, None, None, None, None, None, None

    if np.random.random() < 0.5:
        aug_abs_obs[:, 1] *= -1
        aug_abs_next_obs[:, 1] *= -1
        aug_abs_obs[:, 3] *= -1
        aug_abs_next_obs[:, 3] *= -1
        aug_abs_obs[:, 4] *= -1
        aug_abs_next_obs[:, 4] *= -1

        aug_action[:, 0] *= -1
        aug_action[:, 1] *= 1
        aug_action[:, 2] *= -1

    # aug_reward, aug_done = env.calculate_reward_vec(aug_abs_next_obs, aug_abs_obs)
    n = len(obs)
    aug_reward = np.empty(n)
    aug_obs = np.empty((n, 12))
    aug_next_obs = np.empty((n, 12))
    for i in range(len(obs)):
        aug_reward[i], ball_is_at_goal, ball_is_out_of_bounds = env.calculate_reward(aug_abs_next_obs[i])
        # if done[i]:
        #     aug_done[i] = True
        # else:
        #     aug_done[i] = ball_is_out_of_bounds
        aug_done[i] = ball_is_out_of_bounds

        robot_pos = aug_abs_obs[i, :2]
        ball_pos = aug_abs_obs[i, 2:4]
        robot_angle = aug_abs_obs[i, 4]
        aug_obs[i] = env.get_obs(robot_pos, ball_pos, robot_angle)

        robot_pos = aug_abs_next_obs[i, :2]
        ball_pos = aug_abs_next_obs[i, 2:4]
        robot_angle = aug_abs_next_obs[i, 4]
        aug_next_obs[i] = env.get_obs(robot_pos, ball_pos, robot_angle)

    # if np.any(aug_abs_obs[:,0] > 5000):
    #     stop = 0
    return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done, aug_abs_obs, aug_abs_next_obs


def rotate_reflect_traj_kick(env, obs, action, next_obs, reward, done, guided):


    ball_at_goal_x = obs[-1, 2].copy()
    ball_at_goal_y = obs[-1, 3].copy()

    delta_ball_pos = next_obs[2:4] - obs[2:4]
    ball_is_kicked = (delta_ball_pos[:,0] > 0) | (delta_ball_pos[:, 1] > 0)
    first_kick_index = np.argmax(ball_is_kicked)

    aug_abs_obs = obs[first_kick_index:]
    aug_abs_next_obs = next_obs[first_kick_index:]
    aug_action = action[first_kick_index:]
    aug_done = done[first_kick_index:]

    attempts = 0
    while attempts < 100:
        attempts += 1

        xmin = np.min([aug_abs_obs[0], aug_abs_next_obs[0], aug_abs_obs[2], aug_abs_next_obs[2]])
        ymin = np.min([aug_abs_obs[1], aug_abs_next_obs[1], aug_abs_obs[3], aug_abs_next_obs[3]])
        xmax = np.max([aug_abs_obs[0], aug_abs_next_obs[0], aug_abs_obs[2], aug_abs_next_obs[2]])
        ymax = np.max([aug_abs_obs[1], aug_abs_next_obs[1], aug_abs_obs[3], aug_abs_next_obs[3]])

        # Translate bottom left corner of the righter bounding box containing the robot and ball
        new_x = np.random.uniform(-4500, 4500 - (xmax - xmin))
        new_y = np.random.uniform(-3000, 3000 - (ymax - ymin))
        delta_x = new_x - xmin
        delta_y = new_y - ymin

        aug_abs_obs[:, 0] += delta_x
        aug_abs_obs[:, 1] += delta_y
        aug_abs_obs[:, 2] += delta_x
        aug_abs_obs[:, 3] += delta_y

        aug_abs_next_obs[:, 0] += delta_x
        aug_abs_next_obs[:, 1] += delta_y
        aug_abs_next_obs[:, 2] += delta_x
        aug_abs_next_obs[:, 3] += delta_y

        if guided:
            new_ball_final_pos_x = np.random.uniform(4500, 4700)
            new_ball_final_pos_y = np.random.uniform(-500, 500)
        else:
            new_ball_final_pos_x = np.random.uniform(-4500, 4500)
            new_ball_final_pos_y = np.random.uniform(-3500, 3500)
        # new_ball_final_pos_x = ball_at_goal_x
        # new_ball_final_pos_y = ball_at_goal_y

        traj_delta_x = new_ball_final_pos_x - ball_at_goal_x
        traj_delta_y = new_ball_final_pos_y - ball_at_goal_y

        aug_abs_obs[:, 0] += traj_delta_x
        aug_abs_obs[:, 1] += traj_delta_y
        aug_abs_obs[:, 2] += traj_delta_x
        aug_abs_obs[:, 3] += traj_delta_y

        aug_abs_next_obs[:, 0] += traj_delta_x
        aug_abs_next_obs[:, 1] += traj_delta_y
        aug_abs_next_obs[:, 2] += traj_delta_x
        aug_abs_next_obs[:, 3] += traj_delta_y

        #
        # Translate origin to the final ball position
        aug_abs_obs[:, 0] -= new_ball_final_pos_x
        aug_abs_obs[:, 1] -= new_ball_final_pos_y
        aug_abs_obs[:, 2] -= new_ball_final_pos_x
        aug_abs_obs[:, 3] -= new_ball_final_pos_y
        aug_abs_next_obs[:, 0] -= new_ball_final_pos_x
        aug_abs_next_obs[:, 1] -= new_ball_final_pos_y
        aug_abs_next_obs[:, 2] -= new_ball_final_pos_x
        aug_abs_next_obs[:, 3] -= new_ball_final_pos_y

        # rotate robot and ball position about ball's final position
        theta = np.random.uniform(-180, -180) * np.pi / 180
        M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        aug_abs_obs[:, :2] = M.dot(aug_abs_obs[:, :2].T).T
        aug_abs_obs[:, 2:4] = M.dot(aug_abs_obs[:, 2:4].T).T
        aug_abs_obs[:, 4] += theta

        aug_abs_next_obs[:, :2] = M.dot(aug_abs_next_obs[:, :2].T).T
        aug_abs_next_obs[:, 2:4] = M.dot(aug_abs_next_obs[:, 2:4].T).T
        aug_abs_next_obs[:, 4] += theta


        aug_abs_obs[:, 0] += new_ball_final_pos_x
        aug_abs_obs[:, 1] += new_ball_final_pos_y
        aug_abs_obs[:, 2] += new_ball_final_pos_x
        aug_abs_obs[:, 3] += new_ball_final_pos_y
        aug_abs_next_obs[:, 0] += new_ball_final_pos_x
        aug_abs_next_obs[:, 1] += new_ball_final_pos_y
        aug_abs_next_obs[:, 2] += new_ball_final_pos_x
        aug_abs_next_obs[:, 3] += new_ball_final_pos_y

        # Verify that translation doesn't move the agent out of bounds.
        # Only need to check y since we only translate vertically.
        # break
        # check if agent and ball are in bounds
        if check_in_bounds(env, aug_abs_obs):
            break
        else:
            aug_abs_obs = copy.deepcopy(obs)
            aug_abs_next_obs = copy.deepcopy(next_obs)
        # print('theta', is_valid_theta)
    if attempts >= 100:
        print(f'Skipping trajectory after {attempts} augmentation attempts.')
        return None, None, None, None, None, None, None

    if np.random.random() < 0.5:
        aug_abs_obs[:, 1] *= -1
        aug_abs_next_obs[:, 1] *= -1
        aug_abs_obs[:, 3] *= -1
        aug_abs_next_obs[:, 3] *= -1
        aug_abs_obs[:, 4] *= -1
        aug_abs_next_obs[:, 4] *= -1

        aug_action[:, 0] *= -1
        aug_action[:, 1] *= 1
        aug_action[:, 2] *= -1

    # aug_reward, aug_done = env.calculate_reward_vec(aug_abs_next_obs, aug_abs_obs)
    n = len(obs)
    aug_reward = np.empty(n)
    aug_obs = np.empty((n, 12))
    aug_next_obs = np.empty((n, 12))
    for i in range(len(obs)):
        aug_reward[i], ball_is_at_goal, ball_is_out_of_bounds = env.calculate_reward(aug_abs_next_obs[i])
        # if done[i] and ball_is_at_goal:
        #     aug_done[i] = True
        # else:
        #     aug_done[i] = ball_is_out_of_bounds
        # aug_done[i] = ball_is_out_of_bounds or ball_is_at_goal
        aug_done[i] = ball_is_out_of_bounds or done[i]

        robot_pos = aug_abs_obs[i, :2]
        ball_pos = aug_abs_obs[i, 2:4]
        robot_angle = aug_abs_obs[i, 4]
        aug_obs[i] = env.get_obs(robot_pos, ball_pos, robot_angle)

        robot_pos = aug_abs_next_obs[i, :2]
        ball_pos = aug_abs_next_obs[i, 2:4]
        robot_angle = aug_abs_next_obs[i, 4]
        aug_next_obs[i] = env.get_obs(robot_pos, ball_pos, robot_angle)

    return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done, aug_abs_obs, aug_abs_next_obs