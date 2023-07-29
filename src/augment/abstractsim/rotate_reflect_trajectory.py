import copy
import numpy as np

from src.augment.utils import convert_to_absolute_obs, calculate_reward, convert_to_relative_obs, check_valid, \
    check_in_bounds


def rotate_reflect_traj(env, obs, action, next_obs, reward, done, check_goal_post):
    aug_abs_obs = obs.copy()
    aug_abs_next_obs = next_obs.copy()
    aug_action = action.copy()
    aug_done = done.copy()

    ball_at_goal_x = obs[-1, 2].copy()
    ball_at_goal_y = obs[-1, 3].copy()

    attempts = 0
    while attempts < 100:
        attempts += 1

        new_ball_final_pos_x = 4800
        new_ball_final_pos_y = np.random.uniform(-500, 500)
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
        if check_in_bounds(aug_abs_obs, check_goal_post=check_goal_post):
            break
        else:
            aug_abs_obs = copy.deepcopy(obs)
            aug_abs_next_obs = copy.deepcopy(next_obs)
        # print('theta', is_valid_theta)
    if attempts >= 100:
        print(f'Skipping trajectory after {attempts} augmentation attempts.')
        return None, None, None, None, None, None, None

    if np.random.random() < 1:
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
        aug_reward[i], is_goal, is_out_of_bounds = env.calculate_reward(aug_abs_next_obs[i])
        aug_done[i] = is_goal or is_out_of_bounds

        robot_pos = aug_abs_obs[i, :2]
        ball_pos = aug_abs_obs[i, 2:4]
        robot_angle = aug_abs_obs[i, 4]
        aug_obs[i] = env.get_obs(robot_pos, ball_pos, robot_angle)

        robot_pos = aug_abs_next_obs[i, :2]
        ball_pos = aug_abs_next_obs[i, 2:4]
        robot_angle = aug_abs_next_obs[i, 4]
        aug_next_obs[i] = env.get_obs(robot_pos, ball_pos, robot_angle)

    return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done, aug_abs_obs, aug_abs_next_obs
