import copy
import numpy as np

OBSERVATION_SIZE = 28
ROBOT_POS = np.zeros(OBSERVATION_SIZE).astype(bool)
ROBOT_POS[:2+1] = True

ROBOT_VEL = np.zeros(OBSERVATION_SIZE).astype(bool)
ROBOT_VEL[20:22+1] = True

OBJECT_POS = np.zeros(OBSERVATION_SIZE).astype(bool)
OBJECT_POS[3:5+1] = True

OBJECT_RELATIVE_POS = np.zeros(OBSERVATION_SIZE).astype(bool)
OBJECT_RELATIVE_POS[6:8+1] = True

OBJECT_VEL = np.zeros(OBSERVATION_SIZE).astype(bool)
OBJECT_VEL[17:19+1] = True

GOAL = np.zeros(OBSERVATION_SIZE).astype(bool)
GOAL[-3:] = True

def is_contact(obs, next_obs):
    '''
    If the object changed position, the robot contacted it.

    :param obs:
    :param next_obs:
    :return:
    '''

    obj_displacement = next_obs[OBJECT_POS] - obs[OBJECT_POS]
    eps = 1e-8
    return np.any(np.linalg.norm(obj_displacement, axis=-1) > eps)

def translate_robot(obs, new_pos):
    obs[ROBOT_POS] = new_pos
    obs[OBJECT_RELATIVE_POS] = obs[ROBOT_POS] - obs[OBJECT_POS]

def translate_obj(obs, new_pos):
    obs[OBJECT_POS] = new_pos
    obs[OBJECT_RELATIVE_POS] = obs[ROBOT_POS] - obs[OBJECT_POS]

def translate_goal(obs, new_pos):
    obs[GOAL] = new_pos


def translate(obs, action, next_obs, reward, done, check_goal_post):
    aug_obs = copy.deepcopy(obs)
    aug_next_obs = copy.deepcopy(next_obs)
    aug_action = action.copy()
    aug_done = done.copy()

    new_obj_pos = np.random.uniform(-0.15, 0.15, size=(2,))
    translate_obj(obs, new_obj_pos)

    #
    #
    # traj_delta_x = new_ball_final_pos_x - ball_at_goal_x
    # traj_delta_y = new_ball_final_pos_y - ball_at_goal_y
    #
    #     absolute_obs[:, 0] += traj_delta_x
    #     absolute_obs[:, 1] += traj_delta_y
    #     absolute_obs[:, 2] += traj_delta_x
    #     absolute_obs[:, 3] += traj_delta_y
    #
    #     absolute_next_obs[:, 0] += traj_delta_x
    #     absolute_next_obs[:, 1] += traj_delta_y
    #     absolute_next_obs[:, 2] += traj_delta_x
    #     absolute_next_obs[:, 3] += traj_delta_y
    #
    #     ball_at_goal_x = absolute_obs[-1, 2]
    #     ball_at_goal_y = absolute_obs[-1, 3]
    #
    #     # Translate origin to the final ball position
    #     aug_absolute_obs[:, 0] -= ball_at_goal_x
    #     aug_absolute_obs[:, 1] -= ball_at_goal_y
    #     aug_absolute_obs[:, 2] -= ball_at_goal_x
    #     aug_absolute_obs[:, 3] -= ball_at_goal_y
    #     aug_absolute_next_obs[:, 0] -= ball_at_goal_x
    #     aug_absolute_next_obs[:, 1] -= ball_at_goal_y
    #     aug_absolute_next_obs[:, 2] -= ball_at_goal_x
    #     aug_absolute_next_obs[:, 3] -= ball_at_goal_y
    #
    #     # rotate robot and ball position about ball's final position
    #     theta = np.random.uniform(-180, 180) * np.pi / 180
    #     M = np.array([
    #         [np.cos(theta), -np.sin(theta)],
    #         [np.sin(theta), np.cos(theta)]
    #     ])
    #     aug_absolute_obs[:, :2] = M.dot(aug_absolute_obs[:, :2].T).T
    #     aug_absolute_obs[:, 2:4] = M.dot(aug_absolute_obs[:, 2:4].T).T
    #
    #     robot_angle = aug_absolute_obs[:, 4] + theta
    #     robot_angle[robot_angle < 0] += 2 * np.pi
    #     aug_absolute_obs[:, 4] += theta
    #
    #     aug_absolute_next_obs[:, :2] = M.dot(aug_absolute_next_obs[:, :2].T).T
    #     aug_absolute_next_obs[:, 2:4] = M.dot(aug_absolute_next_obs[:, 2:4].T).T
    #
    #     next_robot_angle = aug_absolute_next_obs[:, 4] + theta
    #     next_robot_angle[next_robot_angle < 0] += 2 * np.pi
    #     aug_absolute_next_obs[:, 4] += theta
    #
    #     aug_absolute_obs[:, 0] += ball_at_goal_x
    #     aug_absolute_obs[:, 1] += ball_at_goal_y
    #     aug_absolute_obs[:, 2] += ball_at_goal_x
    #     aug_absolute_obs[:, 3] += ball_at_goal_y
    #     aug_absolute_next_obs[:, 0] += ball_at_goal_x
    #     aug_absolute_next_obs[:, 1] += ball_at_goal_y
    #     aug_absolute_next_obs[:, 2] += ball_at_goal_x
    #     aug_absolute_next_obs[:, 3] += ball_at_goal_y
    #
    #     # Verify that translation doesn't move the agent out of bounds.
    #     # Only need to check y since we only translate vertically.
    #
    #     # check if agent and ball are in bounds
    #     if check_in_bounds(aug_absolute_obs, check_goal_post=check_goal_post) and \
    #             check_in_bounds(aug_absolute_next_obs, check_goal_post=check_goal_post):
    #         break
    #     else:
    #         aug_absolute_obs = copy.deepcopy(absolute_obs)
    #         aug_absolute_next_obs = copy.deepcopy(absolute_next_obs)
    #     # print('theta', is_valid_theta)
    # if attempts >= 100:
    #     print(f'Skipping trajectory after {attempts} augmentation attempts.')
    #     return None, None, None, None, None
    #
    # if np.random.random() < 0.5:
    #     aug_absolute_obs[:, 1] *= -1
    #     aug_absolute_next_obs[:, 1] *= -1
    #     aug_absolute_obs[:, 3] *= -1
    #     aug_absolute_next_obs[:, 3] *= -1
    #     aug_absolute_obs[:, 4] *= -1
    #     aug_absolute_next_obs[:, 4] *= -1
    #
    #     aug_action[:, 0] *= -1
    #     aug_action[:, 1] *= 1
    #     aug_action[:, 2] *= -1
    #
    # aug_reward, _ = calculate_reward(aug_absolute_next_obs)
    #
    # aug_obs = convert_to_relative_obs(aug_absolute_obs)
    # aug_next_obs = convert_to_relative_obs(aug_absolute_next_obs)
    #
    # return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done
