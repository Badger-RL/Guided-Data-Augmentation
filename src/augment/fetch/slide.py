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

def rotate_obj(obs, new_pos):
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

    '''
    1. choose theta
    2. origin = final position of the object
    3. We want to rotate with the origin set to the object's final position, 
       so subtract the origin from all object positions.
    4. Make rotation matrix
    5. apply the rotation matrix to all object positions in the trajectory segment.
    6. Add the origin back to all the object positions
    '''

    # translation
    traj_delta_x = origin_x - current_final_obj_x
    traj_delta_y = origin_y - current_final_obj_y

    #
    aug_obs[:, 0] += traj_delta_x
    aug_obs[:, 1] += traj_delta_y
    aug_next_obs[:, 0] += traj_delta_x
    aug_next_obs[:, 1] += traj_delta_y


    ball_at_goal_x = aug_obs[-1, 2]
    ball_at_goal_y = aug_obs[-1, 3]

    # rotation.

    # Translate origin to the final ball position
    aug_aug_obs[:, 0] -= ball_at_goal_x
    aug_aug_obs[:, 1] -= ball_at_goal_y
    aug_aug_obs[:, 2] -= ball_at_goal_x
    aug_aug_obs[:, 3] -= ball_at_goal_y
    aug_aug_next_obs[:, 0] -= ball_at_goal_x
    aug_aug_next_obs[:, 1] -= ball_at_goal_y
    aug_aug_next_obs[:, 2] -= ball_at_goal_x
    aug_aug_next_obs[:, 3] -= ball_at_goal_y

    # rotate robot and ball position about ball's final position
    theta = np.random.uniform(-180, 180) * np.pi / 180
    M = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    aug_aug_obs[:, :2] = M.dot(aug_aug_obs[:, :2].T).T
    aug_aug_obs[:, 2:4] = M.dot(aug_aug_obs[:, 2:4].T).T

    robot_angle = aug_aug_obs[:, 4] + theta
    robot_angle[robot_angle < 0] += 2 * np.pi
    aug_aug_obs[:, 4] += theta

    aug_aug_next_obs[:, :2] = M.dot(aug_aug_next_obs[:, :2].T).T
    aug_aug_next_obs[:, 2:4] = M.dot(aug_aug_next_obs[:, 2:4].T).T

    next_robot_angle = aug_aug_next_obs[:, 4] + theta
    next_robot_angle[next_robot_angle < 0] += 2 * np.pi
    aug_aug_next_obs[:, 4] += theta

    aug_aug_obs[:, 0] += ball_at_goal_x
    aug_aug_obs[:, 1] += ball_at_goal_y
    aug_aug_obs[:, 2] += ball_at_goal_x
    aug_aug_obs[:, 3] += ball_at_goal_y
    aug_aug_next_obs[:, 0] += ball_at_goal_x
    aug_aug_next_obs[:, 1] += ball_at_goal_y
    aug_aug_next_obs[:, 2] += ball_at_goal_x
    aug_aug_next_obs[:, 3] += ball_at_goal_y

    aug_reward, _ = calculate_reward(aug_aug_next_obs)

    aug_obs = convert_to_relative_obs(aug_aug_obs)
    aug_next_obs = convert_to_relative_obs(aug_aug_next_obs)

    return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done
