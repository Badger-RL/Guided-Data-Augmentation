import copy
import time

import h5py
import numpy as np

import gymnasium as gym
from matplotlib import pyplot as plt

from generate.utils import load_dataset, reset_data, append_data, npify, extend_data

OBSERVATION_SIZE = 28
ROBOT_POS = np.zeros(OBSERVATION_SIZE).astype(bool)
ROBOT_POS[:2+1] = True

ROBOT_XY = np.zeros(OBSERVATION_SIZE).astype(bool)
ROBOT_XY[:2] = True

ROBOT_VEL = np.zeros(OBSERVATION_SIZE).astype(bool)
ROBOT_VEL[20:22+1] = True

OBJECT_POS = np.zeros(OBSERVATION_SIZE).astype(bool)
OBJECT_POS[3:5+1] = True

OBJECT_RELATIVE_POS = np.zeros(OBSERVATION_SIZE).astype(bool)
OBJECT_RELATIVE_POS[6:8+1] = True

OBJECT_RELATIVE_VEL = np.zeros(OBSERVATION_SIZE).astype(bool)
OBJECT_RELATIVE_VEL[14:16+1] = True

OBJECT_VEL = np.zeros(OBSERVATION_SIZE).astype(bool)
OBJECT_VEL[17:19+1] = True

GOAL = np.zeros(OBSERVATION_SIZE).astype(bool)
GOAL[-3:] = True

OBJ_HEIGHT_OFFSET = 0.424702091
INITIAL_GRIPPER_POS = np.array([1.34820577, 0.74894884, 0.41361829])

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

def translate_robot(obs, delta):
    obs[:, ROBOT_POS] += delta
    obs[:, OBJECT_RELATIVE_POS] = obs[:, ROBOT_POS] - obs[:, OBJECT_POS]

def translate_obj(obs, delta):
    obs[:, OBJECT_POS] += delta
    obs[:, OBJECT_RELATIVE_POS] = obs[:, ROBOT_POS] - obs[:, OBJECT_POS]

def rotate_obj(obs, new_pos):
    pass
    # obs[OBJECT_POS] = new_pos
    # obs[OBJECT_RELATIVE_POS] = obs[ROBOT_POS] - obs[OBJECT_POS]

def translate_goal(obs, new_pos):
    obs[:, GOAL] = new_pos


def translate(obs, action, next_obs, reward, done):
    aug_obs = copy.deepcopy(obs)
    aug_next_obs = copy.deepcopy(next_obs)
    aug_action = action.copy()
    aug_reward = reward.copy()
    aug_done = done.copy()

    # translate object
    # delta_obj = next_obs[OBJECT_POS] - obs[OBJECT_POS]
    new_obj_pos = np.random.uniform(-0.15, 0.15, size=(3,)) + INITIAL_GRIPPER_POS[:3]
    new_obj_pos[-1] = OBJ_HEIGHT_OFFSET # keep z coordinate
    delta_obj = new_obj_pos - obs[-1, OBJECT_POS]
    aug_obs[:, OBJECT_POS] += delta_obj
    aug_next_obs[:, OBJECT_POS] += delta_obj

    # translate_obj(aug_obs, delta_obj)
    # translate_obj(aug_next_obs, delta_obj)

    # translate robot
    new_robot_pos = np.random.uniform(-0.15, 0.15, size=(2,)) + INITIAL_GRIPPER_POS[:2]
    # new_robot_pos[-1] = obs[:, ROBOT_XY] # keep z coordinate
    # delta_robot = next_obs[ROBOT_POS] - obs[ROBOT_POS]
    delta_robot = new_robot_pos - obs[-1, ROBOT_XY].copy()
    aug_obs[:, ROBOT_XY] += delta_robot
    aug_next_obs[:, ROBOT_XY] += delta_robot
    # translate_robot(aug_obs, delta_robot)
    # translate_robot(aug_next_obs, delta_robot)

    # zero out relative features to make aug simpler
    aug_obs[:, OBJECT_RELATIVE_POS] = aug_obs[:, ROBOT_POS] - aug_obs[:, OBJECT_POS]
    aug_obs[:, OBJECT_RELATIVE_VEL] = aug_obs[:, ROBOT_VEL] - aug_obs[:, OBJECT_VEL]
    aug_next_obs[:, OBJECT_RELATIVE_POS] = aug_next_obs[:, ROBOT_POS] - aug_next_obs[:, OBJECT_POS]
    aug_next_obs[:, OBJECT_RELATIVE_VEL] = aug_next_obs[:, ROBOT_VEL] - aug_next_obs[:, OBJECT_VEL]
    # aug_obs[:, OBJECT_RELATIVE_POS] = 0
    # aug_obs[:, OBJECT_RELATIVE_VEL] = 0
    # aug_next_obs[:, OBJECT_RELATIVE_POS] = 0
    # aug_next_obs[:, OBJECT_RELATIVE_VEL] = 0

    # set goal equal to final object pos
    aug_goal = aug_next_obs[-1, OBJECT_POS]
    aug_obs[:, GOAL] = aug_goal
    aug_next_obs[:, GOAL] = aug_goal

    is_at_goal = np.linalg.norm(aug_next_obs[:, OBJECT_POS] - aug_next_obs[:, GOAL], axis=-1) < 0.05
    aug_reward[:] = -1
    aug_reward[is_at_goal] = 0
    aug_done[:] = False

    return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done



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
    # traj_delta_x = origin_x - current_final_obj_x
    # traj_delta_y = origin_y - current_final_obj_y
    #
    # #
    # aug_obs[:, 0] += traj_delta_x
    # aug_obs[:, 1] += traj_delta_y
    # aug_next_obs[:, 0] += traj_delta_x
    # aug_next_obs[:, 1] += traj_delta_y
    #
    #
    # ball_at_goal_x = aug_obs[-1, 2]
    # ball_at_goal_y = aug_obs[-1, 3]
    #
    # # rotation.
    #
    # # Translate origin to the final ball position
    # aug_aug_obs[:, 0] -= ball_at_goal_x
    # aug_aug_obs[:, 1] -= ball_at_goal_y
    # aug_aug_obs[:, 2] -= ball_at_goal_x
    # aug_aug_obs[:, 3] -= ball_at_goal_y
    # aug_aug_next_obs[:, 0] -= ball_at_goal_x
    # aug_aug_next_obs[:, 1] -= ball_at_goal_y
    # aug_aug_next_obs[:, 2] -= ball_at_goal_x
    # aug_aug_next_obs[:, 3] -= ball_at_goal_y
    #
    # # rotate robot and ball position about ball's final position
    # theta = np.random.uniform(-180, 180) * np.pi / 180
    # M = np.array([
    #     [np.cos(theta), -np.sin(theta)],
    #     [np.sin(theta), np.cos(theta)]
    # ])
    # aug_aug_obs[:, :2] = M.dot(aug_aug_obs[:, :2].T).T
    # aug_aug_obs[:, 2:4] = M.dot(aug_aug_obs[:, 2:4].T).T
    #
    # robot_angle = aug_aug_obs[:, 4] + theta
    # robot_angle[robot_angle < 0] += 2 * np.pi
    # aug_aug_obs[:, 4] += theta
    #
    # aug_aug_next_obs[:, :2] = M.dot(aug_aug_next_obs[:, :2].T).T
    # aug_aug_next_obs[:, 2:4] = M.dot(aug_aug_next_obs[:, 2:4].T).T
    #
    # next_robot_angle = aug_aug_next_obs[:, 4] + theta
    # next_robot_angle[next_robot_angle < 0] += 2 * np.pi
    # aug_aug_next_obs[:, 4] += theta
    #
    # aug_aug_obs[:, 0] += ball_at_goal_x
    # aug_aug_obs[:, 1] += ball_at_goal_y
    # aug_aug_obs[:, 2] += ball_at_goal_x
    # aug_aug_obs[:, 3] += ball_at_goal_y
    # aug_aug_next_obs[:, 0] += ball_at_goal_x
    # aug_aug_next_obs[:, 1] += ball_at_goal_y
    # aug_aug_next_obs[:, 2] += ball_at_goal_x
    # aug_aug_next_obs[:, 3] += ball_at_goal_y
    #
    # aug_reward, _ = calculate_reward(aug_aug_next_obs)
    #
    # aug_obs = convert_to_relative_obs(aug_aug_obs)
    # aug_next_obs = convert_to_relative_obs(aug_aug_next_obs)

def flatten_obs(obs):
    return np.concatenate([obs['observation'], obs['desired_goal']])

def gen_aug_dataset():
    dataset_path = f"../../datasets/FetchPush-v2/no_aug.hdf5"
    observed_dataset = load_dataset(dataset_path)
    n = len(observed_dataset['observations'])

    observations = observed_dataset['observations']
    actions = observed_dataset['actions']
    next_observations = observed_dataset['next_observations']
    rewards = observed_dataset['rewards']
    terminals = observed_dataset['terminals']

    obj_pos = observations[:, OBJECT_POS]
    plt.scatter(obj_pos[:, 0], obj_pos[:, 1])
    plt.xlim(1, 2)
    plt.ylim(0, 1)
    plt.show()

    aug_dataset = reset_data()

    max_aug = 1e6
    aug_count = 0

    max_aug = 100000
    while aug_count < max_aug:
        start = 0
        for i in range(n):
            if observed_dataset['terminals'][i]:
                end = (i + 1)
                s, a, r, ns, done = translate(
                    observations[start:end],
                    actions[start:end],
                    next_observations[start:end],
                    rewards[start:end],
                    terminals[start:end])
                if s is None:
                    continue
                extend_data(aug_dataset, s, a, r, ns, done )
                aug_count += len(s)
                start = end

    npify(aug_dataset)
    observations = aug_dataset['observations']
    # actions = observed_dataset['actions']
    # next_observations = observed_dataset['next_observations']
    # rewards = observed_dataset['rewards']
    # terminals = observed_dataset['terminals']

    obj_pos = observations[:, OBJECT_POS]
    goal_pos = observations[:, GOAL]

    plt.scatter(obj_pos[:, 0], obj_pos[:, 1])
    plt.scatter(goal_pos[:, -3], goal_pos[:, -2])

    plt.xlim(1, 2)
    plt.ylim(0, 1)

    plt.show()

    dataset = h5py.File("../../datasets/FetchPush-v2/random.hdf5", 'w')
    for k,v in aug_dataset.items():
        aug_dataset[k] = np.concatenate([observed_dataset[k], v])
        dataset.create_dataset(k, data=np.array(aug_dataset[k]), compression='gzip')


if __name__ == '__main__':

    gen_aug_dataset()

    env_id = 'FetchPush-v2'
    env = gym.make(env_id, render_mode='human')

    obs, _ = env.reset()
    action = env.action_space.sample()
    action = np.zeros_like(action)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    obs = flatten_obs(obs)
    next_obs = flatten_obs(next_obs)

    # env.render()
    # time.sleep(1)

    aug_obs, aug_action, aug_next_obs, aug_reward, aug_done = translate(obs, action, next_obs, reward, done)

    sim = env.unwrapped.sim
    env.sim.set_state(aug_obs, aug_obs)
    env.render()

    time.sleep(20)