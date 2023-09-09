import copy
import time

import h5py
import numpy as np

import gymnasium as gym
from matplotlib import pyplot as plt

from src.generate.utils import load_dataset, reset_data, append_data, npify, extend_data

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
    obs[:, GOAL] = new_pos

def translate_obj(obs, delta):
    obs[:, OBJECT_POS] += delta
    obs[:, OBJECT_RELATIVE_POS] = obs[:, ROBOT_POS] - obs[:, OBJECT_POS]

def sample_rotation_matrix():
    thetas = np.array([0, np.pi/2, np.pi, np.pi*3/2])
    rotation_matrices = []
    for theta in thetas:
        M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotation_matrices.append(M)
    idx = np.random.randint(len(rotation_matrices))
    M = rotation_matrices[idx]
    return M
def generate_aug_trajectory(trajectory):

    n = len(trajectory['observations'])

    observations = trajectory['observations']
    next_observations = trajectory['next_observations']
    actions = trajectory['actions']
    rewards = trajectory['rewards']
    terminals = trajectory['terminals']

    robot_pos = observations[:, ROBOT_POS]
    obj_pos = observations[:, OBJECT_POS]
    goal_pos = observations[:, GOAL]

    plt.scatter(robot_pos[:, 0], robot_pos[:, 1], label='robot')
    plt.scatter(obj_pos[:, 0], obj_pos[:, 1], label='obj')
    plt.scatter(goal_pos[:, -3], goal_pos[:, -2], label='goal')



    

    aug_observations = observations.copy()
    aug_next_observations = next_observations.copy()
    aug_actions = actions.copy()
    aug_rewards = rewards.copy()
    aug_terminals = terminals.copy()

    new_obj_pos = np.random.uniform(-0.1, 0.1, size=(3,)) + INITIAL_GRIPPER_POS[:3]
    new_obj_pos[-1] = OBJ_HEIGHT_OFFSET
    pos_delta_obj = new_obj_pos - observations[-1, OBJECT_POS]


    aug_observations[:, OBJECT_POS] += pos_delta_obj
    aug_next_observations[:, OBJECT_POS] += pos_delta_obj

    # M = sample_rotation_matrix()
    # pos_delta_obj[:2] = M.dot(pos_delta_obj[:2]).T
    # aug_observations[:, OBJECT_VEL][:, :2] = M.dot(aug_observations[:, OBJECT_VEL][:, :2].T).T
    # aug_next_observations[:, OBJECT_VEL][:, :2] = M.dot(aug_next_observations[:, OBJECT_VEL][:, :2].T).T


    aug_observations[:, OBJECT_RELATIVE_POS] = aug_observations[:, OBJECT_POS] - aug_observations[:, ROBOT_POS]
    aug_observations[:, OBJECT_RELATIVE_VEL] = aug_observations[:, OBJECT_VEL] - aug_observations[:, ROBOT_VEL]
    aug_next_observations[:, OBJECT_RELATIVE_POS] = aug_next_observations[:, OBJECT_POS] - aug_next_observations[:, ROBOT_POS]
    aug_next_observations[:, OBJECT_RELATIVE_VEL] = aug_next_observations[:, OBJECT_VEL] - aug_next_observations[:, ROBOT_VEL]

    aug_goal = aug_next_observations[-1, OBJECT_POS]
    aug_observations[:, GOAL] = aug_goal
    aug_next_observations[:, GOAL] = aug_goal

    is_at_goal = np.linalg.norm(aug_next_observations[:, OBJECT_POS] - aug_next_observations[:, GOAL], axis=-1) < 0.05
    aug_rewards[:] = -1
    aug_rewards[is_at_goal] = 0
    aug_terminals[:] = False

    robot_pos = aug_observations[:, ROBOT_POS]
    obj_pos = aug_observations[:, OBJECT_POS]
    goal_pos = aug_observations[:, GOAL]

    plt.scatter(robot_pos[:, 0], robot_pos[:, 1], label='aug_robot')
    plt.scatter(obj_pos[:, 0], obj_pos[:, 1], label='aug_obj')
    plt.scatter(goal_pos[:, -3], goal_pos[:, -2], label='aug_goal')
    plt.legend()
    plt.show()
    plt.close()

def init_trajectory():
    trajectory = {
        'actions': [],
        'desired_goal': [],
        'next_observations': [],
        'observations': [],
        'rewards': [],
        'terminals': [],
        'infos': []
    }

    return trajectory

def append_trajectory(trajectories, trajectory):
    for key, val in trajectory.items():
        trajectories[key].append(val)

def get_trajectories(dataset, start_timestamp, end_timestamp):
    trajectory = init_trajectory()

    for i in range(start_timestamp, end_timestamp + 1):
        obs = dataset['observations'][i]
        next_obs = dataset['next_observations'][i]
        if is_contact(obs, next_obs):
            start_timestamp = i
            break
    print(f'start_timestamp = {start_timestamp}')
    for key in dataset:
        for i in range(start_timestamp, end_timestamp + 1):
            trajectory[key].append(dataset[key][i])
        trajectory[key] = np.array(trajectory[key])
    return trajectory


if __name__ == '__main__':
    dataset_path = f"/Users/yxqu/Desktop/Research/GuDA/GuidedDataAugmentationForRobotics/src/datasets/FetchSlide-v2/no_aug.hdf5"
    observed_dataset = load_dataset(dataset_path)
    n = len(observed_dataset['observations'])

    start_timestamp = 0
    for i in range(n):
        if observed_dataset['terminals'][i]:
            trajectory = get_trajectories(observed_dataset, start_timestamp, i)
            start_timestamp = (i + 1)
            aug_trajectory = generate_aug_trajectory(trajectory)
