import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from src.generate.utils import reset_data, append_data, load_dataset, npify
import time
import h5py


def init_trajectory():
    trajectory = {
        'actions': [],
        'desired_goal': [],
        'next_observations': [],
        'observations': [],
        'rewards': [],
        'terminals': []
    }

    return trajectory

def append_trajectory(trajectories, trajectory):
    for key, val in trajectory.items():
        trajectories[key].append(val)

def get_trajectories(dataset, start_timestamp, end_timestamp):
    trajectory = init_trajectory()

    for key in dataset:
        for i in range(start_timestamp, end_timestamp + 1):
            trajectory[key].append(dataset[key][i])
        trajectory[key] = np.array(trajectory[key])
    return trajectory

GOALS = [
    # top left
    [-0.26, -0.14, 0, 0, 0, -1],
    [-0.22, -0.14, 0, 0, 0, -1],
    [-0.18, -0.14, 0, 0, 0, -1],
    [-0.14, -0.14, 0, 0, 0, -1],
    [-0.10, -0.14, 0, 0, 0, -1],
    [-0.06, -0.14, 0, 0, 0, -1],
    [-0.02, -0.14, 0, 0, 0, -1],
    # top right
    [0.26, -0.14, 0, 0, 0, -1],
    [0.22, -0.14, 0, 0, 0, -1],
    [0.18, -0.14, 0, 0, 0, -1],
    [0.14, -0.14, 0, 0, 0, -1],
    [0.10, -0.14, 0, 0, 0, -1],
    [0.06, -0.14, 0, 0, 0, -1],
    [0.02, -0.14, 0, 0, 0, -1],
    # bottom left
    [-0.26, 0.14, 0, 0, 0, 1],
    [-0.22, 0.14, 0, 0, 0, 1],
    [-0.18, 0.14, 0, 0, 0, 1],
    [-0.14, 0.14, 0, 0, 0, 1],
    [-0.10, 0.14, 0, 0, 0, 1],
    [-0.06, 0.14, 0, 0, 0, 1],
    [-0.02, 0.14, 0, 0, 0, 1],
    # bottom right
    [0.26, 0.14, 0, 0, 0, 1],
    [0.22, 0.14, 0, 0, 0, 1],
    [0.18, 0.14, 0, 0, 0, 1],
    [0.14, 0.14, 0, 0, 0, 1],
    [0.10, 0.14, 0, 0, 0, 1],
    [0.06, 0.14, 0, 0, 0, 1],
    [0.02, 0.14, 0, 0, 0, 1],
]
GOALS = np.array(GOALS)


def is_valid(obs):
    x, y = obs[:, 0], obs[:, 1]

    if np.all((x >= -0.26) & (x <= 0.26) & (y >= -0.15) & (y <= 0.15)):
        return True
    else:
        return False
    
def generate_aug_trajectory(trajectory):


    aug_trajectory = init_trajectory()
    n = len(trajectory['observations'])
    original_desired_goal = trajectory['desired_goal'][0]

    # new_desired_goal = state[0]['desired_goal']
    # idx = np.random.randint(len(GOALS))
    # new_desired_goal = GOALS[idx]

    success = False
    max_attempts = 10
    attempts = 0
    while not success and attempts < max_attempts:
        attempts += 1
        env = gym.make('parking-v0', render_mode='rgb_array')
        state = env.reset()

        idx = np.random.randint(len(GOALS))
        new_desired_goal = GOALS[idx]
        # new_desired_goal[0] = -0.18
        # print(new_desired_goal)

        final_pos = trajectory['observations'][-1, :2]
        delta_to_goal = new_desired_goal[:2] - final_pos

        # compute theta
        final_heading = np.arctan2(trajectory['observations'][-1, 5], trajectory['observations'][-1, 4])
        if new_desired_goal[-1] > 0:
            desired_theta = np.pi/2
        else:
            desired_theta = -np.pi/2

        # compute rotation matrix
        # theta = np.random.uniform(-np.pi, np.pi)
        theta = desired_theta - final_heading
        M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        original_obs = trajectory['observations']
        original_next_obs = trajectory['next_observations']

        aug_obs = original_obs.copy()
        aug_next_obs = original_next_obs.copy()

        # TRANSLATE
        # set goal
        aug_obs[:, 6:] = new_desired_goal.copy()
        aug_next_obs[:, 6:] = new_desired_goal.copy()

        # translate such that final pos is at goal
        aug_obs[:, :2] += delta_to_goal
        aug_next_obs[:, :2] += delta_to_goal

        # ROTATE
        # set origin to goal, since we are rotating about the goal.
        aug_obs[:, :2] -= new_desired_goal[:2]
        aug_next_obs[:, :2] -= new_desired_goal[:2]

        # rotate positions about desired goal
        aug_obs[:, :2] = M.dot(aug_obs[:, :2].T).T
        aug_next_obs[:, :2] = M.dot(aug_next_obs[:, :2].T).T
        # rotate velocities
        aug_obs[:, 2:4] = M.dot(aug_obs[:, 2:4].T).T
        aug_next_obs[:, 2:4] = M.dot(aug_next_obs[:, 2:4].T).T

        # shift origin back to normal
        aug_obs[:, :2] += new_desired_goal[:2]
        aug_next_obs[:, :2] += new_desired_goal[:2]

        # rotate heading
        heading = np.arctan2(aug_obs[:, 5], aug_obs[:, 4])
        next_heading = np.arctan2(aug_next_obs[:, 5], aug_next_obs[:, 4])
        aug_heading = heading + theta
        aug_next_heading = next_heading + theta
        aug_obs[:, 4] = np.cos(aug_heading)
        aug_obs[:, 5] = np.sin(aug_heading)
        aug_next_obs[:, 4] = np.cos(aug_next_heading)
        aug_next_obs[:, 5] = np.sin(aug_next_heading)

        # action does not change, since it's defined related to the car's heading.

        ## TODO how to get achieved goal, reward, and terminal:
        achieved_goal = aug_next_obs[:, :6].copy()
        p = 0.5
        aug_reward = -np.power(np.dot(np.abs(achieved_goal - new_desired_goal), np.array(env.config["reward_weights"])), p)
        # aug_reward = env.compute_reward(achieved_goal, new_desired_goal, {})
        # print(aug_reward)
        aug_action = trajectory['actions'].copy()
        # aug_terminal = trajectory['terminals'].copy()
        aug_terminal = aug_reward > -env.config['success_goal_reward']
        print(np.any(aug_terminal))

        # rotate position
        if (not is_valid(aug_obs)) or (not is_valid(aug_next_obs)):
            success = False
            break

        aug_trajectory['observations'] = aug_obs
        aug_trajectory['actions'] = aug_action
        aug_trajectory['next_observations'] = aug_next_obs
        aug_trajectory['rewards'] = aug_reward
        aug_trajectory['terminals'] = aug_terminal
        success = True
    # print(f'success = {success}')
    if not success:
        return None
    # print(len(aug_trajectory['observations']))
    return aug_trajectory


dataset_path = f"../../datasets/parking-v0/no_aug.hdf5"
observed_dataset = load_dataset(dataset_path)

n = len(observed_dataset['observations'])

aug_trajectories = init_trajectory()


max_aug = 1e4
aug_count = 0

while aug_count < max_aug:
    start_timestamp = 0
    for i in range(n):
        if observed_dataset['terminals'][i]:
            trajectory = get_trajectories(observed_dataset, start_timestamp, i)
            start_timestamp = (i + 1)
            aug_trajectory = generate_aug_trajectory(trajectory)
            if aug_trajectory is None:
                continue
            append_trajectory(aug_trajectories, aug_trajectory)
            aug_count += len(aug_trajectory['observations'])
    print(f'aug_count = {aug_count}')

dataset = h5py.File("../../datasets/parking-v0/guided.hdf5", 'w')
for k in aug_trajectories:
    aug_trajectories[k] = np.concatenate(aug_trajectories[k])
    dataset.create_dataset(k, data=np.array(aug_trajectories[k]), compression='gzip')



