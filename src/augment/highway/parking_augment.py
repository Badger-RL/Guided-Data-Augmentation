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
    for key in trajectory:
        for i in range(len(trajectory[key])):
            trajectories[key].append(trajectory[key][i])

def get_trajectories(dataset, start_timestamp, end_timestamp):
    trajectory = init_trajectory()

    for key in dataset:
        for i in range(start_timestamp, end_timestamp + 1):
            trajectory[key].append(dataset[key][i])
        trajectory[key] = np.array(trajectory[key])
    return trajectory

GOALS = [
    # top left
    [-0.26, -0.14, 0, 0, 1, 0],
    [-0.22, -0.14, 0, 0, 1, 0],
    [-0.18, -0.14, 0, 0, 1, 0],
    [-0.14, -0.14, 0, 0, 1, 0],
    [-0.10, -0.14, 0, 0, 1, 0],
    [-0.6, -0.14, 0, 0, 1, 0],
    [-0.2, -0.14, 0, 0, 1, 0],
    # top right
    [0.26, -0.14, 0, 0, 1, 0],
    [0.22, -0.14, 0, 0, 1, 0],
    [0.18, -0.14, 0, 0, 1, 0],
    [0.14, -0.14, 0, 0, 1, 0],
    [0.10, -0.14, 0, 0, 1, 0],
    [0.6, -0.14, 0, 0, 1, 0],
    [0.2, -0.14, 0, 0, 1, 0],
    # bottom left
    [-0.26, 0.14, 0, 0, 0, -1],
    [-0.22, 0.14, 0, 0, 0, -1],
    [-0.18, 0.14, 0, 0, 0, -1],
    [-0.14, 0.14, 0, 0, 0, -1],
    [-0.10, 0.14, 0, 0, 0, -1],
    [-0.6, 0.14, 0, 0, 0, -1],
    [-0.2, 0.14, 0, 0, 0, -1],
    # bottom right
    [0.26, 0.14, 0, 0, 0, -1],
    [0.22, 0.14, 0, 0, 0, -1],
    [0.18, 0.14, 0, 0, 0, -1],
    [0.14, 0.14, 0, 0, 0, -1],
    [0.10, 0.14, 0, 0, 0, -1],
    [0.6, 0.14, 0, 0, 0, -1],
    [0.2, 0.14, 0, 0, 0, -1],
]
GOALS = np.array(GOALS)
#
# def sample_new_goal():
#     if np.random.random() < 0.5:
#         # top
#         idx = np.random.randint(1,7+1) # generate number in [1, 7]
#         is_left = np.random.random() < 0.5 # sample left spots w.p. 0.5
#         is_top = np.random.random() < 0.5 # sample top spots w.p. 0.5
#
#         pos = idx * 2



def is_valid(obs):
    x, y = obs[0], obs[1]
    if x >= -0.26 and x <= 0.26 and y >= -0.15 and y <= 0.15:
        return True
    else:
        return False
    
def generate_aug_trajectory(trajectory):

    env = gym.make('parking-v0', render_mode='rgb_array')
    
    aug_trajectory = init_trajectory()
    n = len(trajectory['observations'])
    original_desired_goal = trajectory['desired_goal'][0]

    state = env.reset()
    new_desired_goal = state[0]['desired_goal']
    idx = np.random.randint(len(GOALS))
    new_desired_goal = GOALS[idx]

    while new_desired_goal[1] * original_desired_goal[1] < 0:
        state = env.reset()
        new_desired_goal = state[0]['desired_goal']
    print(f"original_desired_goal: {original_desired_goal}")
    print(f"new_desired_goal: {new_desired_goal}")

    final_pos = trajectory['observations'][-1, :2]
    delta_to_goal = new_desired_goal[:2] - final_pos


    for i in range(n):
        original_obs = trajectory['observations'][i]
        original_next_obs = trajectory['next_observations'][i]

        aug_obs = original_obs.copy()
        aug_next_obs = original_next_obs.copy()

        # TRANSLATE
        # set goal
        aug_obs[6:] = new_desired_goal.copy()
        aug_next_obs[6:] = new_desired_goal.copy()

        # translate such that final pos is at goal
        aug_obs[:2] += delta_to_goal
        aug_next_obs[:2] += delta_to_goal


        # ROTATE
        # set origin to goal, since we are rotating about the goal.
        aug_obs[:2] -= new_desired_goal[:2]
        aug_next_obs[:2] -= new_desired_goal[:2]

        # compute rotation matrix
        theta = np.random.uniform(-np.pi/2, np.pi/2)
        M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # rotate positions about desired goal
        aug_obs[:2] = M.dot(aug_obs[:2]).T
        aug_next_obs[:2] = M.dot(aug_next_obs[:2]).T
        # rotate velocities
        aug_obs[2:4] = M.dot(aug_obs[2:4]).T
        aug_next_obs[2:4] = M.dot(aug_next_obs[2:4]).T

        # shift origin back to normal
        aug_obs[:2] += new_desired_goal[:2]
        aug_next_obs[:2] += new_desired_goal[:2]

        # rotate heading
        heading = np.arctan2(aug_obs[5], aug_obs[4])
        next_heading = np.arctan2(aug_next_obs[5], aug_next_obs[4])
        aug_heading = heading + theta
        aug_next_heading = next_heading + theta
        aug_obs[4] = np.cos(aug_heading)
        aug_obs[5] = np.sin(aug_heading)
        aug_next_obs[4] = np.cos(aug_next_heading)
        aug_next_obs[5] = np.sin(aug_next_heading)

        # action does not change, since it's defined related to the car's heading.

        ## TODO how to get achieved goal, reward, and terminal:
        achieved_goal = aug_next_obs[:6].copy()
        aug_reward = env.compute_reward(achieved_goal, new_desired_goal, {})
        print(aug_reward)
        aug_action = trajectory['actions'][i]
        aug_terminal = trajectory['terminals'][i]

        # rotate position
        if (not is_valid(aug_obs)) or (not is_valid(aug_next_obs)):
            break

        aug_trajectory['observations'].append(aug_obs)
        aug_trajectory['actions'].append(aug_action)
        aug_trajectory['next_observations'].append(aug_next_obs)
        aug_trajectory['rewards'].append(aug_reward)
        aug_trajectory['terminals'].append(aug_terminal)
        aug_trajectory['desired_goal'].append(new_desired_goal)
        aug_state = {
            'observation': aug_obs.copy(),
            'desired_goal': new_desired_goal.copy(),
            'achieved_goal': achieved_goal.copy()
        }
        env.set_state(aug_state)
        new_state, _, _, _, _ = env.step(aug_action)
        true_next_obs = new_state['observation']
        if len(aug_trajectory['observations']) != 0:
            print(f"{i} difference: {true_next_obs - aug_next_obs[:6]}")
    return aug_trajectory

#
# def generate_aug_trajectory(trajectory):
#
#     env = gym.make('parking-v0', render_mode='rgb_array')
#     env.reset()
#
#     n = len(trajectory['observations'])
#
#     success = False
#     max_attempts = 1000
#     attempts = 0
#     while not success and attempts < max_attempts:
#         attempts += 1
#         aug_trajectory = init_trajectory()
#
#         # sample random goal
#         idx = np.random.randint(len(GOALS))
#         new_desired_goal = GOALS[idx]
#         # new_desired_goal[0] = 0.26
#         # new_desired_goal[1] = 0
#
#         # compute displacement between new goal to car's final position.
#         final_pos = trajectory['observations'][-1, :2]
#         delta_to_goal = new_desired_goal[:2] - final_pos
#
#         for i in range(n):
#             original_obs = trajectory['observations'][i]
#             original_next_obs = trajectory['next_observations'][i]
#
#             aug_obs = original_obs.copy()
#             aug_next_obs = original_next_obs.copy()
#
#             # TRANSLATE
#             # set goal
#             aug_obs[6:] = new_desired_goal.copy()
#             aug_next_obs[6:] = new_desired_goal.copy()
#
#             # translate such that final pos is at goal
#             aug_obs[:2] += delta_to_goal
#             aug_next_obs[:2] += delta_to_goal
#
#
#             # ROTATE
#             # set origin to goal, since we are rotating about the goal.
#             aug_obs[:2] -= new_desired_goal[:2]
#             aug_next_obs[:2] -= new_desired_goal[:2]
#
#             # compute rotation matrix
#             theta = np.random.uniform(-np.pi/2, np.pi/2)
#             M = np.array([
#                 [np.cos(theta), -np.sin(theta)],
#                 [np.sin(theta), np.cos(theta)]
#             ])
#
#             # rotate positions about desired goal
#             aug_obs[:2] = M.dot(aug_obs[:2]).T
#             aug_next_obs[:2] = M.dot(aug_next_obs[:2]).T
#             # rotate velocities
#             aug_obs[2:4] = M.dot(aug_obs[2:4]).T
#             aug_next_obs[2:4] = M.dot(aug_next_obs[2:4]).T
#
#             # shift origin back to normal
#             aug_obs[:2] += new_desired_goal[:2]
#             aug_next_obs[:2] += new_desired_goal[:2]
#
#             # rotate heading
#             heading = np.arctan2(aug_obs[5], aug_obs[4])
#             next_heading = np.arctan2(aug_next_obs[5], aug_next_obs[4])
#             aug_heading = heading + theta
#             aug_next_heading = next_heading + theta
#             aug_obs[4] = np.cos(aug_heading)
#             aug_obs[5] = np.sin(aug_heading)
#             aug_next_obs[4] = np.cos(aug_next_heading)
#             aug_next_obs[5] = np.sin(aug_next_heading)
#
#             # action does not change, since it's defined related to the car's heading.
#
#             ## TODO how to get achieved goal, reward, and terminal:
#             achieved_goal = aug_next_obs[:6].copy()
#             aug_reward = env.compute_reward(achieved_goal, new_desired_goal, {})
#             aug_action = trajectory['actions'][i]
#             aug_terminal = trajectory['terminals'][i]
#
#             if (not is_valid(aug_obs)) or (not is_valid(aug_next_obs)):
#                 success = False
#                 break
#
#             aug_trajectory['observations'].append(aug_obs)
#             aug_trajectory['actions'].append(aug_action)
#             aug_trajectory['next_observations'].append(aug_next_obs)
#             aug_trajectory['rewards'].append(aug_reward)
#             aug_trajectory['terminals'].append(aug_terminal)
#             aug_trajectory['desired_goal'].append(new_desired_goal)
#             aug_state = {
#                 'observation': aug_obs.copy(),
#                 'desired_goal': new_desired_goal.copy(),
#                 'achieved_goal': achieved_goal.copy()
#             }
#             env.set_state(aug_state)
#             new_state, _, _, _, _ = env.step(aug_action)
#             true_next_obs = new_state['observation']
#             if len(aug_trajectory['observations']) != 0:
#                 print(f"{i} difference: {true_next_obs - aug_next_obs[:6]}")
#
#     print(f'success = {success}')
#     # if not success:
#     #     return None
#     print(len(aug_trajectory['observations']))
#     return aug_trajectory


dataset_path = f"../../datasets/parking-v0/no_aug.hdf5"
observed_dataset = load_dataset(dataset_path)

n = len(observed_dataset['observations'])

start_timestamp = 0
aug_trajectories = init_trajectory()

for i in range(n):
    if observed_dataset['terminals'][i]:
        trajectory = get_trajectories(observed_dataset, start_timestamp, i)
        start_timestamp = i + 1
        aug_trajectory = generate_aug_trajectory(trajectory)
        if aug_trajectory is None:
            continue
        append_trajectory(aug_trajectories, aug_trajectory)

dataset = h5py.File("tmp.hdf5", 'w')
for k in aug_trajectories:
    dataset.create_dataset(k, data=np.array(aug_trajectories[k]), compression='gzip')



