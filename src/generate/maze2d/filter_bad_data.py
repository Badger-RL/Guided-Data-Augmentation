# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import os

import gym
import h5py
import argparse

import d4rl
from src.generate.utils import reset_data, append_data, load_dataset, npify

import numpy as np
from src.augment.augmentation_function_base import AugmentationFunctionBase
from d4rl.pointmaze.maze_model import EMPTY, GOAL, WALL


class PointMazeFilter(AugmentationFunctionBase):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.agent_offset = 0.2  # the agent and target coordinate systems are different for some reason.
        self.effective_wall_width = 0.603
        self.target = self.env.get_target()
        self.wall_locations = []
        self.valid_locations = []
        width, height = self.env.maze_arr.shape
        for w in range(width):
            for h in range(height):
                location_type = self.env.maze_arr[w, h]
                box_location = np.array((w, h))
                if location_type in [WALL]:
                    self.wall_locations.append(box_location)
                elif location_type in [EMPTY, GOAL]:
                    self.valid_locations.append(box_location)
        self.valid_locations = np.array(self.valid_locations)

    def _get_valid_boundaries(self, w, h):

        w = int(w)
        h = int(h)

        # assert self.env.maze_arr[w, h] in [EMPTY, GOAL] # must be valid location

        xhi = w + 1 - 0.5
        yhi = h + 1 - 0.5
        xlo = w - 1 + 0.5
        ylo = h - 1 + 0.5

        if self.env.maze_arr[w + 1, h] in [WALL]:
            xhi = w + 1 - self.effective_wall_width
        if self.env.maze_arr[w, h + 1] in [WALL]:
            yhi = h + 1 - self.effective_wall_width
        if self.env.maze_arr[w - 1, h] in [WALL]:
            xlo = w - 1 + self.effective_wall_width
        if self.env.maze_arr[w, h - 1] in [WALL]:
            ylo = h - 1 + self.effective_wall_width

        xlo -= self.agent_offset
        ylo -= self.agent_offset
        xhi -= self.agent_offset
        yhi -= self.agent_offset

        return (xlo, ylo, xhi, yhi)

    def _sample_from_box(self, xlo, ylo, xhi, yhi):
        return self.env.np_random.uniform(low=[xlo, ylo], high=[xhi, yhi])

    def _is_in_wall(self, box_location, x, y):
        xlo, ylo = box_location - self.agent_offset - self.effective_wall_width
        xhi, yhi = box_location - self.agent_offset + self.effective_wall_width

        if (x > xlo and y > ylo) and (x < xhi and y < yhi):
            return True
        else:
            return False

    def _check_corners(self, xy, location):
        x, y = xy[0], xy[1]
        is_valid_position = True

        w, h = int(location[0]), int(location[1])
        for loc in [(w + 1, h + 1), (w + 1, h - 1), (w - 1, h + 1), (w - 1, h - 1)]:
            if self.env.maze_arr[loc[0], loc[1]] == WALL:
                loc = np.array(loc)
                if self._is_in_wall(loc, x, y):
                    is_valid_position = False
                    break

        return is_valid_position

    def _sample_pos(self, n=1):
        idx = np.random.choice(len(self.env.empty_and_goal_locations))
        location = np.array(self.env.empty_and_goal_locations[idx]).astype(self.env.observation_space.dtype)
        boundaries = self._get_valid_boundaries(*location)
        return self._sample_from_box(*boundaries), location

    def _sample_theta(self, **kwargs):
        return np.random.choice(self.thetas)

    def _sample_rotation_matrix(self, **kwargs):
        idx = np.random.randint(len(self.rotation_matrices))
        return self.rotation_matrices[idx]

    def _reward(self, next_obs):
        # Rewar dshould intuitively be computed using next_obs, but D4RL uses the current obs (D4RL bug)
        if self.env.reward_type == 'sparse':
            reward = 1.0 if np.linalg.norm(next_obs[0:2] - self.target) <= 0.5 else 0.0
        elif self.env.reward_type == 'dense':
            reward = np.exp(-np.linalg.norm(next_obs[0:2] - self.target))
        else:
            raise ValueError('Unknown reward type %s' % self.env.reward_type)

        return reward

    def _get_location(self, obs):
        for i in range(len(self.env.empty_and_goal_locations)):
            location = np.array(self.env.empty_and_goal_locations[i]).astype(self.env.observation_space.dtype)
            boundaries = self._get_valid_boundaries(*location)
            if self._is_in_box(*boundaries, obs[0], obs[1]):
                return location
        return None

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs, ):

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()

        for i in range(len(self.wall_locations)):
            # location = np.array(self.env.empty_and_goal_locations[i]).astype(self.env.observation_space.dtype)
            location = self.wall_locations[i]
            if self._is_in_wall(location, obs[0], obs[1]):
                return None, None, None, None, None

        if len(self.env.maze_arr) == 8:
            if aug_obs[0] > 7:
                return None, None, None, None, None
        elif len(self.env.maze_arr) == 9:
            if aug_obs[1] > 11 or aug_obs[1] < 0:
                return None, None, None, None, None

        aug_action = action
        aug_reward = self._reward(aug_next_obs)
        aug_done = False

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='maze2d-umaze-v1')
    parser.add_argument('--observed-dataset-path', type=str, default='/Users/nicholascorrado/code/mocoda/datasets/maze2d-umaze-v1/mocoda.hdf5')
    parser.add_argument('--observed-dataset-frac', '-frac', type=float, default=None)
    parser.add_argument('--observed-dataset-size', '-size', type=int, default=10000000)

    parser.add_argument('--aug-func', type=str, default='guided')
    parser.add_argument('--aug-size', type=int, default=int(1e6))
    parser.add_argument('--save-dir', '-fd', type=str, default='maze2d-umaze-v1')
    parser.add_argument('--save-name', '-fn', type=str, default='mocoda.hdf5')

    parser.add_argument('--check-valid', type=int, default=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f'../datasets/{args.env_id}'
    if args.save_name is None:
        args.save_name = f'{args.aug_func}_{int(args.aug_size/1e3)}k.hdf5'

    env = gym.make(args.env_id)
    np.random.seed(seed=args.seed)

    if args.observed_dataset_path:
        observed_dataset = load_dataset(args.observed_dataset_path)
        original_observed_dataset = None
    else:
        original_observed_dataset = None
        observed_dataset = d4rl.qlearning_dataset(env)

    if args.observed_dataset_frac:
        n = observed_dataset['observations'].shape[0]
        end = int(n * args.observed_dataset_frac)
    elif args.observed_dataset_size:
        end = args.observed_dataset_size
    else:
        end = observed_dataset['observations'].shape[0]
    for key in observed_dataset:
        observed_dataset[key] = observed_dataset[key][:end]
    n = observed_dataset['observations'].shape[0]

    observed_dataset_obs = observed_dataset['observations']
    observed_dataset_action = observed_dataset['actions']
    observed_dataset_next_obs = observed_dataset['next_observations']

    f = PointMazeFilter(env)

    aug_dataset = reset_data()
    aug_count = 0 # number of valid augmentations produced
    i = 0
    while aug_count < int(1e6):
        idx = i % n
        obs, action, reward, next_obs, done = f.augment(
            obs=observed_dataset_obs[idx],
            action=observed_dataset_action[idx],
            next_obs=observed_dataset_next_obs[idx],
            reward=None,
            done=None,
        )

        i += 1
        if obs is not None:
            aug_count += 1
            if aug_count % 10000 == 0: print('aug_count:', aug_count)
            append_data(aug_dataset, obs, action, reward, next_obs, done)
        # if aug_count >= n * m:
        #     break

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = f'{args.save_dir}/{args.save_name}'
    new_dataset = h5py.File(save_path, 'w')
    npify(aug_dataset)
    for k in aug_dataset:
        if k == 'truncateds': continue
        data = np.concatenate([aug_dataset[k]])
        new_dataset.create_dataset(k, data=data, compression='gzip')
    new_dataset.create_dataset('original_size', data=n)
    new_dataset.create_dataset('aug_size', data=aug_count)

    print(f"New dataset size: {len(new_dataset['observations'])}")

