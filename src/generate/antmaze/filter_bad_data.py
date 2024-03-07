import argparse
import os
from collections import defaultdict

import gym
import h5py
import numpy as np

import d4rl
from src.augment.augmentation_function_base import AugmentationFunctionBase
from src.generate.utils import reset_data, append_data, load_dataset, npify
from d4rl.locomotion.maze_env import GOAL, RESET

G = GOAL
R = RESET
U_MAZE = np.array([[1, 1, 1, 1, 1],
              [1, R, 0, 0, 1],
              [1, 1, 1, 0, 1],
              [1, G, 0, 0, 1],
              [1, 1, 1, 1, 1]]).T

MEDIUM_MAZE = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, 0, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]]
).T

LARGE_MAZE = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, G, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
).T

class AntMazeFilter(AugmentationFunctionBase):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.agent_offset = 4 # the agent and target coordinate systems are different for some reason.
        self.effective_wall_width = 1
        self.maze_scale = 4

        # precompute rotation matrices
        self.thetas = np.array([0, np.pi/2, np.pi, np.pi*3/2])
        self.rotation_matrices = []
        for theta in self.thetas:
            M = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            self.rotation_matrices.append(M)

        # store target goal
        l = len(self.env.maze_arr)
        if l == 5:
            self.target = np.array([0.75, 8.5])
            self.env.maze_arr = U_MAZE
        elif l == 8:
            self.target = np.array([20.5, 20.5])
            self.env.maze_arr = MEDIUM_MAZE
        else:
            self.target = np.array([32.5, 24.5])
            self.env.maze_arr = LARGE_MAZE

        # store valid maze locations (cells)
        self.wall_locations = []
        self.valid_locations = []
        width, height = self.env.maze_arr.shape
        for w in range(width):
            for h in range(height):
                location_type = self.env.maze_arr[w, h]
                box_location = np.array((w, h))
                if location_type in ['1']:
                    self.wall_locations.append(box_location)
                elif location_type in ['0', RESET, GOAL]:
                    self.valid_locations.append(box_location)

        self.wall_locations = np.array(self.wall_locations)
        self.valid_locations = np.array(self.valid_locations)

    def _sample_umaze(self):
        idx = np.random.choice(len(self.valid_locations))
        location = np.array(self.valid_locations[idx]).astype(self.env.observation_space.dtype)

        # bottom
        location = tuple(location)
        if location in [(1,3), (2,3)]:
            xy = self._sample_from_box(0, -0.5, 8.5, 0.5)

        # right side
        elif location in [(3,1), (3,2)]:

            xy = self._sample_from_box(8, 0, 9, 8)

        elif location in [(1,2), (2,1), (3,1)]:
            # top
            xy = self._sample_from_box(2, 9, 8, 8.5)
        else:
            xy = None

        return xy

    def _get_valid_boundaries(self, w, h):
        w = int(w)
        h = int(h)
        xhi = (w+1)*4-2
        yhi = (h+1)*4-2
        xlo = (w-1)*4+2
        ylo = (h-1)*4+2

        maze_width, maze_height = self.env.maze_arr.shape

        # Empty/goal locations are surrounded by walls, so we don't need to check if w+1/w-1/h+1/h-1 are valid locations.
        if w+1 < maze_width and self.env.maze_arr[w+1, h] in ['1']:
            xhi -= self.effective_wall_width
        if h+1 < maze_height and  self.env.maze_arr[w, h+1] in ['1']:
            yhi -= self.effective_wall_width
        if w-1 >= 0 and self.env.maze_arr[w-1, h] in ['1']:
            xlo += self.effective_wall_width
        if h-1 >= 0 and self.env.maze_arr[w, h-1] in ['1']:
            ylo += self.effective_wall_width

        xlo -= self.agent_offset
        ylo -= self.agent_offset
        xhi -= self.agent_offset
        yhi -= self.agent_offset

        return (xlo, ylo, xhi, yhi)

    def _sample_from_box(self, xlo, ylo, xhi, yhi):
        return self.env.np_random.uniform(low=[xlo, ylo], high=[xhi,yhi])

    def _is_in_wall(self, box_location, x, y):
        xlo, ylo = box_location*4 - self.agent_offset - 2 - self.effective_wall_width
        xhi, yhi = box_location*4 - self.agent_offset + 2 + self.effective_wall_width

        if (x > xlo and y > ylo) and (x < xhi and y < yhi):
            return True
        else:
            return False

    def _check_corners(self, xy, location):
        x, y = xy[0], xy[1]
        is_valid_position = True

        w, h = int(location[0]), int(location[1])
        for loc in [(w + 1, h + 1), (w + 1, h - 1), (w - 1, h + 1), (w - 1, h - 1)]:
            if self.env.maze_arr[loc[0], loc[1]] == '1':
                loc = np.array(loc)
                if self._is_in_wall(loc, x, y):
                    is_valid_position = False
                    break

        return is_valid_position

    def _sample_pos(self, n=1):
        idx = np.random.choice(len(self.valid_locations))
        location = np.array(self.valid_locations[idx]).astype(self.env.observation_space.dtype)
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
            # print(np.linalg.norm(next_obs[0:2] - (self.target )))
            reward = 1.0 if np.linalg.norm(next_obs[0:2] - self.target) <= 0.5 else 0.0
        elif self.env.reward_type == 'dense':
            reward = np.exp(-np.linalg.norm(next_obs[0:2] - self.target))
        else:
            raise ValueError('Unknown reward type %s' % self.env.reward_type)

        return reward

    def _xy_to_rowcol(self, xy):
        size_scaling = self.maze_scale
        xy = (max(xy[0], 1e-4), max(xy[1], 1e-4))
        return (int(np.round(1 + (xy[0]) / size_scaling)),
                int(np.round(1 + (xy[1]) / size_scaling)))

    def is_valid_input(self, obs, next_obs):
        r, c = self._xy_to_rowcol(obs[:2])
        # print(r,c)
        try:
            if self.env.maze_arr[r,c] == '1':
                return False
            x, y = obs[:2]
            xlo, ylo, xhi, yhi = self._get_valid_boundaries(r, c)
            if (x < xlo or x > xhi) or (y < ylo or y > yhi):
                return False

            r, c = self._xy_to_rowcol(next_obs[:2])
            if self.env.maze_arr[r,c] == '1':
                return False
            x, y = next_obs[:2]
            xlo, ylo, xhi, yhi = self._get_valid_boundaries(r, c)
            if (x < xlo or x > xhi) or (y < ylo or y > yhi):
                return False
        except:
            return False
        return True

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        if not self.is_valid_input(obs, next_obs):
            return None, None, None, None, None

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()
        is_valid = self.is_valid_input(aug_obs, aug_next_obs)
        if not is_valid:
            return None, None, None, None, None

        aug_location = self._xy_to_rowcol(aug_obs[:2])
        pos_is_valid = self._check_corners(aug_obs[:2], aug_location)
        next_pos_is_valid = self._check_corners(aug_next_obs[:2], aug_location)
        if not (pos_is_valid and next_pos_is_valid):
            return None, None, None, None, None


        aug_action = action.copy()
        # aug_obs[:2] += 0.5
        # aug_next_obs[:2] += 0.5
        aug_reward = self._reward(aug_next_obs)
        aug_done = aug_reward > 0

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

    f = AntMazeFilter(env)

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
