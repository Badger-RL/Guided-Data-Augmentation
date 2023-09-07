from collections import defaultdict

import numpy as np
from src.augment.augmentation_function_base import AugmentationFunctionBase
from d4rl.locomotion.maze_env import GOAL, RESET

'''
    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        # if not self._check_valid_input(obs, next_obs):
        #     return None, None, None, None, None

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()
        while True:
            # new_pos, aug_location = self._sample_pos()
            new_pos = aug_obs[:2]
            aug_location = self._xy_to_rowcol(aug_obs[:2])
            # print(new_pos,aug_location)
            guide_alpha, rotate_alpha = self._sample_theta(obs, next_obs, new_pos, aug_location)

            M = np.array([
                [np.cos(-rotate_alpha), -np.sin(-rotate_alpha)],
                [np.sin(-rotate_alpha), np.cos(-rotate_alpha)]
            ])

            aug_obs[:2] = new_pos
            delta_pos = next_obs[:2] - obs[:2]
            rotated_delta_obs = M.dot(delta_pos[:2]).T
            aug_next_obs[:2] = aug_obs[:2] + rotated_delta_obs

            # mujoco stores quats as (qw, qx, qy, qz)
            sin = np.sin(rotate_alpha / 2)
            cos = np.cos(rotate_alpha / 2)
            quat_rotate_by = np.array([sin, 0, 0, cos])
            self._rotate_torso(aug_obs, quat_rotate_by)
            self._rotate_torso(aug_next_obs, quat_rotate_by)

            # sin = np.sin(guide_alpha / 2)
            # cos = np.cos(guide_alpha / 2)
            # delta_quat = aug_next_obs[3:6+1] - aug_obs[3:6+1]
            # aug_obs[3] = sin
            # aug_obs[6] = cos
            # aug_next_obs[3] = cos + delta_quat[0]
            # aug_next_obs[6] = sin + delta_quat[3]

            # Not sure why we need -alpha here...
            sin = np.sin(-rotate_alpha)
            cos = np.cos(-rotate_alpha)
            self._rotate_vel(aug_obs, sin, cos)
            self._rotate_vel(aug_next_obs, sin, cos)
            break

        aug_action = action.copy()
        aug_reward = self._reward(aug_next_obs)
        aug_done = done


        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done
'''

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

class AntMazeAugmentationFunction(AugmentationFunctionBase):
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


    # def _sample_umaze(self, obs, last_obs):
    #     x, y = obs[0], obs[1]
    #
    #     # bottom
    #     if x > 0 and x < 8.5 and y > -2 and y < 2:
    #         new_pos = np.random.uniform(
    #             low=np.array([0, -1]),
    #             high=np.array([8, 1])
    #         )
    #
    #     # right side
    #     elif x > 7 and x < 10 and y > 0 and y < 8:
    #         new_pos = np.random.uniform(
    #             low=np.array([7, -1]),
    #             high=np.array([9, 8])
    #         )
    #     elif x > 2 and x < 9 and y > 8 and y < 8.5:
    #         new_pos = np.random.uniform(
    #             low=np.array([0, 7]),
    #             high=np.array([9, 9])
    #         )
    #     else:
    #         new_pos = None
    #
    #     return new_pos


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
        # if np.random.random() < 0.3:
            # idx = np.random.choice(3)
            # location = np.array([(1,3), (2,3), (3,3)])[idx]
            # location = np.array([1,3])
        # else:
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

    def quat_mul(self, quat0, quat1):
        # assert quat0.shape == quat1.shape
        # assert quat0.shape[-1] == 4

        # mujoco stores quats as (qw, qx, qy, qz)
        w0 = quat0[..., 3]
        x0 = quat0[..., 0]
        y0 = quat0[..., 1]
        z0 = quat0[..., 2]

        w1 = quat1[..., 3]
        x1 = quat1[..., 0]
        y1 = quat1[..., 1]
        z1 = quat1[..., 2]

        w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
        z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
        quat = np.stack([x, y, z, w], axis=-1)

        # assert quat.shape == quat0.shape
        return quat

    def _rotate_torso(self, obs, quat_rotate_by):
        quat_curr = obs[3:6 + 1]
        quat_result = self.quat_mul(quat0=quat_curr, quat1=quat_rotate_by)
        # quat already normalized
        obs[3:6 + 1] = quat_result

    def _rotate_vel(self, obs, sin, cos):
        x = obs[2+13].copy()
        y = obs[2+14].copy()
        obs[2+13] = x * cos - y * sin
        obs[2+14] = x * sin + y * cos

    def _xy_to_rowcol(self, xy):
        size_scaling = self.maze_scale
        xy = (max(xy[0], 1e-4), max(xy[1], 1e-4))
        return (int(np.round(1 + (xy[0]) / size_scaling)),
                int(np.round(1 + (xy[1]) / size_scaling)))

    def is_valid_input(self, obs, next_obs):
        r, c = self._xy_to_rowcol(obs[:2])
        # print(r,c)
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
        while True:
            new_pos, aug_location = self._sample_pos()
            # new_pos = aug_obs[:2]
            # aug_location = self._xy_to_rowcol(aug_obs[:2])
            # print(new_pos,aug_location)
            rotate_alpha = np.random.uniform(0, 2*np.pi)
            rotate_alpha = 0

            M = np.array([
                [np.cos(-rotate_alpha), -np.sin(-rotate_alpha)],
                [np.sin(-rotate_alpha), np.cos(-rotate_alpha)]
            ])

            aug_obs[:2] = new_pos
            delta_pos = next_obs[:2] - obs[:2]
            rotated_delta_obs = M.dot(delta_pos[:2]).T
            aug_next_obs[:2] = aug_obs[:2] + rotated_delta_obs

            # corner case (literally): check that the agent isn't inside a wall
            pos_is_valid = self._check_corners(aug_obs[:2], aug_location)
            next_pos_is_valid = self._check_corners(aug_next_obs[:2], aug_location)

            # if new positions are not valid, immediately sample a new position
            if not (pos_is_valid and next_pos_is_valid):
                continue

            # mujoco stores quats as (qw, qx, qy, qz) internally but uses (qx, qy, qz, qw) in the observation
            sin = np.sin(rotate_alpha / 2)
            cos = np.cos(rotate_alpha / 2)
            quat_rotate_by = np.array([sin, 0, 0, cos])
            self._rotate_torso(aug_obs, quat_rotate_by)
            self._rotate_torso(aug_next_obs, quat_rotate_by)

            sin = np.sin(-rotate_alpha)
            cos = np.cos(-rotate_alpha)
            self._rotate_vel(aug_obs, sin, cos)
            self._rotate_vel(aug_next_obs, sin, cos)

            break

        aug_action = action.copy()
        aug_obs[:2] += 0.5
        aug_next_obs[:2] += 0.5
        aug_reward = self._reward(aug_next_obs)
        aug_done = aug_reward > 0

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done


guide_thetas = {
    (1, 1): [0],
    (2, 1): [0],
    (3, 1): [np.pi / 2],

    (3, 2): [np.pi / 2],

    (1, 3): [0, np.pi / 2, np.pi, np.pi * 3 / 2],
    (2, 3): [np.pi],
    (3, 3): [np.pi]
}


class AntMazeTrajectoryGuidedAugmentationFunction(AntMazeAugmentationFunction):

    def __init__(self, env, **kwargs):

        super().__init__(env, **kwargs)

        l = len(self.env.maze_arr)
        if l == 5:
            self.get_valid_locations = self.valid_locations_umaze
            self.get_guide_theta = self.get_guide_theta_umaze
        elif l == 8:
            self.get_valid_locations = self.valid_locations_medium
            self.get_guide_medium = self.get_guide_theta_medium
        else:
            self.get_valid_locations = self.valid_locations_large
            self.get_guide_large = self.get_guide_theta_large

    def _get_location(self, obs):
        #location = (int(np.round(obs[0]+self.agent_offset)), int(np.round(obs[1]+self.agent_offset)))
        for i in range(len(self.valid_locations)):
            location = np.array(self.valid_locations[i]).astype(self.env.observation_space.dtype)
            boundaries = self._get_valid_boundaries(*location)
            if self._is_in_box(*boundaries, obs[0], obs[1]):
                return location
        return None

    def _sample_theta(self, obs, next_obs, new_pos, new_location, **kwargs):

        # delta_obs = next_obs[:2] - obs[:2]
        # theta = np.arctan2(delta_obs[1], delta_obs[0])

        # guide_thetas = self.guide_thetas[(int(new_location[0]), int(new_location[1]))]
        # guide_theta = np.random.choice(guide_thetas)

        # aug_theta = -(guide_theta - theta) #+ np.random.uniform(low=-np.pi/6, high=np.pi/6)

        if new_pos[0] < 7.5 and new_pos[1] > -2 and new_pos[1] < 2:
            guide_theta = 0
        elif new_pos[0] > 7.5 and new_pos[0] < 9 and new_pos[1] < 7.5:
            guide_theta = np.pi / 2
        else:
            guide_theta = np.pi

        # print(guide_theta)
        # guide_theta = 0

        delta_obs = next_obs - obs
        theta = np.arctan2(delta_obs[1], delta_obs[0])
        # theta = 2 * np.arctan2(delta_obs[3], delta_obs[6])

        aug_theta = -(guide_theta - theta)  # + np.random.uniform(low=-np.pi/6, high=np.pi/6)

        do_reflect = np.abs(theta-guide_theta) > np.pi*2/3

        return aug_theta, do_reflect

    def valid_locations_umaze(self, direction):

        valid_locations = []
        if direction == 0: # random
            valid_locations = [(1, 3)]
            # valid_locations = [(1, 3), (1, 1), (2, 1), (3, 1), (3, 2), (2, 3), (3, 3)]
        elif direction == 1: # right
            valid_locations = [(1, 1), (2, 1)]
        elif direction == 2: # up
            valid_locations = [(3, 1), (3, 2)]
        elif direction == 3: # left
            valid_locations = [(2, 3), (3, 3)]
        elif direction == 4:
            valid_locations = []

        return valid_locations

    def get_guide_theta_umaze(self, location):

        location = (int(location[0]), int(location[1]))
        if location in [(1, 3)]:
            return np.random.uniform(0, 2 * np.pi)
        elif location in [(1, 1), (2, 1)]: # right
            return 0
        elif location in [(3, 1), (3, 2)]: # up
            return np.pi/2
        elif location in  [(2, 3), (3, 3)]: # left
            return np.pi


    def valid_locations_medium(self, direction):

        valid_locations = []
        if direction == 0:  # random
            valid_locations = [(6, 6)]
        elif direction == 1:  # right
            valid_locations = [(1, 1), (1, 2),
                               (2, 3),
                               (3, 3),
                               (4, 4),
                               (5, 4),
                               (5, 6)]
        elif direction == 2:  # up
            valid_locations = [(2, 1), (2, 2),
                               (4, 2), (4, 3),
                               (6, 4), (6, 5),]
        elif direction == 3:  # left
            valid_locations = []
        elif direction == 4: # down
            valid_locations = [(2, 4),
                               (4, 5),]

        return valid_locations

    def valid_locations_large(self, direction):

        valid_locations = []
        if direction == 0:  # random
            valid_locations = [(9, 7)]
        elif direction == 1:  # right
            valid_locations = [(1, 1), (1, 3),
                               (2, 1), (2, 3),
                               (3, 1), (3, 3),
                               (4, 3),
                               (5, 3),
                               (6, 5),
                               (7, 5),
                               (8, 7),
                               ]
        elif direction == 2:  # up
            valid_locations = [(1, 1),
                               (4, 1), (4, 2),
                               (6, 1), (6, 2), (6, 3), (6, 4),
                               (8, 5), (8, 6),
                               ]
        elif direction == 3:  # left
            valid_locations = [
                               (9, 5),
                               (10, 7)]
        elif direction == 4: # down
            valid_locations = [(1, 4),
                               (6, 6), (6, 7),]

        return valid_locations


    def get_direction(self, obs, final_obs):
        init_pos = obs[:2]
        final_pos = final_obs[:2]

        delta = final_pos - init_pos
        theta = np.arctan2(delta[1], delta[0])

        if np.linalg.norm(delta) < 2:
            return 0
        if np.abs(theta - 0) < np.pi/4:
            return 1
        if np.abs(theta - np.pi/2) < np.pi/2:
            return 2
        if np.abs(theta - np.pi) < np.pi/2:
            return 3
        if np.abs(theta + np.pi/2) < np.pi/2:
            return 4

    def augment_trajectory(self, trajectory: dict, direction):
        length = len(trajectory['observations'])

        # print(direction, direction2)

        augmented_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
        }
        is_new_trajectory = True

        for i in range(length):
            observation = trajectory['observations'][i]
            action = trajectory['actions'][i]
            next_observation = trajectory['next_observations'][i]


            if is_new_trajectory:
                direction = self.get_direction(trajectory['observations'][i], trajectory['next_observations'][-1])
                valid_locations = self.get_valid_locations(direction)
                if len(valid_locations) == 0:
                    return augmented_trajectory

                idx = np.random.choice(len(valid_locations))
                location = np.array(valid_locations[idx]).astype(self.env.observation_space.dtype)
                boundary = self._get_valid_boundaries(*location)
                new_origin = self._sample_from_box(*boundary)
                delta_pos = new_origin[:2] - observation[:2]

            augmented_obs = observation.copy()
            augmented_next_obs = next_observation.copy()
            augmented_action = action.copy()
            # augmented_reward = reward.copy()
            # augmented_done = done.copy()

            augmented_obs[:2] = observation[:2] + delta_pos
            augmented_next_obs[:2] = next_observation[:2] + delta_pos
            augmented_reward = self._reward(augmented_next_obs)
            augmented_done = augmented_reward > 0  # recompute reward *after* making all changes to observations.

            aug_location = self._get_location(augmented_obs)
            if aug_location is None:
                is_new_trajectory = True
                i -= 1
                continue
            pos_is_valid = self._check_corners(augmented_obs[:2], aug_location)
            next_pos_is_valid = self._check_corners(augmented_next_obs[:2], aug_location)

            if not (pos_is_valid and next_pos_is_valid):
                is_new_trajectory = True
                i -= 1
                continue

            if self.is_valid_input(augmented_obs, augmented_next_obs):
                is_new_trajectory = False
                augmented_trajectory['observations'].append(augmented_obs)
                augmented_trajectory['actions'].append(augmented_action)
                augmented_trajectory['rewards'].append(augmented_reward)
                augmented_trajectory['next_observations'].append(augmented_next_obs)
                augmented_trajectory['terminals'].append(augmented_done)
            else:
                is_new_trajectory = True
                i -= 1
        return augmented_trajectory

    def augment_trajectory(self, trajectory: dict, direction):
        length = len(trajectory['observations'])

        # print(direction, direction2)

        augmented_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
        }
        is_new_trajectory = True

        for i in range(length):
            observation = trajectory['observations'][i]
            action = trajectory['actions'][i]
            next_observation = trajectory['next_observations'][i]


            if is_new_trajectory:
                init_obs = trajectory['observations'][i]
                final_obs = trajectory['next_observations'][-1]
                # direction = self.get_direction(init_obs, final_obs)
                # print(direction)
                new_pos, aug_location = self._sample_pos()
                delta_pos = new_pos - init_obs[:2]

            augmented_obs = observation.copy()
            augmented_next_obs = next_observation.copy()
            augmented_action = action.copy()
            # augmented_reward = reward.copy()
            # augmented_done = done.copy()

            augmented_obs[:2] = observation[:2] + delta_pos
            augmented_next_obs[:2] = next_observation[:2] + delta_pos

            # rotate_alpha = np.random.uniform(0, 2 * np.pi)
            # print(direction)

            guide_theta = self.get_guide_theta(aug_location)

            delta = final_obs[:2] - init_obs[:2]
            observed_theta = np.arctan2(delta[1], delta[0])
            rotate_alpha = guide_theta - observed_theta

            M = np.array([
                [np.cos(-rotate_alpha), -np.sin(-rotate_alpha)],
                [np.sin(-rotate_alpha), np.cos(-rotate_alpha)]
            ])

            # zero center
            augmented_obs[:2] -= init_obs[:2]
            augmented_next_obs[:2] -= init_obs[:2]

            augmented_obs[:2] = M.dot(augmented_obs[:2]).T
            augmented_next_obs[:2] = M.dot(augmented_next_obs[:2]).T

            augmented_obs[:2] += init_obs[:2]
            augmented_next_obs[:2] += init_obs[:2]

            # corner case (literally): check that the agent isn't inside a wall
            pos_is_valid = self._check_corners(augmented_obs[:2], aug_location)
            next_pos_is_valid = self._check_corners(augmented_next_obs[:2], aug_location)

            # if new positions are not valid, immediately sample a new position
            if not (pos_is_valid and next_pos_is_valid):
                is_new_trajectory = True
                i -= 1
                continue

            # mujoco stores quats as (qw, qx, qy, qz) internally but uses (qx, qy, qz, qw) in the observation
            sin = np.sin(rotate_alpha / 2)
            cos = np.cos(rotate_alpha / 2)
            quat_rotate_by = np.array([sin, 0, 0, cos])
            self._rotate_torso(augmented_obs, quat_rotate_by)
            self._rotate_torso(augmented_next_obs, quat_rotate_by)

            sin = np.sin(-rotate_alpha)
            cos = np.cos(-rotate_alpha)
            self._rotate_vel(augmented_obs, sin, cos)
            self._rotate_vel(augmented_next_obs, sin, cos)


            augmented_reward = self._reward(augmented_next_obs)
            augmented_done = augmented_reward > 0  # recompute reward *after* making all changes to observations.

            aug_location = self._get_location(augmented_obs)
            if aug_location is None:
                is_new_trajectory = True
                i -= 1
                continue

            is_new_trajectory = False
            augmented_trajectory['observations'].append(augmented_obs)
            augmented_trajectory['actions'].append(augmented_action)
            augmented_trajectory['rewards'].append(augmented_reward)
            augmented_trajectory['next_observations'].append(augmented_next_obs)
            augmented_trajectory['terminals'].append(augmented_done)

        return augmented_trajectory



class AntMazeTrajectoryRandomAugmentationFunction(AntMazeTrajectoryGuidedAugmentationFunction):

    def __init__(self, env, **kwargs):

        super().__init__(env, **kwargs)


    def augment_trajectory(self, trajectory: dict, direction):
        length = len(trajectory['observations'])

        augmented_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
        }
        is_new_trajectory = True
        for i in range(length):
            observation = trajectory['observations'][i]
            action = trajectory['actions'][i]
            next_observation = trajectory['next_observations'][i]

            if is_new_trajectory:
                idx = np.random.choice(len(self.valid_locations))
                location = np.array(self.valid_locations[idx]).astype(self.env.observation_space.dtype)
                boundary = self._get_valid_boundaries(*location)
                new_origin = self._sample_from_box(*boundary)
                delta_pos = new_origin[:2] - observation[:2]

            augmented_obs = observation.copy()
            augmented_next_obs = next_observation.copy()
            augmented_action = action.copy()
            # augmented_reward = reward.copy()
            # augmented_done = done.copy()

            augmented_obs[:2] = observation[:2] + delta_pos
            augmented_next_obs[:2] = next_observation[:2] + delta_pos
            augmented_reward = self._reward(augmented_next_obs)
            augmented_done = augmented_reward > 0  # recompute reward *after* making all changes to observations.

            aug_location = self._get_location(augmented_obs)
            if aug_location is None:
                is_new_trajectory = True
                i -= 1
                continue
            pos_is_valid = self._check_corners(augmented_obs[:2], aug_location)
            next_pos_is_valid = self._check_corners(augmented_next_obs[:2], aug_location)

            if not (pos_is_valid and next_pos_is_valid):
                is_new_trajectory = True
                i -= 1
                continue

            if self.is_valid_input(augmented_obs, augmented_next_obs):
                is_new_trajectory = False
                augmented_trajectory['observations'].append(augmented_obs)
                augmented_trajectory['actions'].append(augmented_action)
                augmented_trajectory['rewards'].append(augmented_reward)
                augmented_trajectory['next_observations'].append(augmented_next_obs)
                augmented_trajectory['terminals'].append(augmented_done)
            else:
                is_new_trajectory = True
                i -= 1
        return augmented_trajectory