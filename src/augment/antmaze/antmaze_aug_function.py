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
        self.effective_wall_width = 1.5
        self.maze_scale = 4
        self.thetas = np.array([0, np.pi/2, np.pi, np.pi*3/2])
        self.rotation_matrices = []
        for theta in self.thetas:
            M = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            self.rotation_matrices.append(M)
        # self.target = np.array([1.00675, 8.845378])
        # self.target = np.array([21.23277744269721, 20.98104580473052])
        l = len(self.env.maze_arr)
        if l == 5:
            self.target = np.array([1.00675, 8.845378])
            self.env.maze_arr = U_MAZE
        elif l == 8:
            self.target = np.array([21.23277744269721, 20.98104580473052])
            self.env.maze_arr = MEDIUM_MAZE

        else:
            self.target = np.array([33.334047004876766, 24.540936989331545])
            self.env.maze_arr = LARGE_MAZE

        self.wall_locations = []
        self.valid_locations = []
        width, height = self.env.maze_arr.shape
        for w in range(width):
            for h in range(height):
                location_type = self.env.maze_arr[w, h]
                box_location = np.array((w, h))
                if location_type in ['1']:
                    self.wall_locations.append(box_location)
                elif location_type in ['0', GOAL, RESET]:
                    self.valid_locations.append(box_location)
        self.valid_locations = np.array(self.valid_locations)

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


    def quat_mul(self, quat0, quat1):
        assert quat0.shape == quat1.shape
        assert quat0.shape[-1] == 4

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

        assert quat.shape == quat0.shape
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

    # def _rotate_vel(self, obs, alpha):
    #     x = obs[2+13].copy()
    #     y = obs[2+14].copy()
    #     norm = np.sqrt(x**2 + y**2)
    #
    #     obs[2+13] = norm * np.cos(alpha)
    #     obs[2+14] = norm * np.sin(alpha)

    def _xy_to_rowcol(self, xy):
        size_scaling = self.maze_scale
        xy = (max(xy[0], 1e-4), max(xy[1], 1e-4))
        return (int(np.round(1 + (xy[0]) / size_scaling)),
                int(np.round(1 + (xy[1]) / size_scaling)))

    def _rowcol_to_xy(self, rowcol, add_random_noise=False):
        row, col = rowcol
        x = col * self.maze_scale - 0
        y = row * self.maze_scale - 0
        if add_random_noise:
            x = x + np.random.uniform(low=0, high=self.maze_scale * 0.25)
            y = y + np.random.uniform(low=0, high=self.maze_scale * 0.25)
        return (x, y)

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

    # def _check_valid_input(self, obs):


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
        aug_reward = self._reward(aug_next_obs)
        aug_done = done
        aug_obs[:2] += 0.5
        aug_next_obs[:2] += 0.5

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done

class AntMazeGuidedAugmentationFunction(AntMazeAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.theta_to_quat = np.array([
            [1, 0, 0, 0],
            [0.707, 0, 0, 0.707],
            [1, 0, 0, 0],
            [0.707, 0, 0, -0.707],
        ])
        if self.env.maze_arr.shape[0] == 5:
            self.guide_thetas = {
                (1, 1): [0],
                (2, 1): [0],
                (3, 1): [np.pi/2],

                (3, 2): [np.pi/2],

                (1, 3): [0, np.pi/2, np.pi, np.pi*3/2],
                (2, 3): [np.pi],
                (3, 3): [np.pi]
            }
        elif self.env.maze_arr.shape[0] == 8:
            print('medium')
            self.guide_thetas = {
                (1, 1): [0],
                (2, 1): [np.pi/2],
                (3, 1): [],
                (4, 1): [],
                (5, 1): [np.pi/2],
                (6, 1): [np.pi],

                (1, 2): [0],
                (2, 2): [np.pi/2],
                (3, 2): [],
                (4, 2): [np.pi/2],
                (5, 2): [np.pi],
                (6, 2): [np.pi],

                (1, 3): [],
                (2, 3): [0],
                (3, 3): [0],
                (4, 3): [np.pi/2],
                (5, 3): [],
                (6, 3): [],

                (1, 4): [0],
                (2, 4): [np.pi*3/2],
                (3, 4): [],
                (4, 4): [0],
                (5, 4): [0],
                (6, 4): [np.pi/2],

                (1, 5): [np.pi/2],
                (2, 5): [],
                (3, 5): [0],
                (4, 5): [np.pi*3/2],
                (5, 5): [],
                (6, 5): [np.pi / 2],

                (1, 6): [0],
                (2, 6): [0],
                (3, 6): [np.pi*3/2],
                (4, 6): [],
                (5, 6): [0],
                (6, 6): [0, np.pi / 2, np.pi, np.pi * 3 / 2],
            }
        elif self.env.maze_arr.shape[0] == 12:
            self.guide_thetas = {
                (1, 1): [0],
                (2, 1): [0],
                (3, 1): [0],
                (4, 1): [0],
                (5, 1): [],
                (6, 1): [np.pi/2],
                (7, 1): [np.pi],
                (8, 1): [np.pi/2],
                (9, 1): [0],
                (10, 1): [np.pi/2],

                (1, 2): [np.pi/2],
                (2, 2): [],
                (3, 2): [],
                (4, 2): [np.pi/2],
                (5, 2): [],
                (6, 2): [np.pi/2],
                (7, 2): [],
                (8, 2): [np.pi/2],
                (9, 2): [0],
                (10, 2): [np.pi/2],

                (1, 3): [0],
                (2, 3): [0],
                (3, 3): [0],
                (4, 3): [0],
                (5, 3): [0],
                (6, 3): [np.pi / 2],
                (7, 3): [],
                (8, 3): [0],
                (9, 3): [0],
                (10, 3): [np.pi / 2],

                (1, 4): [np.pi*3 / 2],
                (2, 4): [],
                (3, 4): [],
                (4, 4): [],
                (5, 4): [],
                (6, 4): [np.pi / 2],
                (7, 4): [],
                (8, 4): [],
                (9, 4): [],
                (10, 4): [np.pi / 2],

                (1, 5): [np.pi*3 / 2],
                (2, 5): [np.pi],
                (3, 5): [],
                (4, 5): [np.pi / 2],
                (5, 5): [],
                (6, 5): [0],
                (7, 5): [0],
                (8, 5): [np.pi / 2],
                (9, 5): [np.pi],
                (10, 5): [np.pi],

                (1, 6): [],
                (2, 6): [np.pi*3 / 2],
                (3, 6): [],
                (4, 6): [np.pi / 2],
                (5, 6): [],
                (6, 6): [np.pi*3 / 2],
                (7, 6): [],
                (8, 6): [np.pi / 2],
                (9, 6): [],
                (10, 6): [],

                (1, 7): [0],
                (2, 7): [np.pi*3 / 2],
                (3, 7): [],
                (4, 7): [0],
                (5, 7): [0],
                (6, 7): [np.pi*3 / 2],
                (7, 7): [],
                (8, 7): [0],
                (9, 7): [0, np.pi/2, np.pi, np.pi*3/2],
                (10, 7): [np.pi],
            }

    def _sample_theta(self, obs, next_obs, new_pos, new_location, **kwargs):

        delta_obs = next_obs[:2] - obs[:2]
        theta = np.arctan2(delta_obs[1], delta_obs[0])

        guide_thetas = self.guide_thetas[(int(new_location[0]), int(new_location[1]))]
        guide_theta = np.random.choice(guide_thetas)

        aug_theta = -(guide_theta - theta) #+ np.random.uniform(low=-np.pi/6, high=np.pi/6)

        return aug_theta

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
            rotate_alpha = self._sample_theta(obs, next_obs, new_pos, aug_location)

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
        aug_reward = self._reward(aug_next_obs)
        aug_done = done
        aug_obs[:2] += 0.5
        aug_next_obs[:2] += 0.5

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done



