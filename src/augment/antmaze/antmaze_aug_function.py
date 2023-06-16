import numpy as np
from src.augment.augmentation_function_base import AugmentationFunctionBase
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

class AntMazeAugmentationFunction(AugmentationFunctionBase):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.agent_offset = 4 # the agent and target coordinate systems are different for some reason.
        self.effective_wall_width = 1
        self.maze_scale = 4
        self.thetas = np.array([0, np.pi/2, np.pi, np.pi*3/2])
        self.rotation_matrices = []
        for theta in self.thetas:
            M = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            self.rotation_matrices.append(M)
        self.target = np.array([1.00675, 8.845378])
        self.wall_locations = []
        self.valid_locations = []
        self.env.maze_arr = LARGE_MAZE
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

    # def _sample_pos(self, ):
    #     prob = (1.0 - self.env.maze_arr) / np.sum(1.0 - self.env.maze_arr)
    #     prob_row = np.sum(prob, 1)
    #     row_sample = np.random.choice(np.arange(self.env.maze_arr.shape[0]), p=prob_row)
    #     col_sample = np.random.choice(np.arange(self.env.maze_arr.shape[1]),
    #                                   p=prob[row_sample] * 1.0 / prob_row[row_sample])
    #     reset_location = self._rowcol_to_xy((row_sample, col_sample))
    #
    #     # Add some random noise
    #     random_x = np.random.uniform(low=0, high=0.5) * 0.5 * self.maze_scale
    #     random_y = np.random.uniform(low=0, high=0.5) * 0.5 * self.maze_scale
    #
    #     return (max(reset_location[0] + random_x, 0), max(reset_location[1] + random_y, 0))

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
        xlo, ylo = box_location*4 - self.agent_offset - 2.5
        xhi, yhi = box_location*4 - self.agent_offset + 2.5

        if (x > xlo and y > ylo) and (x < xhi and y < yhi):
            return True
        else:
            return False


    def _xy_to_rowcol(self, xy):
        xy = (max(xy[0], 1e-4), max(xy[1], 1e-4))
        return (int(1 + (xy[1]) / self.maze_scale),
                int(1 + (xy[0]) / self.maze_scale))

    # def _is_valid_position(self, xy):
    #     row, col = self._xy_to_rowcol(xy)
    #     x, y = xy[1], xy[0]
    #     is_valid_position = True
    #     for r in range(row - 1, row + 2):
    #         for c in range(col- 1, col + 2):
    #             if self.env.maze_arr[r,c] == 1:
    #                 if self._is_in_wall(np.array([r,c]), x, y):
    #                     is_valid_position = False
    #                     break

    def _is_valid_position(self, xy):
        x, y = xy[0], xy[1]

        is_valid_position = True
        for box_location in self.wall_locations:
            if self._is_in_wall(box_location, x, y):
                is_valid_position = False
                break

        return is_valid_position

    def _sample_pos(self, n=1):
        idx = np.random.choice(len(self.valid_locations))
        location = np.array(self.valid_locations[idx]).astype(self.env.observation_space.dtype)

        boundaries = self._get_valid_boundaries(*location)
        return self._sample_from_box(*boundaries)

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
        quat_curr = obs[2+1:2+4 + 1]
        quat_result = self.quat_mul(quat0=quat_curr, quat1=quat_rotate_by)
        # quat already normalized
        obs[2+1:2+4 + 1] = quat_result

    def _rotate_vel(self, obs, sin, cos):
        x = obs[2+13].copy()
        y = obs[2+14].copy()
        obs[2+13] = x * cos - y * sin
        obs[2+14] = x * sin + y * cos

    def _xy_to_rowcol(self, xy):
        size_scaling = self.maze_scale
        xy = (max(xy[0], 1e-4), max(xy[1], 1e-4))
        return (int(1 + (xy[1]) / size_scaling),
                int(1 + (xy[0]) / size_scaling))

    def _rowcol_to_xy(self, rowcol, add_random_noise=False):
        row, col = rowcol
        x = col * self.maze_scale - 0
        y = row * self.maze_scale - 0
        if add_random_noise:
            x = x + np.random.uniform(low=0, high=self.maze_scale * 0.25)
            y = y + np.random.uniform(low=0, high=self.maze_scale * 0.25)
        return (x, y)

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        is_valid = False
        while not is_valid:
            aug_obs = obs.copy()
            aug_next_obs = next_obs.copy()

            alpha = np.random.uniform(low=0, high=2*np.pi)
            # alpha = np.pi/2
            sin = np.sin(alpha / 2)
            cos = np.cos(alpha / 2)

            # mujoco stores quats as (qw, qx, qy, qz)
            quat_rotate_by = np.array([sin, 0, 0, cos])

            self._rotate_torso(aug_obs, quat_rotate_by)
            self._rotate_torso(aug_next_obs, quat_rotate_by)

            # Not sure why we need -alpha here...
            sin = np.sin(-alpha)
            cos = np.cos(-alpha)
            self._rotate_vel(aug_obs, sin, cos)
            self._rotate_vel(aug_next_obs, sin, cos)

            M = np.array([
                [np.cos(-alpha), -np.sin(-alpha)],
                [np.sin(-alpha), np.cos(-alpha)]
            ])

            aug_obs[:2] = self._sample_pos()
            delta_pos = next_obs[:2] - obs[:2]
            rotated_delta_obs = M.dot(delta_pos[:2]).T
            aug_next_obs[:2] = aug_obs[:2] + rotated_delta_obs

            # initial pos is always valid, so we only need to check that the next pos isn't inside a wall
            next_pos = aug_next_obs[:2]
            is_valid = self._is_valid_position(next_pos)

        aug_action = action.copy()
        aug_reward = self._reward(aug_next_obs)
        aug_done = done
        # return obs, action, reward, next_obs, done

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done

class AntMazeGuidedAugmentationFunction(AntMazeAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

        if self.env.maze_arr.shape[0] == 5:
            self.guide_thetas = {
                #left
                (1, 1): [0],
                (1, 2): [0],
                (1, 3): [np.pi/2],
                # top
                (2, 3): [np.pi/2],
                (3, 3): [np.pi],
                # right
                (3, 1): [0, np.pi / 2, np.pi, np.pi * 3 / 2],
                (3, 2): [np.pi],
            }
        elif self.env.maze_arr.shape[0] == 8:
            self.guide_thetas = {
                (1, 1): [np.pi/2],
                (2, 1): [0],
                # (1, 1): [],
                (4, 1): [0],
                (5, 1): [np.pi*3/2],
                (6, 1): [0],

                (1, 2): [np.pi/2],
                (2, 2): [np.pi/2],
                (3, 2): [0],
                (4, 2): [np.pi + np.pi/2],
                # (5, 2): [np.pi],
                (6, 2): [0],

                # (1, 3): [0],
                # (2, 3): [0],
                (3, 3): [0],
                # (4, 3): [np.pi],
                (5, 3): [0],
                (6, 3): [np.pi + np.pi/2],

                (1, 4): [0 + np.pi/2],
                (2, 4): [0 + np.pi/2],
                (3, 4): [0 + np.pi/2],
                (4, 4): [0],
                (5, 4): [np.pi*3/2],
                # (6, 4): [np.pi],

                (1, 5): [0 + np.pi/2],
                (2, 5): [np.pi],
                # (3, 5): [np.pi / 2],
                (4, 5): [0],
                # (5, 5): [np.pi],
                (6, 5): [0],

                (1, 6): [np.pi/2],
                (2, 6): [np.pi],
                # (3, 6): [np.pi / 2],
                (4, 6): [0 + np.pi/2],
                (5, 6): [0 + np.pi/2],
                (6, 6): [0, np.pi / 2, np.pi, np.pi * 3 / 2],
            }
        elif self.env.maze_arr.shape[0] == 12:
            self.guide_thetas = {
                (1, 1): [0],
                (2, 1): [np.pi/2],
                (3, 1): [0],
                (4, 1): [np.pi*3/2],
                (5, 1): [np.pi*3/2],
                # (6, 1): [np.pi / 2],
                (7, 1): [0],

                (1, 2): [0],
                # (2, 2): [0],
                (3, 2): [0],
                # (4, 2): [np.pi],
                (5, 2): [np.pi*3/2],
                (6, 2): [np.pi*3/2],
                (7, 2): [np.pi*3/2],

                (1, 3): [0],
                # (2, 3): [0],
                (3, 3): [0],
                # (4, 3): [np.pi],
                # (5, 3): [np.pi / 2],
                # (6, 3): [np.pi],
                # (7, 3): [np.pi],

                (1, 4): [np.pi/2],
                (2, 4): [np.pi/2],
                (3, 4): [0],
                # (4, 4): [np.pi / 2],
                (5, 4): [np.pi/2],
                (6, 4): [np.pi/2],
                (7, 4): [0],

                # (1, 5): [0],
                # (2, 5): [np.pi * 3 / 2],
                (3, 5): [0],
                # (4, 5): [np.pi / 2],
                # (5, 5): [np.pi],
                # (6, 5): [np.pi / 2],
                (7, 5): [0],

                (1, 6): [np.pi/2],
                (2, 6): [np.pi/2],
                (3, 6): [np.pi/2],
                (4, 6): [np.pi/2],
                (5, 6): [0],
                (6, 6): [np.pi*3/2],
                (7, 6): [np.pi*3/2],

                (1, 7): [np.pi],
                # (2, 7): [0],
                # (3, 7): [0],
                # (4, 7): [0],
                (5, 7): [0],
                # (6, 7): [np.pi],
                # (7, 7): [np.pi],

                (1, 8): [np.pi/2],
                (2, 8): [np.pi/2],
                (3, 8): [0],
                # (4, 8): [0],
                (5, 8): [np.pi/2],
                (6, 8): [np.pi/2],
                (7, 8): [0],

                (1, 9): [np.pi/2],
                # (2, 9): [0],
                (3, 9): [0],
                # (4, 9): [0],
                (5, 9): [np.pi],
                # (6, 9): [np.pi],
                (7, 9): [0, np.pi / 2, np.pi, np.pi * 3 / 2],

                (1, 10): [np.pi/2],
                (2, 10): [np.pi/2],
                (3, 10): [np.pi/2],
                (4, 10): [np.pi/2],
                (5, 10): [np.pi],
                # (6, 10): [0],
                (7, 10): [np.pi],
            }

    def _sample_theta(self, obs, next_obs, new_pos, **kwargs):

        x, y = new_pos[0], new_pos[1]
        delta_obs = next_obs - obs
        theta = np.arctan2(delta_obs[1], delta_obs[0])

        location = (int(np.round((y+self.agent_offset)/4)), int(np.round((x+self.agent_offset)/4)))

        # print(obs[:2], location)
        guide_thetas = self.guide_thetas[location]
        guide_theta = np.random.choice(guide_thetas)

        aug_theta = guide_theta - theta + np.random.uniform(low=-np.pi/6, high=np.pi/6)
        return -aug_theta

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        is_valid = False
        while not is_valid:
            aug_obs = obs.copy()
            aug_next_obs = next_obs.copy()

            new_pos = self._sample_pos()

            # alpha = np.random.uniform(low=0, high=2*np.pi)
            alpha = self._sample_theta(obs, next_obs, new_pos)
            # alpha = np.pi/2
            sin = np.sin(alpha / 2)
            cos = np.cos(alpha / 2)

            # mujoco stores quats as (qw, qx, qy, qz)
            quat_rotate_by = np.array([sin, 0, 0, cos])

            self._rotate_torso(aug_obs, quat_rotate_by)
            self._rotate_torso(aug_next_obs, quat_rotate_by)

            # Not sure why we need -alpha here...
            sin = np.sin(-alpha)
            cos = np.cos(-alpha)
            self._rotate_vel(aug_obs, sin, cos)
            self._rotate_vel(aug_next_obs, sin, cos)

            M = np.array([
                [np.cos(-alpha), -np.sin(-alpha)],
                [np.sin(-alpha), np.cos(-alpha)]
            ])

            aug_obs[:2] = new_pos
            delta_pos = next_obs[:2] - obs[:2]
            rotated_delta_obs = M.dot(delta_pos[:2]).T
            aug_next_obs[:2] = aug_obs[:2] + rotated_delta_obs

            # initial pos is always valid, so we only need to check that the next pos isn't inside a wall
            next_pos = aug_next_obs[:2]
            is_valid = True
            is_valid = self._is_valid_position(next_pos)

        aug_action = action.copy()
        aug_reward = self._reward(aug_next_obs)
        aug_done = done
        # return obs, action, reward, next_obs, done

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done




