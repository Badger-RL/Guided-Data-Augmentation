from collections import defaultdict

import numpy as np

from augment.antmaze.antmaze_aug_function import AntMazeAugmentationFunction


class AntMazeGuidedTrajAugmentationFunction(AntMazeAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.num_aug = 0

        # precompute quaternion conversions
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
                (3, 1): [np.pi / 2],

                (3, 2): [np.pi / 2],

                (1, 3): [0, np.pi / 2, np.pi, np.pi * 3 / 2],
                (2, 3): [np.pi],
                (3, 3): [np.pi]
            }
        elif self.env.maze_arr.shape[0] == 8:
            print('medium')

            UP = 0
            DOWN = 1
            LEFT = 2
            RIGHT = 3

            self.cell_to_guide = {
                (1, 1): LEFT,
                (2, 1): UP,
                (3, 1): None,
                (4, 1): None,
                (5, 1): UP,
                (6, 1): LEFT,

                (1, 2): RIGHT,
                (2, 2): UP,
                (3, 2): None,
                (4, 2): UP,
                (5, 2): LEFT,
                (6, 2): LEFT,

                (1, 3): None,
                (2, 3): RIGHT,
                (3, 3): RIGHT,
                (4, 3): UP,
                (5, 3): None,
                (6, 3): None,

                (1, 4): RIGHT,
                (2, 4): DOWN,
                (3, 4): None,
                (4, 4): RIGHT,
                (5, 4): RIGHT,
                (6, 4): UP,

                (1, 5): UP,
                (2, 5): None,
                (3, 5): RIGHT,
                (4, 5): DOWN,
                (5, 5): None,
                (6, 5): UP,

                (1, 6): RIGHT,
                (2, 6): RIGHT,
                (3, 6): DOWN,
                (4, 6): None,
                (5, 6): RIGHT,
                (6, 6): None,
            }

            self.guide_to_cell = defaultdict(lambda: [])
            for key, val in self.cell_to_guide.items():
                self.guide_to_cell[val].append(key)

    def _is_in_box(self, xlo, xhi, ylo, yhi, x, y):
        if (x > xlo and x < xhi) and (y > ylo and y < yhi):
            return True
        else:
            return False

    def _sample_umaze(self, obs, last_obs):
        x, y = obs[0], obs[1]

        # bottom
        if x > 0 and x < 8.5 and y > 0 and y < 0.5:
            new_pos = np.random.uniform(
                low=np.array([0, 0]),
                high=np.array([8.5, 0.5])
            )

        # right side
        elif x > 8.5 and x < 9 and y > 0 and y < 8:
            new_pos = np.random.uniform(
                low=np.array([8.5, 0]),
                high=np.array([9, 8])
            )
        elif x > 2 and x < 9 and y > 8 and y < 8.5:
            new_pos = np.random.uniform(
                low=np.array([0, 8]),
                high=np.array([9, 8.5])
            )
        else:
            new_pos = None

        return new_pos

    def _sample_medium(self, obs, last_obs):

        location = self._xy_to_rowcol(obs[:2])

        guide = self.cell_to_guide[location]
        if guide is None: return None
        possible_new_locations = self.guide_to_cell[guide]
        new_location = possible_new_locations[np.random.randint(len(possible_new_locations))]
        # print(new_location)
        if new_location == (5,6):
            stop = 0
        while True:
            new_pos = self._sample_from_box(*self._get_valid_boundaries(new_location[0], new_location[1]))
            new_pos_is_valid = self._check_corners(new_pos, new_location) or True
            if new_pos_is_valid: break

        return new_pos + 0.5

    #
    # def _sample_medium(self, obs, last_obs):
    #     x, y = obs[0], obs[1]
    #
    #     # bottom
    #     if x > 0 and x < 6 and y > 0 and y < 0.5:
    #         new_pos = np.random.uniform(
    #             low=np.array([0, 0]),
    #             high=np.array([8.5, 0.5])
    #         )
    #
    #     # right side
    #     elif x > 8.5 and x < 9 and y > 0 and y < 8:
    #         new_pos = np.random.uniform(
    #             low=np.array([8.5, 0]),
    #             high=np.array([9, 8])
    #         )
    #     elif x > 2 and x < 9 and y > 8 and y < 8.5:
    #         new_pos = np.random.uniform(
    #             low=np.array([0, 8]),
    #             high=np.array([9, 8.5])
    #         )
    #     else:
    #         new_pos = None
    #
    #     return new_pos


    def _is_valid_umaze(self, obs):
        x, y = obs[:,0], obs[:,1]

        # bottom
        mask1 = (x > 0) & (x < 9) & (y > -1) & (y < 1)
        # right side
        mask2 = (x > 7) & (x < 9) & (y > 0) & (y < 8)
        # top
        mask3 = (x > 0) & (x < 9) & (y > 7) & (y < 9)

        return mask1 | mask2 | mask3

    def _is_valid_medium(self, obs):
        x, y = obs[:,0], obs[:,1]

        mask = np.zeros_like(x).astype(bool)
        for i in range(len(x)):
            if x[i] < 0 or x[i] > 24:
                is_valid = False
            elif y[i] < 0 or y[i] > 24:
                is_valid = False
            else:
                is_valid = self._check_nearby_walls(obs[i, :2])
            mask[i] = is_valid

        return mask

    def _check_nearby_walls(self, xy):
        x, y = xy[0], xy[1]
        location = self._xy_to_rowcol(xy)
        is_valid_position = True

        w, h = int(location[0]-1), int(location[1])
        try:
            for loc in [(w + 1, h), (w - 1, h), (w, h + 1), (w, h - 1)]:
                if self.env.maze_arr[loc[0], loc[1]] == '1':
                    loc = np.array(loc)
                    if self._is_in_wall(loc, x, y):
                        is_valid_position = False
                        break
        except:
            is_valid_position = False

        return is_valid_position

    def _is_in_wall(self, box_location, x, y):
        xlo, ylo = box_location*4 - self.agent_offset - 2 - self.effective_wall_width
        xhi, yhi = box_location*4 - self.agent_offset + 2 + self.effective_wall_width

        if (x > xlo and y > ylo) and (x < xhi and y < yhi):
            return True
        else:
            return False

    def _is_done(self, aug_next_obs, aug_reward):
        aug_done = ~((aug_next_obs[:,2] >= 0.2) & (aug_next_obs[:, 2] <= 1.0)) | (aug_reward > 0)
        return aug_done

    def _reward(self, next_obs):
        # Rewar dshould intuitively be computed using next_obs, but D4RL uses the current obs (D4RL bug)
        if self.env.reward_type == 'sparse':
            # print(np.linalg.norm(next_obs[0:2] - (self.target )))
            reward = (np.linalg.norm(next_obs[:,0:2] - self.target, axis=-1) <= 0.5).astype(float)
        elif self.env.reward_type == 'dense':
            reward = np.exp(-np.linalg.norm(next_obs[:,0:2] - self.target), axis=-1)
        else:
            raise ValueError('Unknown reward type %s' % self.env.reward_type)

        return reward

    # def augment(self,
    #             obs: np.ndarray,
    #             action: np.ndarray,
    #             next_obs: np.ndarray,
    #             reward: np.ndarray,
    #             done: np.ndarray,
    #             **kwargs, ):
    #
    #     aug_obs = obs.copy()
    #     aug_next_obs = next_obs.copy()
    #
    #     new_pos = self._sample_umaze(obs[0])
    #
    #     if new_pos is None:
    #         return None, None, None, None, None
    #
    #     delta_pos = next_obs[:,:2] - obs[:,:2]
    #     delta_new_pos = new_pos - obs[0,:2]
    #
    #     aug_obs[:,:2] += delta_new_pos
    #     aug_next_obs[:,:2] = aug_obs[:,:2] + delta_pos
    #
    #     aug_action = action.copy()
    #     aug_reward = self._reward(aug_next_obs)
    #     aug_done = self._is_done(aug_next_obs, aug_reward)
    #     self.num_aug = 0
    #
    #     mask = self._is_valid_umaze(aug_obs)
    #
    #     return aug_obs[mask], aug_action[mask], aug_reward[mask], aug_next_obs[mask], aug_done[mask]


    def _get_guided_theta_umaze(self, new_pos):

        if new_pos[0] < 7.5 and new_pos[1] > -2 and new_pos[1] < 2:
            guide_theta = 0
        elif new_pos[0] > 7.5 and new_pos[0] < 9 and new_pos[1] < 7.5:
            guide_theta = np.pi / 2
        else:
            guide_theta = np.pi

        return guide_theta

    def _sample_theta(self, obs, next_obs, new_pos, new_location, **kwargs):
        guide_theta = self._get_guided_theta_umaze(new_pos)
        if guide_theta == 0:
            stop = 0
        # guide_theta = np.pi*3/2
        delta_obs = next_obs[:2] - obs[:2]
        theta = 2*np.arctan2(delta_obs[1], delta_obs[0])
        aug_theta = -(guide_theta - theta) #+ np.random.uniform(low=-np.pi/6, high=np.pi/6)

        return aug_theta

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        # if not self.is_valid_input(obs, next_obs):
        #     return None, None, None, None, None
        # displacement = next_obs[-1,:2] - obs[0,:2]
        # if np.linalg.norm(displacement) < 1:
        #     return None, None, None, None, None
        #
        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()
        #
        # new_pos = self._sample_umaze(obs[0], aug_next_obs[-1, :2])
        new_pos = self._sample_medium(obs[0], aug_next_obs[-1, :2])

        # new_pos, new_location = self._sample_pos()

        if new_pos is None:
            return None, None, None, None, None

        offset = new_pos - obs[0,:2]
        aug_obs[:,:2] += offset
        aug_next_obs[:,:2] += offset
        # aug_location = self._xy_to_rowcol(aug_obs[0,:2])
        #
        # rotate_alpha = self._sample_theta(aug_obs[0,:2], aug_next_obs[-1,:2], new_pos, aug_location)
        #
        # M = np.array([
        #     [np.cos(-rotate_alpha), -np.sin(-rotate_alpha)],
        #     [np.sin(-rotate_alpha), np.cos(-rotate_alpha)]
        # ])
        #
        # # translate initial position to 0,0 before rotation
        # origin = aug_obs[0, :2].copy()
        # aug_obs[:,:2] -= origin
        # aug_next_obs[:,:2] -= origin
        #
        # aug_obs[:,:2] = M.dot(aug_obs[:,:2].T).T
        # aug_next_obs[:,:2] = M.dot(aug_next_obs[:,:2].T).T
        #
        # # translate back to initial position after rotation
        # aug_obs[:,:2] += origin
        # aug_next_obs[:,:2] += origin
        # # delta_pos = next_obs[:,:2] - obs[:,:2]
        # # rotated_delta_obs = M.dot(delta_pos[:,:2].T).T
        # # aug_next_obs[:,:2] = aug_obs[:,:2] + rotated_delta_obs
        #
        #
        #
        # # mujoco stores quats as (qw, qx, qy, qz) internally but uses (qx, qy, qz, qw) in the observation
        # for i in range(len(obs)):
        #     sin = np.sin(rotate_alpha / 2)
        #     cos = np.cos(rotate_alpha / 2)
        #     quat_rotate_by = np.array([sin, 0, 0, cos])
        #     self._rotate_torso(aug_obs[i], quat_rotate_by)
        #     self._rotate_torso(aug_next_obs[i], quat_rotate_by)
        #
        #     sin = np.sin(-rotate_alpha)
        #     cos = np.cos(-rotate_alpha)
        #     self._rotate_vel(aug_obs[i], sin, cos)
        #     self._rotate_vel(aug_next_obs[i], sin, cos)

        aug_action = action.copy()
        # aug_reward = reward
        # aug_done = done
        aug_reward = self._reward(aug_next_obs)
        aug_done = self._is_done(aug_next_obs, aug_reward)
        # mask = self._is_valid_umaze(aug_obs)
        mask = self._is_valid_medium(aug_obs)

        # mask = np.ones_like(reward).astype(bool)

        return aug_obs[mask], aug_action[mask], aug_reward[mask], aug_next_obs[mask], aug_done[mask]
