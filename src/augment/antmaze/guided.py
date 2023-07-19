from collections import defaultdict

import numpy as np

from augment.antmaze.antmaze_aug_function import AntMazeAugmentationFunction


class AntMazeGuidedAugmentationFunction(AntMazeAugmentationFunction):
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
        elif self.env.maze_arr.shape[0] == 12:
            self.guide_thetas = {
                (1, 1): [0],
                (2, 1): [0],
                (3, 1): [0],
                (4, 1): [0],
                (5, 1): [],
                (6, 1): [np.pi / 2],
                (7, 1): [np.pi],
                (8, 1): [np.pi / 2],
                (9, 1): [0],
                (10, 1): [np.pi / 2],

                (1, 2): [np.pi / 2],
                (2, 2): [],
                (3, 2): [],
                (4, 2): [np.pi / 2],
                (5, 2): [],
                (6, 2): [np.pi / 2],
                (7, 2): [],
                (8, 2): [np.pi / 2],
                (9, 2): [0],
                (10, 2): [np.pi / 2],

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

                (1, 4): [np.pi * 3 / 2],
                (2, 4): [],
                (3, 4): [],
                (4, 4): [],
                (5, 4): [],
                (6, 4): [np.pi / 2],
                (7, 4): [],
                (8, 4): [],
                (9, 4): [],
                (10, 4): [np.pi / 2],

                (1, 5): [np.pi * 3 / 2],
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
                (2, 6): [np.pi * 3 / 2],
                (3, 6): [],
                (4, 6): [np.pi / 2],
                (5, 6): [],
                (6, 6): [np.pi * 3 / 2],
                (7, 6): [],
                (8, 6): [np.pi / 2],
                (9, 6): [],
                (10, 6): [],

                (1, 7): [0],
                (2, 7): [np.pi * 3 / 2],
                (3, 7): [],
                (4, 7): [0],
                (5, 7): [0],
                (6, 7): [np.pi * 3 / 2],
                (7, 7): [],
                (8, 7): [0],
                (9, 7): [0, np.pi / 2, np.pi, np.pi * 3 / 2],
                (10, 7): [np.pi],
            }

    def _get_guided_theta_umaze(self, new_pos):

        if new_pos[0] < 7.5 and new_pos[1] > -2 and new_pos[1] < 2:
            guide_theta = 0
        elif new_pos[0] > 7.5 and new_pos[0] < 9 and new_pos[1] < 7.5:
            guide_theta = np.pi / 2
        else:
            guide_theta = np.pi

        return guide_theta

    def _get_guided_theta_medium(self, new_pos):

        if new_pos[0] < 7.5 and new_pos[1] > -2 and new_pos[1] < 2:
            guide_theta = 0
        elif new_pos[0] > 7.5 and new_pos[0] < 9 and new_pos[1] < 7.5:
            guide_theta = np.pi / 2
        else:
            guide_theta = np.pi

        return guide_theta

    # def _sample_theta(self, obs, next_obs, new_pos, new_location, **kwargs):
    #     guide_theta = self._get_guided_theta_umaze(new_pos)
    #
    #     delta_obs = next_obs[:2] - obs[:2]
    #     theta = np.arctan2(delta_obs[1], delta_obs[0])
    #     aug_theta = -(guide_theta - theta) #+ np.random.uniform(low=-np.pi/6, high=np.pi/6)
    #
    #     return aug_theta

    # def augment(self,
    #             obs: np.ndarray,
    #             action: np.ndarray,
    #             next_obs: np.ndarray,
    #             reward: np.ndarray,
    #             done: np.ndarray,
    #             **kwargs,):
    #
    #     # if not self.is_valid_input(obs, next_obs):
    #     #     return None, None, None, None, None
    #
    #     aug_obs = obs.copy()
    #     aug_next_obs = next_obs.copy()
    #     while True:
    #         new_pos, aug_location = self._sample_pos()
    #         # if obs[0] < 2.5 and obs[1] > 6:
    #         #     continue
    #         # new_pos = aug_obs[:2]
    #         # aug_location = self._xy_to_rowcol(aug_obs[:2])
    #         # print(new_pos,aug_location)
    #         rotate_alpha = self._sample_theta(obs, next_obs, new_pos, aug_location)
    #
    #         M = np.array([
    #             [np.cos(-rotate_alpha), -np.sin(-rotate_alpha)],
    #             [np.sin(-rotate_alpha), np.cos(-rotate_alpha)]
    #         ])
    #
    #         aug_obs[:2] = new_pos
    #         delta_pos = next_obs[:2] - obs[:2]
    #         rotated_delta_obs = M.dot(delta_pos[:2]).T
    #         aug_next_obs[:2] = aug_obs[:2] + rotated_delta_obs
    #
    #         # corner case (literally): check that the agent isn't inside a wall
    #         pos_is_valid = self._check_corners(aug_obs[:2], aug_location)
    #         next_pos_is_valid = self._check_corners(aug_next_obs[:2], aug_location)
    #
    #         # if new positions are not valid, immediately sample a new position
    #         if not (pos_is_valid and next_pos_is_valid):
    #             continue
    #
    #         # mujoco stores quats as (qw, qx, qy, qz) internally but uses (qx, qy, qz, qw) in the observation
    #         sin = np.sin(rotate_alpha / 2)
    #         cos = np.cos(rotate_alpha / 2)
    #         quat_rotate_by = np.array([sin, 0, 0, cos])
    #         self._rotate_torso(aug_obs, quat_rotate_by)
    #         self._rotate_torso(aug_next_obs, quat_rotate_by)
    #
    #         sin = np.sin(-rotate_alpha)
    #         cos = np.cos(-rotate_alpha)
    #         self._rotate_vel(aug_obs, sin, cos)
    #         self._rotate_vel(aug_next_obs, sin, cos)
    #
    #         break
    #
    #     aug_obs[:2] += 0.5
    #     aug_next_obs[:2] += 0.5
    #     aug_action = action.copy()
    #     aug_reward = self._reward(aug_next_obs)
    #     aug_done = aug_reward > 0
    #     # aug_obs[:2] += np.random.uniform(-0.1,0.1, size=(2,))
    #     # aug_next_obs[:2] += np.random.uniform(-0.1,0.1, size=(2,))
    #
    #     return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done

    def _is_in_box(self, xlo, xhi, ylo, yhi, x, y):
        if (x > xlo and x < xhi) and (y > ylo and y < yhi):
            return True
        else:
            return False

    #
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
        guide_theta = 0

        delta_obs = next_obs - obs
        theta = 2 * np.arctan2(delta_obs[1], delta_obs[0])
        # theta = 2 * np.arctan2(delta_obs[3], delta_obs[6])

        aug_theta = -(guide_theta - theta)  # + np.random.uniform(low=-np.pi/6, high=np.pi/6)

        return aug_theta

    def _sample_umaze(self, obs):
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
        elif x > 0 and x < 9 and y > 8 and y < 8.5:
            new_pos = np.random.uniform(
                low=np.array([0, 8]),
                high=np.array([9, 8.5])
            )
        else:
            new_pos = None

        return new_pos

    def _sample_medium(self, obs):

        location = self._xy_to_rowcol(obs[:2])

        guide = self.cell_to_guide[location]
        if guide is None: return None
        possible_new_locations = self.guide_to_cell[guide]
        new_location = possible_new_locations[np.random.randint(len(possible_new_locations))]

        while True:
            new_pos = self._sample_from_box(*self._get_valid_boundaries(*new_location))
            new_pos_is_valid = self._check_corners(new_pos, new_location)
            if new_pos_is_valid: break

        return new_pos + 0.5

    def _is_done(self, aug_next_obs, aug_reward):
        if not (aug_next_obs[2] >= 0.2 and aug_next_obs[2] <= 1.0):
            aug_done = True
        elif aug_reward > 0:
            aug_done = True
        else:
            aug_done = False

        return aug_done

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs, ):

        # if not self.is_valid_input(obs, next_obs):
        #     return None, None, None, None, None

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()

        new_pos = self._sample_umaze(obs)
        # new_pos = self._sample_medium(obs)

        if new_pos is None:
            return None, None, None, None, None

        delta_pos = next_obs[:2] - obs[:2]
        aug_obs[:2] = new_pos
        aug_next_obs[:2] = aug_obs[:2] + delta_pos

        aug_action = action.copy()
        aug_reward = self._reward(aug_next_obs)
        aug_done = self._is_done(aug_next_obs, aug_reward)
        self.num_aug = 0

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done