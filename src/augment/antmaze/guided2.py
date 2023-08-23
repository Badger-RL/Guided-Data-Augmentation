import numpy as np

from augment.antmaze.guided import AntMazeGuidedAugmentationFunction


class AntMazeGuided2AugmentationFunction(AntMazeGuidedAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

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

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        # if not self.is_valid_input(obs, next_obs):
        #     return None, None, None, None, None

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()
        while True:
            new_pos, aug_location = self._sample_pos()
            # if obs[0] < 2.5 and obs[1] > 6:
            #     continue
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

        # aug_obs[:2] += 0.5
        # aug_next_obs[:2] += 0.5
        aug_action = action.copy()
        aug_reward = self._reward(aug_next_obs)
        aug_done = aug_reward > 0
        # aug_obs[:2] += np.random.uniform(-0.1,0.1, size=(2,))
        # aug_next_obs[:2] += np.random.uniform(-0.1,0.1, size=(2,))

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done

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

        if new_pos[0] < 7 and new_pos[1] > -2 and new_pos[1] < 2:
            guide_theta = 0
        elif new_pos[0] > 7 and new_pos[1] < 7.5:
            guide_theta = np.pi / 2
        else:
            guide_theta = np.pi

        # print(guide_theta)
        # guide_theta = 0

        delta_obs = next_obs - obs
        theta = np.arctan2(delta_obs[1], delta_obs[0])
        # theta = 2 * np.arctan2(delta_obs[3], delta_obs[6])

        aug_theta = -(guide_theta - theta) + np.random.uniform(low=-np.pi/6, high=np.pi/6)

        return aug_theta

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
