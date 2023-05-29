import numpy as np
from augment.augmentation_function_base import AugmentationFunctionBase
from d4rl.pointmaze.maze_model import EMPTY, GOAL


class PointMazeAugmentationFunction(AugmentationFunctionBase):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.target = np.array((0,0))
        self.valid_locations = []
        width, height = self.env.maze_arr.shape
        for w in range(width):
            for h in range(height):
                location_type = self.env.maze_arr[w, h]
                if location_type in [EMPTY, GOAL]:
                    box_location = np.array((w,h))
                    self.valid_locations.append(box_location)

    def _is_in_box(self, box_location, x, y):
        xlo, ylo = box_location - 0.597
        xhi, yhi = box_location + 0.198

        if (x > xlo and y > ylo) and (x < xhi and y < yhi):
            return True
        else:
            return False


    def _is_valid_position(self, xy):
        x, y = xy[0], xy[1]
        is_valid_position = False
        for box_location in self.valid_locations:
            if self._is_in_box(box_location, x, y):
                is_valid_position = True
                break

        return is_valid_position

    def _sample_pos(self, n=1):
        idx = np.random.choice(len(self.env.empty_and_goal_locations))
        reset_location = np.array(self.env.empty_and_goal_locations[idx]).astype(self.env.observation_space.dtype)
        qpos = reset_location + self.env.np_random.uniform(low=-0.597, high=.198, size=(self.env.model.nq,))
        return qpos

    def _sample_theta(self, n=1):
        return np.random.choice([-np.pi, -np.pi/2, +np.pi/2, +np.pi])

    def _reward(self, next_obs):
        if self.env.reward_type == 'sparse':
            reward = 1.0 if np.linalg.norm(next_obs[0:2] - self.target) <= 0.5 else 0.0
        elif self.env.reward_type == 'dense':
            reward = np.exp(-np.linalg.norm(next_obs[0:2] - self.target))
        else:
            raise ValueError('Unknown reward type %s' % self.env.reward_type)

        return reward


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

            # translate
            aug_obs[:2] = self._sample_pos()
            delta_obs = next_obs - obs
            aug_next_obs[:2] = aug_obs[:2] + delta_obs[:2]

            theta = self._sample_theta()
            M = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])

            # rotate action
            aug_action = M.dot(action).T

            # rotate next_obs about obs
            rotated_delta_obs = M.dot(delta_obs[:2]).T
            aug_next_obs[:2] = aug_obs[:2] + rotated_delta_obs

            # rotate velocity
            aug_obs[2:] = M.dot(aug_obs[2:]).T
            aug_next_obs[2:] = M.dot(aug_next_obs[2:]).T

            # initial pos is always valid, so we only need to check that the next pos isn't inside a wall
            next_pos = aug_next_obs[:2]
            is_valid = self._is_valid_position(next_pos)

        aug_reward = self._reward(aug_next_obs)
        aug_done = done

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done
