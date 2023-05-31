import numpy as np
from src.augment.augmentation_function_base import AugmentationFunctionBase
from d4rl.pointmaze.maze_model import EMPTY, GOAL, WALL


class PointMazeAugmentationFunction(AugmentationFunctionBase):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.contact_dist = 0.397
        self.agent_offset = 0.2 # the agent and target coordinate systems are different for some reason.
        self.effective_wall_width = 0.603
        self.thetas = np.array([0, np.pi/2, np.pi, np.pi*3/2])
        self.target = self.env.get_target()
        self.wall_locations = []
        width, height = self.env.maze_arr.shape
        for w in range(width):
            for h in range(height):
                location_type = self.env.maze_arr[w, h]
                if location_type in [WALL]:
                    box_location = np.array((w,h))
                    self.wall_locations.append(box_location)

    def _is_in_wall(self, box_location, x, y):
        xlo, ylo = box_location - self.agent_offset - self.effective_wall_width
        xhi, yhi = box_location - self.agent_offset + self.effective_wall_width

        if (x > xlo and y > ylo) and (x < xhi and y < yhi):
            return True
        else:
            return False

    def _is_valid_position(self, xy):
        x, y = xy[0], xy[1]
        is_valid_position = True
        for box_location in self.wall_locations:
            if self._is_in_wall(box_location, x, y):
                is_valid_position = False
                break

        return is_valid_position

    def _sample_pos(self, n=1):
        # idx = np.random.choice(len(self.env.empty_and_goal_locations))
        # reset_location = np.array(self.env.empty_and_goal_locations[idx]).astype(self.env.observation_space.dtype)
        # qpos = reset_location + self.env.np_random.uniform(low=-1, high=1, size=(self.env.model.nq,))
        # return qpos

        width, height = self.env.maze_arr.shape
        return self.env.np_random.uniform(low=0.403-self.agent_offset, high=height-2 + 0.198+self.agent_offset, size=(self.env.model.nq,))

    def _sample_theta(self, **kwargs):
        return np.random.choice(self.thetas)

    def _reward(self, obs):
        # Rewar dshould intuitively be computed using next_obs, but D4RL uses the current obs (D4RL bug)
        if self.env.reward_type == 'sparse':
            reward = 1.0 if np.linalg.norm(obs[0:2] - self.target) <= 0.5 else 0.0
        elif self.env.reward_type == 'dense':
            reward = np.exp(-np.linalg.norm(obs[0:2] - self.target))
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

            theta = self._sample_theta(obs=aug_obs, next_obs=aug_next_obs)
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

class PointMazeGuidedAugmentationFunction(PointMazeAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)


    def _sample_theta(self, obs, next_obs, **kwargs):

        x, y = obs[0], obs[1]
        delta_obs = next_obs - obs
        theta = np.arctan2(delta_obs[1], delta_obs[0])

        guide_theta = theta

        if x < 1.198 and y < 1.198:
            guide_theta = np.random.choice(self.thetas)
        elif x < 1.198:
            guide_theta = np.pi * 3 / 2
        elif x < 2.403:
            guide_theta = np.pi
        elif x > 2.403 and y > 2.403:
            guide_theta = np.pi
        elif x > 2.403 and y < 2.403:
            guide_theta = np.pi/2

        aug_thetas = (self.thetas + theta)%(2*np.pi)
        dist = np.abs(guide_theta - aug_thetas)
        idx = np.argmin(dist)
        return self.thetas[idx]



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

            if np.random.random() <= 1:
                theta = self._sample_theta(obs=aug_obs, next_obs=aug_next_obs)
            else:
                theta = super()._sample_theta()
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





