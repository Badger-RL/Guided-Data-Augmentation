import numpy as np
from src.augment.augmentation_function_base import AugmentationFunctionBase
from d4rl.pointmaze.maze_model import EMPTY, GOAL, WALL



class PointMazeAugmentationFunction(AugmentationFunctionBase):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.agent_offset = 0.2 # the agent and target coordinate systems are different for some reason.
        self.effective_wall_width = 0.603
        self.thetas = np.array([0, np.pi/2, np.pi, np.pi*3/2])
        self.rotation_matrices = []
        for theta in self.thetas:
            M = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            self.rotation_matrices.append(M)
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
        w = w.astype(int)
        h = h.astype(int)
        xhi = w+1-0.5
        yhi = h+1-0.5
        xlo = w-1+0.5
        ylo = h-1+0.5

        # Empty/goal locations are surrounded by walls, so we don't need to check if w+1/w-1/h+1/h-1 are valid locations.
        mask = self.env.maze_arr[w+1, h] == WALL
        xhi[mask] = w[mask]+1-self.effective_wall_width

        mask = self.env.maze_arr[w, h+1] == WALL
        yhi[mask] = h[mask]+1-self.effective_wall_width

        mask = self.env.maze_arr[w-1, h] == WALL
        xlo[mask] = w[mask]-1+self.effective_wall_width

        mask = self.env.maze_arr[w, h-1] == WALL
        ylo[mask] = h[mask]-1+self.effective_wall_width

        xlo -= self.agent_offset
        ylo -= self.agent_offset
        xhi -= self.agent_offset
        yhi -= self.agent_offset

        return (xlo, ylo, xhi, yhi)

    def _sample_from_box(self, xlo, ylo, xhi, yhi):
        return self.env.np_random.uniform(low=[xlo, ylo], high=[xhi,yhi])

    def _is_in_wall(self, box_location, x, y):
        xlo, ylo = box_location - self.agent_offset - self.effective_wall_width
        xhi, yhi = box_location - self.agent_offset + self.effective_wall_width

        return (x > xlo & y > ylo) & (x < xhi & y < yhi)

    def _is_valid_position(self, xy):
        x, y = xy[:, 0], xy[:, 1]
        is_valid_position = True
        for box_location in self.wall_locations:
            if self._is_in_wall(box_location, x, y):
                is_valid_position = False
                break

        return is_valid_position

    def _sample_pos(self, n=1):
        idx = np.random.choice(len(self.env.empty_and_goal_locations), size=(n,))
        location = np.array(np.array(self.env.empty_and_goal_locations)[idx]).astype(self.env.observation_space.dtype)
        boundaries = self._get_valid_boundaries(*(location.T))
        return self._sample_from_box(*boundaries)

    def _sample_theta(self, **kwargs):
        return np.random.choice(self.thetas)

    def _sample_rotation_matrix(self, **kwargs):
        return np.random.choice(self.rotation_matrices)

    def _reward(self, next_obs):
        # Rewar dshould intuitively be computed using next_obs, but D4RL uses the current obs (D4RL bug)
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

        n = len(obs)
        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()

        # translate
        aug_obs[:, :2] = self._sample_pos(n)
        delta_obs = next_obs - obs
        aug_next_obs[:, :2] = aug_obs[:, :2] + delta_obs[:, :2]

        M = self._sample_rotation_matrix(obs=aug_obs, next_obs=aug_next_obs)

        # rotate action
        aug_action = M.dot(action).T

        # rotate next_obs about obs
        rotated_delta_obs = M.dot(delta_obs[:, :2]).T
        aug_next_obs[:, :2] = aug_obs[:, :2] + rotated_delta_obs

        # rotate velocity
        aug_obs[:, 2:] = M.dot(aug_obs[:, 2:]).T
        aug_next_obs[:, 2:] = M.dot(aug_next_obs[:, 2:]).T

        # initial pos is always valid, so we only need to check that the next pos isn't inside a wall
        next_pos = aug_next_obs[:, :2]
        is_valid = self._is_valid_position(next_pos)

        aug_reward = self._reward(aug_next_obs)
        aug_done = done

        return aug_obs[is_valid], aug_action[is_valid], aug_reward[is_valid], aug_next_obs[is_valid], aug_done[is_valid]

class PointMazeGuidedAugmentationFunction(PointMazeAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

        if self.env.maze_arr.shape[0] == 5:
            self.guide_thetas = {
                #left
                (1, 1): [0, np.pi/2, np.pi, np.pi*3/2],
                (1, 2): [np.pi*3/2],
                (1, 3): [np.pi*3/2],
                # top
                (2, 3): [np.pi],
                (3, 3): [np.pi],
                # right
                (3, 1): [np.pi/2],
                (3, 2): [np.pi/2],
            }
        elif self.env.maze_arr.shape[0] == 8:
            self.guide_thetas = {
                (1, 1): [0],
                (2, 1): [np.pi/2],
                # (1, 1): [],
                (4, 1): [np.pi/2],
                (5, 1): [np.pi],
                (6, 1): [np.pi/2],

                (1, 2): [0],
                (2, 2): [0],
                (3, 2): [np.pi/2],
                (4, 2): [np.pi],
                # (5, 2): [np.pi],
                (6, 2): [np.pi/2],

                # (1, 3): [0],
                # (2, 3): [0],
                (3, 3): [np.pi / 2],
                # (4, 3): [np.pi],
                (5, 3): [np.pi/2],
                (6, 3): [np.pi],

                (1, 4): [0],
                (2, 4): [0],
                (3, 4): [0],
                (4, 4): [np.pi/2],
                (5, 4): [np.pi],
                # (6, 4): [np.pi],

                (1, 5): [0],
                (2, 5): [np.pi*3/2],
                # (3, 5): [np.pi / 2],
                (4, 5): [np.pi/2],
                # (5, 5): [np.pi],
                (6, 5): [np.pi / 2],

                (1, 6): [0],
                (2, 6): [np.pi*3/2],
                # (3, 6): [np.pi / 2],
                (4, 6): [0],
                (5, 6): [0],
                (6, 6): [0, np.pi / 2, np.pi, np.pi * 3 / 2],
            }
        elif self.env.maze_arr.shape[0] == 9:
            self.guide_thetas = {
                (1, 1): [np.pi/2],
                (2, 1): [0],
                (3, 1): [np.pi/2],
                (4, 1): [np.pi],
                (5, 1): [np.pi],
                # (6, 1): [np.pi / 2],
                (7, 1): [np.pi/2],

                (1, 2): [np.pi/2],
                # (2, 2): [0],
                (3, 2): [np.pi / 2],
                # (4, 2): [np.pi],
                (5, 2): [np.pi*3/2],
                (6, 2): [np.pi],
                (7, 2): [np.pi],

                (1, 3): [np.pi/2],
                # (2, 3): [0],
                (3, 3): [np.pi / 2],
                # (4, 3): [np.pi],
                # (5, 3): [np.pi / 2],
                # (6, 3): [np.pi],
                # (7, 3): [np.pi],

                (1, 4): [0],
                (2, 4): [0],
                (3, 4): [np.pi/2],
                # (4, 4): [np.pi / 2],
                (5, 4): [0],
                (6, 4): [0],
                (7, 4): [np.pi/2],

                # (1, 5): [0],
                # (2, 5): [np.pi * 3 / 2],
                (3, 5): [np.pi / 2],
                # (4, 5): [np.pi / 2],
                # (5, 5): [np.pi],
                # (6, 5): [np.pi / 2],
                (7, 5): [np.pi / 2],

                (1, 6): [0],
                (2, 6): [0],
                (3, 6): [0],
                (4, 6): [0],
                (5, 6): [np.pi/2],
                (6, 6): [np.pi],
                (7, 6): [np.pi],

                (1, 7): [np.pi*3/2],
                # (2, 7): [0],
                # (3, 7): [0],
                # (4, 7): [0],
                (5, 7): [np.pi / 2],
                # (6, 7): [np.pi],
                # (7, 7): [np.pi],

                (1, 8): [np.pi/2],
                (2, 8): [0],
                (3, 8): [np.pi/2],
                # (4, 8): [0],
                (5, 8): [0],
                (6, 8): [0],
                (7, 8): [np.pi/2],

                (1, 9): [np.pi/2],
                # (2, 9): [0],
                (3, 9): [np.pi/2],
                # (4, 9): [0],
                (5, 9): [np.pi*3 / 2],
                # (6, 9): [np.pi],
                (7, 9): [0, np.pi / 2, np.pi, np.pi * 3 / 2],

                (1, 10): [0],
                (2, 10): [0],
                (3, 10): [0],
                (4, 10): [0],
                (5, 10): [np.pi*3/2],
                (6, 10): [0],
                (7, 10): [np.pi*3/2],
            }

    def _sample_theta(self, obs, next_obs, **kwargs):

        x, y = obs[0], obs[1]
        delta_obs = next_obs - obs
        theta = np.arctan2(delta_obs[1], delta_obs[0])

        location = (int(np.round(x+self.agent_offset)), int(np.round(y+self.agent_offset)))
        guide_thetas = self.guide_thetas[location]
        guide_theta = np.random.choice(guide_thetas)

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
            aug_obs[:, :2] = self._sample_pos()
            delta_obs = next_obs - obs
            aug_next_obs[:, :2] = aug_obs[:, :2] + delta_obs[:, :2]

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
            rotated_delta_obs = M.dot(delta_obs[:, :2]).T
            aug_next_obs[:, :2] = aug_obs[:, :2] + rotated_delta_obs

            # rotate velocity
            aug_obs[:, 2:] = M.dot(aug_obs[:, 2:]).T
            aug_next_obs[:, 2:] = M.dot(aug_next_obs[:, 2:]).T

            # initial pos is always valid, so we only need to check that the next pos isn't inside a wall
            next_pos = aug_next_obs[:, :2]
            is_valid = self._is_valid_position(next_pos)

        aug_reward = self._reward(aug_next_obs)
        aug_done = done

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done





