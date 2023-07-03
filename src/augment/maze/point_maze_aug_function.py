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

        w = int(w)
        h = int(h)

        # assert self.env.maze_arr[w, h] in [EMPTY, GOAL] # must be valid location

        xhi = w+1-0.5
        yhi = h+1-0.5
        xlo = w-1+0.5
        ylo = h-1+0.5

        if self.env.maze_arr[w+1, h] in [WALL]:
            xhi = w+1-self.effective_wall_width
        if self.env.maze_arr[w, h+1] in [WALL]:
            yhi = h+1-self.effective_wall_width
        if self.env.maze_arr[w-1, h] in [WALL]:
            xlo = w-1+self.effective_wall_width
        if self.env.maze_arr[w, h-1] in [WALL]:
            ylo = h-1+self.effective_wall_width

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

        if (x > xlo and y > ylo) and (x < xhi and y < yhi):
            return True
        else:
            return False

    def _check_corners(self, xy, location):
        x, y = xy[0], xy[1]
        is_valid_position = True

        w, h = int(location[0]), int(location[1])
        for loc in [(w + 1, h + 1), (w + 1, h - 1), (w - 1, h + 1), (w - 1, h - 1)]:
            if self.env.maze_arr[loc[0], loc[1]] == WALL:
                loc = np.array(loc)
                if self._is_in_wall(loc, x, y):
                    is_valid_position = False
                    break

        return is_valid_position

    def _sample_pos(self, n=1):
        idx = np.random.choice(len(self.env.empty_and_goal_locations))
        location = np.array(self.env.empty_and_goal_locations[idx]).astype(self.env.observation_space.dtype)
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

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()

        while True:
            # translate
            aug_obs[:2], aug_location = self._sample_pos()
            delta_obs = next_obs - obs
            aug_next_obs[:2] = aug_obs[:2] + delta_obs[:2]

            M = self._sample_rotation_matrix(obs=aug_obs, next_obs=aug_next_obs)

            # rotate next_obs about obs
            rotated_delta_obs = M.dot(delta_obs[:2]).T
            aug_next_obs[:2] = aug_obs[:2] + rotated_delta_obs

            # corner case (literally): check that the agent isn't inside a wall
            pos_is_valid = self._check_corners(aug_obs[:2], aug_location)
            next_pos_is_valid = self._check_corners(aug_next_obs[:2], aug_location)

            # if new positions are not valid, immediately sample a new position
            if not (pos_is_valid and next_pos_is_valid):
                continue

            # rotate action
            aug_action = M.dot(action).T

            # rotate velocity
            aug_obs[2:] = M.dot(aug_obs[2:]).T
            aug_next_obs[2:] = M.dot(aug_next_obs[2:]).T
            break

        aug_reward = self._reward(aug_next_obs)
        aug_done = done

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done

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
                (3, 1): [],
                (4, 1): [np.pi/2],
                (5, 1): [np.pi],
                (6, 1): [np.pi/2],

                (1, 2): [0],
                (2, 2): [0],
                (3, 2): [np.pi/2],
                (4, 2): [np.pi],
                (5, 2): [],
                (6, 2): [np.pi/2],

                (1, 3): [],
                (2, 3): [],
                (3, 3): [np.pi / 2],
                (4, 3): [],
                (5, 3): [np.pi/2],
                (6, 3): [np.pi],

                (1, 4): [0],
                (2, 4): [0],
                (3, 4): [0],
                (4, 4): [np.pi/2],
                (5, 4): [np.pi],
                (6, 4): [],

                (1, 5): [0],
                (2, 5): [np.pi*3/2],
                (3, 5): [],
                (4, 5): [np.pi/2],
                (5, 5): [],
                (6, 5): [np.pi / 2],

                (1, 6): [0],
                (2, 6): [np.pi*3/2],
                (3, 6): [],
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
                (6, 1): [],
                (7, 1): [np.pi/2],

                (1, 2): [np.pi/2],
                (2, 2): [],
                (3, 2): [np.pi / 2],
                (4, 2): [],
                (5, 2): [np.pi*3/2],
                (6, 2): [np.pi],
                (7, 2): [np.pi],

                (1, 3): [np.pi/2],
                (2, 3): [],
                (3, 3): [np.pi / 2],
                (4, 3): [],
                (5, 3): [],
                (6, 3): [],
                (7, 3): [],

                (1, 4): [0],
                (2, 4): [0],
                (3, 4): [np.pi/2],
                (4, 4): [],
                (5, 4): [0],
                (6, 4): [0],
                (7, 4): [np.pi/2],

                (1, 5): [],
                (2, 5): [],
                (3, 5): [np.pi / 2],
                (4, 5): [],
                (5, 5): [],
                (6, 5): [],
                (7, 5): [np.pi / 2],

                (1, 6): [0],
                (2, 6): [0],
                (3, 6): [0],
                (4, 6): [0],
                (5, 6): [np.pi/2],
                (6, 6): [np.pi],
                (7, 6): [np.pi],

                (1, 7): [np.pi*3/2],
                (2, 7): [],
                (3, 7): [],
                (4, 7): [],
                (5, 7): [np.pi / 2],
                (6, 7): [],
                (7, 7): [],

                (1, 8): [np.pi/2],
                (2, 8): [0],
                (3, 8): [np.pi/2],
                (4, 8): [],
                (5, 8): [0],
                (6, 8): [0],
                (7, 8): [np.pi/2],

                (1, 9): [np.pi/2],
                (2, 9): [],
                (3, 9): [np.pi/2],
                (4, 9): [],
                (5, 9): [np.pi*3 / 2],
                (6, 9): [],
                (7, 9): [0, np.pi / 2, np.pi, np.pi * 3 / 2],

                (1, 10): [0],
                (2, 10): [0],
                (3, 10): [0],
                (4, 10): [0],
                (5, 10): [np.pi*3/2],
                (6, 10): [0],
                (7, 10): [np.pi*3/2],
            }

    def _sample_rotation_matrix(self, obs, next_obs, **kwargs):
        x, y = obs[0], obs[1]
        delta_obs = next_obs - obs
        theta = np.arctan2(delta_obs[1], delta_obs[0])
        if theta < 0:
            theta += 2*np.pi

        location = (int(np.round(x+self.agent_offset)), int(np.round(y+self.agent_offset)))
        guide_thetas = self.guide_thetas[location]
        guide_theta = np.random.choice(guide_thetas)

        aug_thetas = (self.thetas + theta)%(2*np.pi)

        # need to handle 0/2*np.pi separately
        if np.isclose(guide_theta, 0):
            dist1 = np.abs(0 - aug_thetas)
            dist2 = np.abs(2* np.pi - aug_thetas)
            idx = np.argmin(np.concatenate([dist1, dist2])) % 4
        else:
            dist = np.abs(guide_theta - aug_thetas)
            idx = np.argmin(dist)

        return self.rotation_matrices[idx]


    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()

        while True:
            # translate
            aug_obs[:2], aug_location = self._sample_pos()
            delta_obs = next_obs - obs
            aug_next_obs[:2] = aug_obs[:2] + delta_obs[:2]

            M = self._sample_rotation_matrix(obs=aug_obs, next_obs=aug_next_obs)

            # rotate next_obs about obs
            rotated_delta_obs = M.dot(delta_obs[:2]).T
            aug_next_obs[:2] = aug_obs[:2] + rotated_delta_obs

            # corner case (literally): check that the agent isn't inside a wall
            pos_is_valid = self._check_corners(aug_obs[:2], aug_location)
            next_pos_is_valid = self._check_corners(aug_next_obs[:2], aug_location)

            # if new positions are not valid, immediately sample a new position
            if not (pos_is_valid and next_pos_is_valid):
                continue

            # rotate action
            aug_action = M.dot(action).T

            # rotate velocity
            aug_obs[2:] = M.dot(aug_obs[2:]).T
            aug_next_obs[2:] = M.dot(aug_next_obs[2:]).T
            break

        aug_reward = self._reward(aug_next_obs)
        aug_done = done

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done


class PointMazeTrajectoryAugmentationFunction(PointMazeAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    
    def _is_in_box(self, xlo, ylo, xhi, yhi, x, y):
        if (x > xlo and y > ylo) and (x < xhi and y < yhi):
            return True
        else:
            return False
        
    def _get_location(self, obs):
        for i in range(len(self.env.empty_and_goal_locations)):
            location = np.array(self.env.empty_and_goal_locations[i]).astype(self.env.observation_space.dtype)
            boundaries = self._get_valid_boundaries(*location)
            if self._is_in_box(*boundaries, obs[0], obs[1]):
                return location
        return None
    
    def augment_fixed_origin(self, 
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                aug_obs: np.ndarray,
                **kwargs,
                ):
        aug_next_obs = next_obs.copy()
        while True:
            # translate
            aug_location = self._get_location(aug_obs)
            print(aug_location)
            if aug_location is None:
                return None, None, None, None, None
            delta_obs = next_obs - obs
            aug_next_obs[:2] = aug_obs[:2] + delta_obs[:2]

            M = self._sample_rotation_matrix(obs=aug_obs, next_obs=aug_next_obs)

            # rotate next_obs about obs
            rotated_delta_obs = M.dot(delta_obs[:2]).T
            aug_next_obs[:2] = aug_obs[:2] + rotated_delta_obs

            # corner case (literally): check that the agent isn't inside a wall
            pos_is_valid = self._check_corners(aug_obs[:2], aug_location)
            next_pos_is_valid = self._check_corners(aug_next_obs[:2], aug_location)

            # if new positions are not valid, immediately sample a new position
            if not (pos_is_valid and next_pos_is_valid):
                continue

            # rotate action
            aug_action = M.dot(action).T

            # rotate velocity
            aug_obs[2:] = M.dot(aug_obs[2:]).T
            aug_next_obs[2:] = M.dot(aug_next_obs[2:]).T
            break

        aug_reward = self._reward(aug_next_obs)
        aug_done = done

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done


    def augment_trajectory(self, trajectory: dict):
        num_of_transitions = len(trajectory['observations'])
        augmented_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
        }
        for i in range(num_of_transitions):
            observations = trajectory['observations'][i]
            actions = trajectory['actions'][i]
            next_observations = trajectory['next_observations'][i]
            rewards = trajectory['rewards'][i]
            dones = trajectory['terminals'][i]
            if i == 0:
                augmented_obs, augmented_action, augmented_reward, augmented_next_obs, augmented_done = self.augment(
                    observations, actions, next_observations, rewards, dones)
            else:
                augmented_obs, augmented_action, augmented_reward, augmented_next_obs, augmented_done = self.augment_fixed_origin(
                    observations, actions, next_observations, rewards, dones, augmented_obs)
            
            if augmented_obs is None:
                break

            augmented_trajectory['observations'].append(augmented_obs)
            augmented_trajectory['actions'].append(augmented_action)
            augmented_trajectory['rewards'].append(augmented_reward)
            augmented_trajectory['next_observations'].append(augmented_next_obs)
            augmented_trajectory['terminals'].append(augmented_done)
        return augmented_trajectory


class PointMazeGuidedTrajectoryAugmentationFunction(PointMazeGuidedAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    
    def _is_in_box(self, xlo, ylo, xhi, yhi, x, y):
        if (x > xlo and y > ylo) and (x < xhi and y < yhi):
            return True
        else:
            return False
        
    def _get_location(self, obs):
        for i in range(len(self.env.empty_and_goal_locations)):
            location = np.array(self.env.empty_and_goal_locations[i]).astype(self.env.observation_space.dtype)
            boundaries = self._get_valid_boundaries(*location)
            if self._is_in_box(*boundaries, obs[0], obs[1]):
                return location
        return None
    
    def augment_fixed_origin(self, 
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                aug_obs: np.ndarray,
                **kwargs,
                ):
        aug_next_obs = next_obs.copy()
        while True:
            # translate
            aug_location = self._get_location(aug_obs)
            print(aug_location)
            if aug_location is None:
                return None, None, None, None, None
            delta_obs = next_obs - obs
            aug_next_obs[:2] = aug_obs[:2] + delta_obs[:2]

            M = self._sample_rotation_matrix(obs=aug_obs, next_obs=aug_next_obs)

            # rotate next_obs about obs
            rotated_delta_obs = M.dot(delta_obs[:2]).T
            aug_next_obs[:2] = aug_obs[:2] + rotated_delta_obs

            # corner case (literally): check that the agent isn't inside a wall
            pos_is_valid = self._check_corners(aug_obs[:2], aug_location)
            next_pos_is_valid = self._check_corners(aug_next_obs[:2], aug_location)

            # if new positions are not valid, immediately sample a new position
            if not (pos_is_valid and next_pos_is_valid):
                continue

            # rotate action
            aug_action = M.dot(action).T

            # rotate velocity
            aug_obs[2:] = M.dot(aug_obs[2:]).T
            aug_next_obs[2:] = M.dot(aug_next_obs[2:]).T
            break

        aug_reward = self._reward(aug_next_obs)
        aug_done = done

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done


    def augment_trajectory(self, trajectory: dict):
        num_of_transitions = len(trajectory['observations'])
        augmented_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
        }
        for i in range(num_of_transitions):
            observations = trajectory['observations'][i]
            actions = trajectory['actions'][i]
            next_observations = trajectory['next_observations'][i]
            rewards = trajectory['rewards'][i]
            dones = trajectory['terminals'][i]
            if i == 0:
                augmented_obs, augmented_action, augmented_reward, augmented_next_obs, augmented_done = self.augment(
                    observations, actions, next_observations, rewards, dones)
            else:
                augmented_obs, augmented_action, augmented_reward, augmented_next_obs, augmented_done = self.augment_fixed_origin(
                    observations, actions, next_observations, rewards, dones, augmented_obs)
            
            if augmented_obs is None:
                break

            augmented_trajectory['observations'].append(augmented_obs)
            augmented_trajectory['actions'].append(augmented_action)
            augmented_trajectory['rewards'].append(augmented_reward)
            augmented_trajectory['next_observations'].append(augmented_next_obs)
            augmented_trajectory['terminals'].append(augmented_done)
        return augmented_trajectory