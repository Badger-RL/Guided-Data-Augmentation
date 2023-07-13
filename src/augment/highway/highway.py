import numpy as np
from src.augment.augmentation_function_base import AugmentationFunctionBase
from d4rl.pointmaze.maze_model import EMPTY, GOAL, WALL

class ChangeLane(AugmentationFunctionBase):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_lane(self):
        return np.random.randint(low=0, high=4)

    def _sample_y(self):
        lane_i = self._sample_lane()
        return lane_i*4 + np.random.uniform(-1, +1)

    def _convert_to_absolute_y(self, obs):
        pass

    def _convert_to_relative_y(self, obs):
        origin_y = obs[0, 2]

        obs[1:]
        origin_y = obs[0, 2]

    def _sample_new_origin_y(self, aug_obs, old_origin_y):
        aug_obs = aug_obs.reshape(8, 5)

        vehicle_lanes = (old_origin_y + aug_obs[1:, 2])
        attempt_count = 0
        while True:
            attempt_count += 1
            new_lane = self._sample_lane() / 4
            vehicle_x = aug_obs[1:, 1]

            is_in_lane = np.abs(vehicle_lanes - new_lane) < 0.25
            is_far = np.all(np.abs(vehicle_x[is_in_lane]) > 0.15)
            if is_far:
                break
            if attempt_count > 10:
                return None

        new_origin_y = new_lane + np.random.uniform(-0.1, +0.1)
        return new_origin_y

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        if reward < 0:
            return None, None, None, None, None

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()

        aug_obs = aug_obs.reshape(8, 5)
        aug_next_obs = aug_next_obs.reshape(8, 5)

        old_origin_y = obs[2]
        new_origin_y = self._sample_new_origin_y(aug_obs, old_origin_y)
        new_origin_y = 0.75
        if new_origin_y is None:
            return None, None, None, None, None

        aug_obs[0, 2] = new_origin_y
        aug_next_obs[0, 2] = new_origin_y

        if aug_next_obs[0, 2] > 0.875 or aug_next_obs[0, 2] < -0.125:
            return None, None, None, None, None

        aug_obs[1:, 2] += old_origin_y - new_origin_y
        aug_next_obs[1:, 2] = old_origin_y - new_origin_y

        old_lane_reward = 0.1*np.round(old_origin_y*4)/3
        new_lane_reward = 0.1*np.round(new_origin_y*4)/3
        aug_reward = reward - old_lane_reward + new_lane_reward
        aug_action = action.copy()
        aug_done = done

        aug_obs = aug_obs.reshape(-1)
        aug_next_obs = aug_next_obs.reshape(-1)


        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done


class ChangeLaneGuided(ChangeLane):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_new_origin_y(self, aug_obs, old_origin_y):
        vehicle_lanes = old_origin_y + aug_obs[1:, 2]  # convert to aboslute y
        rightmost_lane = np.max(vehicle_lanes)
        if rightmost_lane < 0.65:
            new_lane = 0.75
        else:
            vehicle_x = aug_obs[1:, 1]
            # determine which vehicles are close in x
            is_far = np.abs(vehicle_x) > 0.15
            if not np.any(is_far):
                return None

            # choose rightmost lane that in which vehicles are far in x
            new_lane = np.max(vehicle_lanes[is_far])
        new_origin_y = new_lane + np.random.uniform(-0.1, +0.1)
        return new_origin_y


class CreateCollision(AugmentationFunctionBase):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        if reward < 0:
            return None, None, None, None, None

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()

        aug_obs = aug_obs.reshape(8, 5)
        aug_next_obs = aug_next_obs.reshape(8, 5)

        aug_vehicle = np.random.randint(1,8)

        aug_obs[aug_vehicle, 1] = np.random.uniform(0.02, 0.035)
        aug_obs[aug_vehicle, 2] = np.random.uniform(-0.1, 0.1)

        aug_next_obs[aug_vehicle, 1] = np.random.uniform(0.02, 0.035)
        aug_next_obs[aug_vehicle, 2] = np.random.uniform(-0.1, 0.1)


        aug_reward = reward - 1
        aug_action = action.copy()
        aug_done = True

        aug_obs = aug_obs.reshape(-1)
        aug_next_obs = aug_next_obs.reshape(-1)


        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done

class TranslateVehicle(AugmentationFunctionBase):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        if reward < 0:
            return None, None, None, None, None

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()

        aug_obs = aug_obs.reshape(8, 5)
        aug_next_obs = aug_next_obs.reshape(8, 5)

        aug_vehicle = np.random.randint(1,8)

        new_x = np.random.uniform(0.02, 1)
        new_y = np.random.uniform(-0.15, 0.85)

        delta_x = aug_next_obs[aug_vehicle, 1] - aug_obs[aug_vehicle, 1]
        delta_y = aug_next_obs[aug_vehicle, 2] - aug_obs[aug_vehicle, 2]

        aug_obs[aug_vehicle, 1] = new_x
        aug_obs[aug_vehicle, 2] = new_y

        aug_next_obs[aug_vehicle, 1] = new_x + delta_x
        aug_next_obs[aug_vehicle, 2] = new_y + delta_y

        if new_x < 0.035 and np.abs(new_y) < 0.1:
            aug_reward = reward - 1
            aug_action = action.copy()
            aug_done = True
        else:
            aug_reward = reward
            aug_action = action.copy()
            aug_done = done

        aug_obs = aug_obs.reshape(-1)
        aug_next_obs = aug_next_obs.reshape(-1)


        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done


class TranslateAllVehicles(AugmentationFunctionBase):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        if reward < 0:
            return None, None, None, None, None

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()

        aug_obs = aug_obs.reshape(8, 5)
        aug_next_obs = aug_next_obs.reshape(8, 5)

        collision = False
        for aug_vehicle in range(1,8):

            new_x = np.random.uniform(0.02, 1)
            new_y = np.random.uniform(-0.15, 0.85)

            delta_x = aug_next_obs[aug_vehicle, 1] - aug_obs[aug_vehicle, 1]
            delta_y = aug_next_obs[aug_vehicle, 2] - aug_obs[aug_vehicle, 2]

            aug_obs[aug_vehicle, 1] = new_x
            aug_obs[aug_vehicle, 2] = new_y

            aug_next_obs[aug_vehicle, 1] = new_x + delta_x
            aug_next_obs[aug_vehicle, 2] = new_y + delta_y

            if new_x < 0.035 and np.abs(new_y) < 0.1:
                collision = True

        if collision:
            aug_reward = reward - 1
            aug_action = action.copy()
            aug_done = True
        else:
            aug_reward = reward
            aug_action = action.copy()
            aug_done = done

        aug_obs = aug_obs.reshape(-1)
        aug_next_obs = aug_next_obs.reshape(-1)


        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done

class TranslateAllVehiclesWithCollision(AugmentationFunctionBase):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        if reward < 0:
            return None, None, None, None, None

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()

        aug_obs = aug_obs.reshape(8, 5)
        aug_next_obs = aug_next_obs.reshape(8, 5)

        collision = False
        for aug_vehicle in range(1,8):

            new_x = np.random.uniform(0.02, 1)
            new_y = np.random.uniform(-0.15, 0.85)

            delta_x = aug_next_obs[aug_vehicle, 1] - aug_obs[aug_vehicle, 1]
            delta_y = aug_next_obs[aug_vehicle, 2] - aug_obs[aug_vehicle, 2]

            aug_obs[aug_vehicle, 1] = new_x
            aug_obs[aug_vehicle, 2] = new_y

            aug_next_obs[aug_vehicle, 1] = new_x + delta_x
            aug_next_obs[aug_vehicle, 2] = new_y + delta_y

        aug_vehicle = np.random.randint(1, 8)

        aug_obs[aug_vehicle, 1] = np.random.uniform(0.02, 0.035)
        aug_obs[aug_vehicle, 2] = np.random.uniform(-0.1, 0.1)

        aug_next_obs[aug_vehicle, 1] = np.random.uniform(0.02, 0.035)
        aug_next_obs[aug_vehicle, 2] = np.random.uniform(-0.1, 0.1)

        aug_reward = reward - 1
        aug_action = action.copy()
        aug_done = True

        aug_obs = aug_obs.reshape(-1)
        aug_next_obs = aug_next_obs.reshape(-1)


        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done