import numpy as np
from src.augment.augmentation_function_base import AugmentationFunctionBase
from d4rl.pointmaze.maze_model import EMPTY, GOAL, WALL

class IntersectionTranslate(AugmentationFunctionBase):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _has_collided(self, obs):
        self.env.set_state(obs)
        dt = self.env.config["simulation_frequency"]
        ego_vehicle = self.env.controlled_vehicles[0]
        for other in self.env.road.vehicles[1:]:
            crashed = ego_vehicle.handle_collisions(other, dt)
            if crashed: break

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        if reward < 0 or obs[4] > -0.5 or reward < 0 or obs[2] < -0.25: # ego vehicle is not moving up at sufficient speed
            return None, None, None, None, None

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()

        aug_obs = aug_obs.reshape(8, 7)
        aug_next_obs = aug_next_obs.reshape(8, 7)

        aug_vehicle = 0
        # delta_x = aug_next_obs[aug_vehicle, 1] - aug_obs[aug_vehicle, 1]
        delta_y = aug_next_obs[aug_vehicle, 2] - aug_obs[aug_vehicle, 2]

        new_y = np.random.uniform(-0.1, 0.1)
        aug_obs[0, 2] = new_y
        aug_next_obs[0, 2] = new_y + delta_y

        has_collided = self._has_collided(aug_obs)
        next_has_collided = self._has_collided(aug_next_obs)

        if has_collided or next_has_collided:
            return None, None, None, None, None
            aug_reward = reward - 5
            aug_done = True
        else:
            aug_reward = reward
            aug_done = done

        aug_action = action.copy()
        # aug_reward = self.env.agent_reward(aug_action, self.env.controlled_vehicles[0])
        aug_obs = aug_obs.reshape(-1)
        aug_next_obs = aug_next_obs.reshape(-1)


        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done
