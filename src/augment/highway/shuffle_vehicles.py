import numpy as np
from src.augment.augmentation_function_base import AugmentationFunctionBase

class ShuffleVehicles(AugmentationFunctionBase):
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

        aug_obs = aug_obs.reshape(8, -1)
        aug_next_obs = aug_next_obs.reshape(8, -1)

        indices = np.arange(1,8)
        indices_shuffled = indices.copy()
        np.random.shuffle(indices_shuffled)

        aug_obs[indices] = aug_obs[indices_shuffled]
        aug_next_obs[indices] = aug_next_obs[indices_shuffled]

        aug_reward = reward
        aug_action = action.copy()
        aug_done = done

        aug_obs = aug_obs.reshape(-1)
        aug_next_obs = aug_next_obs.reshape(-1)


        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done
