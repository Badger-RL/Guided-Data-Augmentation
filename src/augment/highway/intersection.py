import numpy as np
from src.augment.augmentation_function_base import AugmentationFunctionBase
from d4rl.pointmaze.maze_model import EMPTY, GOAL, WALL

class TranslateUncontrolled(AugmentationFunctionBase):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()

        aug_obs = aug_obs.reshape(8, 5)

        aug_reward = self._reward(aug_next_obs)
        aug_action = action.copy()
        aug_done = done

        return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done
