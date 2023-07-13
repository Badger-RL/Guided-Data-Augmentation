import copy
from typing import Dict

import numpy as np

from GuidedDataAugmentationForRobotics.src.augment.base_augmentation_function import AbstractSimAugmentationFunction
from augment.utils import convert_to_absolute_obs, calculate_reward, convert_to_relative_obs


class RotateReflectTranslate(AbstractSimAugmentationFunction):
    '''
    Translate the robot and ball by the same (delta_x, delta_y).
    '''
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 **kwargs,
                 ):
        delta_x = np.random.uniform(-1, 1)
        delta_y = np.random.uniform(-1, 1)

        obs[0] += delta_x
        obs[1] += delta_y

        next_obs[0] += delta_x
        next_obs[1] += delta_y

        aug_reward, _ = self.calculate_reward(aug_absolute_next_obs)

        aug_obs = self._convert_to_relative_obs(aug_absolute_obs)
        aug_next_obs = self._convert_to_relative_obs(aug_absolute_next_obs)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done