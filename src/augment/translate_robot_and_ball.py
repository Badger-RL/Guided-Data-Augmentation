from typing import Dict

import numpy as np

from GuidedDataAugmentationForRobotics.src.augment.augmentation_function import AbstractSimAugmentationFunction


class TranslateRobotAndBall(AbstractSimAugmentationFunction):
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

        # random robot position
        absolute_obs = self._convert_to_absolute_obs(obs)
        absolute_next_obs = self._convert_to_absolute_obs(next_obs)

        if self.at_goal(absolute_obs[2], absolute_obs[3]):
            return None, None, None, None, None

        xmin = np.min([absolute_obs[0], absolute_next_obs[0], absolute_obs[2], absolute_next_obs[2]])
        ymin = np.min([absolute_obs[1], absolute_next_obs[1], absolute_obs[3], absolute_next_obs[3]])
        xmax = np.max([absolute_obs[0], absolute_next_obs[0], absolute_obs[2], absolute_next_obs[2]])
        ymax = np.max([absolute_obs[1], absolute_next_obs[1], absolute_obs[3], absolute_next_obs[3]])

        # Translate bottom left corner of the righter bounding box containing the robot and ball
        new_x = np.random.uniform(-4500, 4500-(xmax-xmin))
        new_y = np.random.uniform(-3000, 3000-(ymax-ymin))

        delta_x = new_x - xmin
        delta_y = new_y - ymin
        # delta_x = 0
        # delta_y = 0

        absolute_obs[0] += delta_x
        absolute_obs[1] += delta_y
        absolute_obs[2] += delta_x
        absolute_obs[3] += delta_y

        absolute_next_obs[0] += delta_x
        absolute_next_obs[1] += delta_y
        absolute_next_obs[2] += delta_x
        absolute_next_obs[3] += delta_y

        aug_obs = self._convert_to_relative_obs(absolute_obs)
        aug_next_obs = self._convert_to_relative_obs(absolute_next_obs)
        aug_action = action
        aug_reward, _ = self.calculate_reward(absolute_next_obs)
        aug_done = done

        # print(aug_obs - obs)
        # print(aug_next_obs - next_obs)
        # print(aug_reward, reward)
        # assert np.allclose(aug_obs, obs, atol=1e-6)
        # assert np.allclose(aug_next_obs, next_obs, atol=1e-6)
        # assert np.isclose(aug_reward, reward, atol=1e-6)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done