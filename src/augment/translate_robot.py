import numpy as np

from GuidedDataAugmentationForRobotics.src.augment.augmentation_function import AbstractSimAugmentationFunction


class TranslateRobot(AbstractSimAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 p=None,
                 **kwargs,
                 ):

        # random robot position
        absolute_obs = self._convert_to_absolute_obs(obs)
        absolute_next_obs = self._convert_to_absolute_obs(next_obs)

        robot_delta = absolute_next_obs[:2] - absolute_obs[:2]
        target_delta = absolute_next_obs[2:4] - absolute_obs[2:4]

        # assert np.allclose(target_delta, np.zeros_like(target_delta))
        if not np.allclose(target_delta, np.zeros_like(target_delta)):
            return None, None, None, None, None

        dist_to_ball = 0
        next_dist_to_ball = 0

        while dist_to_ball < 30 and next_dist_to_ball < 30:
            new_robot_pos = self._sample_robot_pos()
            # new_robot_pos = absolute_obs[:2]
            new_next_robot_pos = new_robot_pos + robot_delta

            dist_to_ball = np.linalg.norm((new_robot_pos - absolute_obs[2:4]))
            next_dist_to_ball = np.linalg.norm((new_next_robot_pos[:2] - absolute_next_obs[2:4]))

        absolute_obs[:2] = new_robot_pos
        absolute_next_obs[:2] = new_next_robot_pos

        aug_obs = self._convert_to_relative_obs(absolute_obs)
        aug_next_obs = self._convert_to_relative_obs(absolute_next_obs)
        aug_action = action # unchanged
        aug_reward, at_goal = self.calculate_reward(absolute_next_obs)
        aug_done = done # unchanged

        # print(aug_obs - obs)
        # print(aug_next_obs - next_obs)
        # print(aug_reward, reward)
        #
        # assert np.allclose(aug_obs, obs)
        # assert np.allclose(aug_next_obs, next_obs)
        # assert np.allclose(aug_reward, reward)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done

