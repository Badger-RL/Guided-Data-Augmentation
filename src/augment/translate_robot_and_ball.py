import numpy as np

from GuidedDataAugmentationForRobotics.src.augment.augmentation_function import AbstractSimAugmentationFunction


class TranslateRobotAndBall(AbstractSimAugmentationFunction):
    '''
    Translate the robot and ball by the same (delta_x, delta_y).
    '''
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_robot_pos(self, n=1):
        x = np.random.uniform(-3500, 3500)
        y = np.random.uniform(-2500, 2500)
        return np.array([x, y])

    def _sample_robot_angle(self, n=1):
        return np.random.uniform(0, 2 * np.pi, size=(n,))

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

        # Only augment if robot kicked the ball.
        # if np.allclose(target_delta, np.zeros_like(target_delta)):
        #     return None, None, None, None, None

        dist_to_ball = 0
        next_dist_to_ball = 0

        xmin = np.min([absolute_obs[0], absolute_next_obs[0], absolute_obs[2], absolute_next_obs[2]])
        ymin = np.min([absolute_obs[1], absolute_next_obs[1], absolute_obs[3], absolute_next_obs[3]])
        xmax = np.max([absolute_obs[0], absolute_next_obs[0], absolute_obs[2], absolute_next_obs[2]])
        ymax = np.max([absolute_obs[1], absolute_next_obs[1], absolute_obs[3], absolute_next_obs[3]])

        # Translate bottom left corner of the righter bounding box containing the robot and ball
        new_x = np.random.uniform(-4500, 4500-xmax)
        new_y = np.random.uniform(-3000, 3000-ymax)

        delta_x = new_x - xmin
        delta_y = new_y - ymin

        new_robot_pos = np.array
        # new_next_robot_pos = new_robot_pos + robot_delta

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
        aug_reward = reward
        aug_done = done

        # print(aug_obs - obs)
        # print(aug_next_obs - next_obs)
        #
        # assert np.allclose(aug_obs, obs)
        # assert np.allclose(aug_next_obs, next_obs)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done

        # change in robot position won't change reward nor done