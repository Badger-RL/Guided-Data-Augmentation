import copy

import numpy as np

from src.augment.abstractsim.augmentation_function import AbstractSimAugmentationFunction


class RotateReflectTranslate(AbstractSimAugmentationFunction):
    '''
    Translate the robot and ball by the same (delta_x, delta_y).
    '''

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _is_valid_input(self, abs_obs, abs_next_obs):
        if self.at_goal(abs_obs[2], abs_obs[3]) or self.at_goal(abs_next_obs[2], abs_next_obs[3]):
            return False
        return True

    def _rotate(self, aug_abs_obs, aug_abs_next_obs):
        theta = np.random.uniform(-np.pi / 4, np.pi / 4)

        M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        aug_abs_obs[:2] = M.dot(aug_abs_obs[:2].T).T
        aug_abs_obs[2:4] = M.dot(aug_abs_obs[2:4].T).T
        aug_abs_obs[5] += theta

        robot_angle = aug_abs_obs[4] + theta
        if robot_angle < 0:
            robot_angle += 2 * np.pi
        aug_abs_obs[4] += theta

        aug_abs_next_obs[:2] = M.dot(aug_abs_next_obs[:2].T).T
        aug_abs_next_obs[2:4] = M.dot(aug_abs_next_obs[2:4].T).T
        aug_abs_next_obs[5] += theta

        next_robot_angle = aug_abs_next_obs[4] + theta
        if next_robot_angle < 0:
            next_robot_angle += 2 * np.pi
        aug_abs_next_obs[4] += theta

    def _reflect(self, aug_abs_obs, aug_abs_next_obs, aug_action):
        aug_abs_obs[1] *= -1
        aug_abs_next_obs[1] *= -1
        aug_abs_obs[3] *= -1
        aug_abs_next_obs[3] *= -1
        aug_abs_obs[4] *= -1
        aug_abs_next_obs[4] *= -1
        aug_abs_obs[6] += np.pi
        aug_abs_next_obs[6] += np.pi

        aug_action[0] *= -1
        aug_action[1] *= 1
        aug_action[2] *= -1

    def _translate(self, aug_abs_obs, aug_abs_next_obs):

        xmin = np.min([aug_abs_obs[0], aug_abs_next_obs[0], aug_abs_obs[2], aug_abs_next_obs[2]])
        ymin = np.min([aug_abs_obs[1], aug_abs_next_obs[1], aug_abs_obs[3], aug_abs_next_obs[3]])
        xmax = np.max([aug_abs_obs[0], aug_abs_next_obs[0], aug_abs_obs[2], aug_abs_next_obs[2]])
        ymax = np.max([aug_abs_obs[1], aug_abs_next_obs[1], aug_abs_obs[3], aug_abs_next_obs[3]])

        # Translate bottom left corner of the righter bounding box containing the robot and ball
        new_x = np.random.uniform(-4500, 4500 - (xmax - xmin))
        new_y = np.random.uniform(-3000, 3000 - (ymax - ymin))

        delta_x = new_x - xmin
        delta_y = new_y - ymin

        aug_abs_obs[0] += delta_x
        aug_abs_obs[1] += delta_y
        aug_abs_obs[2] += delta_x
        aug_abs_obs[3] += delta_y

        aug_abs_next_obs[0] += delta_x
        aug_abs_next_obs[1] += delta_y
        aug_abs_next_obs[2] += delta_x
        aug_abs_next_obs[3] += delta_y

    def augment(self,
                abs_obs: np.ndarray,
                abs_next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,
                ):

        if not self._is_valid_input(abs_obs, abs_next_obs):
            return None, None, None, None, None, None, None

        aug_abs_obs, aug_abs_next_obs, aug_action, aug_reward, aug_done = \
            self._deepcopy_transition(abs_obs, abs_next_obs, action, reward, done)

        self._rotate(aug_abs_obs, aug_abs_next_obs)
        if np.random.random() < 0.5:
            self._reflect(aug_abs_obs, aug_abs_next_obs, aug_action)
        self._translate(aug_abs_obs, aug_abs_next_obs)

        aug_reward = self.env.calculate_reward_2(aug_abs_obs, aug_abs_next_obs)
        aug_obs = self._convert_to_relative_obs(aug_abs_obs)
        aug_next_obs = self._convert_to_relative_obs(aug_abs_next_obs)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_abs_obs, aug_abs_next_obs


class RotateReflectTranslateGuided(AbstractSimAugmentationFunction):
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

        aug_absolute_obs = copy.deepcopy(absolute_obs)
        aug_absolute_next_obs = copy.deepcopy(absolute_next_obs)
        aug_action = action.copy()
        aug_done = done.copy()

        xmin = np.min([aug_absolute_obs[0], aug_absolute_next_obs[0], aug_absolute_obs[2], aug_absolute_next_obs[2]])
        ymin = np.min([aug_absolute_obs[1], aug_absolute_next_obs[1], aug_absolute_obs[3], aug_absolute_next_obs[3]])
        xmax = np.max([aug_absolute_obs[0], aug_absolute_next_obs[0], aug_absolute_obs[2], aug_absolute_next_obs[2]])
        ymax = np.max([aug_absolute_obs[1], aug_absolute_next_obs[1], aug_absolute_obs[3], aug_absolute_next_obs[3]])

        # Translate bottom left corner of the righter bounding box containing the robot and ball
        new_x = np.random.uniform(-4500, 4500 - (xmax - xmin))
        new_y = np.random.uniform(-3000, 3000 - (ymax - ymin))

        delta_x = new_x - xmin
        delta_y = new_y - ymin

        aug_absolute_obs[0] += delta_x
        aug_absolute_obs[1] += delta_y
        aug_absolute_obs[2] += delta_x
        aug_absolute_obs[3] += delta_y

        aug_absolute_next_obs[0] += delta_x
        aug_absolute_next_obs[1] += delta_y
        aug_absolute_next_obs[2] += delta_x
        aug_absolute_next_obs[3] += delta_y

        if np.random.random() < 0.5:
            aug_absolute_obs[1] *= -1
            aug_absolute_next_obs[1] *= -1
            aug_absolute_obs[3] *= -1
            aug_absolute_next_obs[3] *= -1
            aug_absolute_obs[4] *= -1
            aug_absolute_next_obs[4] *= -1

            aug_action[0] *= -1
            aug_action[1] *= 1
            aug_action[2] *= -1

        delta_ball = aug_absolute_next_obs[2:4] - aug_absolute_obs[2:4]
        dist_ball = np.linalg.norm(delta_ball)
        if dist_ball > 1e-4:
            delta_ball_theta = np.arctan2(delta_ball[1], delta_ball[0])

            delta_ball_to_goal = self.goal - aug_absolute_obs[2:4]
            ball_to_goal_theta = np.arctan2(delta_ball_to_goal[1], delta_ball_to_goal[0])

            theta = ball_to_goal_theta - delta_ball_theta
        else:
            theta = np.random.uniform(-np.pi / 4, np.pi / 4)

        ball_pos = aug_absolute_obs[2:4].copy()
        aug_absolute_obs[:2] -= ball_pos
        aug_absolute_obs[2:4] -= ball_pos
        aug_absolute_next_obs[:2] -= ball_pos
        aug_absolute_next_obs[2:4] -= ball_pos

        M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        aug_absolute_obs[:2] = M.dot(aug_absolute_obs[:2].T).T
        aug_absolute_obs[2:4] = M.dot(aug_absolute_obs[2:4].T).T

        robot_angle = aug_absolute_obs[4] + theta
        if robot_angle < 0:
            robot_angle += 2 * np.pi
        aug_absolute_obs[4] += theta

        aug_absolute_next_obs[:2] = M.dot(aug_absolute_next_obs[:2].T).T
        aug_absolute_next_obs[2:4] = M.dot(aug_absolute_next_obs[2:4].T).T

        next_robot_angle = aug_absolute_next_obs[4] + theta
        if next_robot_angle < 0:
            next_robot_angle += 2 * np.pi
        aug_absolute_next_obs[4] += theta

        aug_absolute_obs[:2] += ball_pos
        aug_absolute_obs[2:4] += ball_pos
        aug_absolute_next_obs[:2] += ball_pos
        aug_absolute_next_obs[2:4] += ball_pos

        aug_reward, _ = self.calculate_reward(aug_absolute_next_obs)

        aug_obs = self._convert_to_relative_obs(aug_absolute_obs)
        aug_next_obs = self._convert_to_relative_obs(aug_absolute_next_obs)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done
