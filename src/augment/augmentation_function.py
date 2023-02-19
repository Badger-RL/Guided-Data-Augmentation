import math

import numpy as np


class BaseAugmentationFunction:

    def __init__(self, env=None, **kwargs):
        self.env = env

    def _deepcopy_transition(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
    ):
        copy_obs = obs.copy()
        copy_next_obs = next_obs.copy()
        copy_action = action.copy()
        copy_reward = reward.copy()
        copy_done = done.copy()

        return copy_obs, copy_next_obs, copy_action, copy_reward, copy_done

    def augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 # terminated: np.ndarray,
                 # truncated: np.ndarray,
                 **kwargs,):

        # copy input transition
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done = \
            self._deepcopy_transition(obs, next_obs, action, reward, done)

        # augment input copy of input transition in-place
        return self._augment(aug_obs, aug_next_obs, aug_action, aug_reward, done, **kwargs)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 terminated: np.ndarray,
                 truncated: np.ndarray,
                 **kwargs,):
        raise NotImplementedError("Augmentation function not implemented.")


class AbstractSimAugmentationFunction(BaseAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.ball_pos_mask = None
        self.robot_pos_mask = None
        self.goal_x = 4800
        self.goal_y = 0
        self.goal = np.array([self.goal_x, self.goal_y])
        self.displacement_coef = 0.2

    def _sample_robot_pos(self, n=1):
        x = np.random.uniform(-3500, 3500)
        y = np.random.uniform(-2500, 2500)
        return np.array([x, y])

    def _sample_robot_angle(self, n=1):
        return np.random.uniform(0, 2 * np.pi, size=(n,))

    def _convert_to_absolute_obs(self, obs):

        target_x = self.goal_x - obs[2]
        target_y = self.goal_y - obs[3]
        robot_x = target_x - obs[0]
        robot_y = target_y - obs[1]

        relative_x = target_x - robot_x
        relative_y = target_y - robot_y
        relative_angle = np.arctan2(relative_y, relative_x)
        if relative_angle < 0:
            relative_angle += 2*np.pi

        relative_angle_minus_robot_angle = np.arctan2(obs[4], obs[5])
        if relative_angle_minus_robot_angle < 0:
            relative_angle_minus_robot_angle += 2*np.pi

        robot_angle = relative_angle - relative_angle_minus_robot_angle
        if robot_angle < 0:
            robot_angle += 2*np.pi

        return np.array([
            robot_x,
            robot_y,
            target_x,
            target_y,
            np.sin(robot_angle),
            np.cos(robot_angle)
        ])

    def _convert_to_relative_obs(self, obs):

        robot_pos = obs[:2]
        target_pos = obs[2:4]

        robot_x = obs[0]
        robot_y = obs[1]
        target_x = obs[2]
        target_y = obs[3]
        relative_x = target_x - robot_x
        relative_y = target_y - robot_y
        relative_angle = np.arctan2(relative_y, relative_x)
        if relative_angle < 0:
            relative_angle += 2*np.pi

        robot_angle = np.arctan2(obs[4], obs[5])
        if robot_angle < 0:
            robot_angle += 2*np.pi

        goal_delta = self.goal - robot_pos
        goal_relative_angle = np.arctan2(goal_delta[1], goal_delta[0])
        if goal_relative_angle < 0:
            goal_relative_angle += 2*np.pi

        return np.concatenate([
            target_pos - robot_pos,
            self.goal - target_pos,
            [np.sin(relative_angle - robot_angle),
            np.cos(relative_angle - robot_angle),
            np.sin(goal_relative_angle - robot_angle),
            np.cos(goal_relative_angle - robot_angle),]
        ])


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

