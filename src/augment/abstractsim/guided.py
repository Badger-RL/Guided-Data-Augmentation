import copy

import numpy as np

from augment.abstractsim.random import RotateReflectTranslate
from src.augment.abstractsim.augmentation_function import AbstractSimAugmentationFunction


class RotateReflectTranslateGuided(RotateReflectTranslate):
    '''
    Translate the robot and ball by the same (delta_x, delta_y).
    '''

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
    '''
    
                if ball_pos[0] > 1000 and ball_pos[1] > 1000:
                    goal = np.array([2000, 1000])
                elif ball_pos[0] > 1000 and ball_pos[1] < -1000:
                    goal = np.array([2000, -1000])
                else:
                    goal = self.goal
    '''
    def _sample_theta(self, aug_abs_obs, aug_abs_next_obs, **kwargs):

        delta_ball = aug_abs_next_obs[2:4] - aug_abs_obs[2:4]
        dist_ball = np.linalg.norm(delta_ball)
        if dist_ball > 1e-4:
            ball_pos = aug_abs_next_obs[2:4]
            # ball near corner -- change guide
            if ball_pos[0] > 2000 and np.abs(ball_pos[1]) < 1000:
                goal = self.goal
            else:
                goal = np.array([2000, 0])

            delta_ball_theta = np.arctan2(delta_ball[1], delta_ball[0])

            delta_ball_to_goal = goal - aug_abs_obs[2:4] # guide theta
            ball_to_goal_theta = np.arctan2(delta_ball_to_goal[1], delta_ball_to_goal[0])

            theta = ball_to_goal_theta - delta_ball_theta
        else:
            theta = super()._sample_robot_angle()

        return theta.squeeze()

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

        self._translate(aug_abs_obs, aug_abs_next_obs)

        # aug_abs_obs[2] = 0
        # aug_abs_next_obs[2] = -500
        #
        # aug_abs_obs[3] = 0
        # aug_abs_next_obs[3] = 0

        theta = self._sample_theta(aug_abs_obs, aug_abs_next_obs)
        self._rotate(aug_abs_obs, aug_abs_next_obs, theta)
        # if np.random.random() < 0.5:
        #     self._reflect(aug_abs_obs, aug_abs_next_obs, aug_action)

        aug_reward = self.env.calculate_reward_2(aug_abs_obs, aug_abs_next_obs)
        aug_obs = self._convert_to_relative_obs(aug_abs_obs)
        aug_next_obs = self._convert_to_relative_obs(aug_abs_next_obs)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_abs_obs, aug_abs_next_obs
