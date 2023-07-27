import copy

import numpy as np

from augment.abstractsim.random import RotateReflectTranslate
from augment.utils import is_in_bounds, is_at_goal
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
        delta_robot = aug_abs_next_obs[:2] - aug_abs_obs[:2]
        dist_robot = np.linalg.norm(delta_ball)

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
            theta += np.random.uniform(-np.pi/12, +np.pi/12)
        # elif
        else:
            theta = super()._sample_robot_angle()

        return theta.squeeze()

    # def _sample_agent_theta(self, aug_abs_obs, aug_abs_next_obs, **kwargs):
    #
    #     delta_robot = aug_abs_next_obs[:2] - aug_abs_obs[:2]
    #     dist_robot = np.linalg.norm(delta_ball)
    #
    #     ball_pos = aug_abs_next_obs[2:4]
    #
    #     delta_ball_to_goal = self.goal - aug_abs_obs[2:4]  # guide theta
    #     ball_to_goal_theta = np.arctan2(delta_ball_to_goal[1], delta_ball_to_goal[0])
    #
    #     goal = np.array([2000, 0])
    #     goal = ball_pos
    #
    #     # delta_ball_theta = np.arctan2(delta_ball[1], delta_ball[0])
    #
    #
    #
    #     theta = ball_to_goal_theta - delta_ball_theta
    #     theta += np.random.uniform(-np.pi / 12, +np.pi / 12)
    #     # elif
    #     else:
    #         theta = super()._sample_robot_angle()
    #
    #     return theta.squeeze()

    def _translate_to_position(self, aug_abs_obs, aug_abs_next_obs, new_pos):
        new_x, new_y = new_pos[0], new_pos[1]

        xmin = np.min([aug_abs_obs[0], aug_abs_next_obs[0], aug_abs_obs[2], aug_abs_next_obs[2]])
        ymin = np.min([aug_abs_obs[1], aug_abs_next_obs[1], aug_abs_obs[3], aug_abs_next_obs[3]])
        xmax = np.max([aug_abs_obs[0], aug_abs_next_obs[0], aug_abs_obs[2], aug_abs_next_obs[2]])
        ymax = np.max([aug_abs_obs[1], aug_abs_next_obs[1], aug_abs_obs[3], aug_abs_next_obs[3]])

        # Translate bottom left corner of the righter bounding box containing the robot and ball
        if np.abs(new_y) < 500:
            max_x = 4800 - (xmax - xmin)
            max_y = 3000 - (ymax - ymin)
        else:
            max_x = 4500 - (xmax - xmin)
            max_y = 3000 - (ymax - ymin)

        if new_pos[0] > max_x or new_pos[1] > max_y:
            return None

        delta_pos = new_pos - aug_abs_obs[2:4]
        delta_x = delta_pos[0]
        delta_y = delta_pos[1]

        aug_abs_obs[0] += delta_x
        aug_abs_obs[1] += delta_y
        aug_abs_obs[2] += delta_x
        aug_abs_obs[3] += delta_y

        aug_abs_next_obs[0] += delta_x
        aug_abs_next_obs[1] += delta_y
        aug_abs_next_obs[2] += delta_x
        aug_abs_next_obs[3] += delta_y


    def _rotate(self, aug_abs_obs, aug_abs_next_obs, theta):
        ball_pos = aug_abs_obs[2:4].copy()
        aug_abs_obs[:2] -= ball_pos
        aug_abs_obs[2:4] -= ball_pos
        aug_abs_next_obs[:2] -= ball_pos
        aug_abs_next_obs[2:4] -= ball_pos

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

        aug_abs_obs[:2] += ball_pos
        aug_abs_obs[2:4] += ball_pos
        aug_abs_next_obs[:2] += ball_pos
        aug_abs_next_obs[2:4] += ball_pos

    #
    # def _rotate_agent(self, aug_abs_obs, aug_abs_next_obs, theta):
    #     robot_pos = aug_abs_obs[:2].copy()
    #     aug_abs_obs[:2] -= robot_pos
    #     aug_abs_next_obs[:2] -= robot_pos
    #
    #     M = np.array([
    #         [np.cos(theta), -np.sin(theta)],
    #         [np.sin(theta), np.cos(theta)]
    #     ])
    #     aug_abs_obs[:2] = M.dot(aug_abs_obs[:2].T).T
    #
    #     robot_angle = aug_abs_obs[4] + theta
    #     if robot_angle < 0:
    #         robot_angle += 2 * np.pi
    #     aug_abs_obs[4] += theta
    #
    #     aug_abs_next_obs[:2] = M.dot(aug_abs_next_obs[:2].T).T
    #
    #     next_robot_angle = aug_abs_next_obs[4] + theta
    #     if next_robot_angle < 0:
    #         next_robot_angle += 2 * np.pi
    #     aug_abs_next_obs[4] += theta
    #
    #     aug_abs_obs[:2] += robot_pos
    #     aug_abs_next_obs[:2] += robot_pos


    def augment(self,
                abs_obs: np.ndarray,
                abs_next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                new_pos=None,
                **kwargs,
                ):

        aug_abs_obs, aug_abs_next_obs, aug_action, aug_reward, aug_done = \
            self._deepcopy_transition(abs_obs, abs_next_obs, action, reward, done)

        if self.at_goal(abs_obs[2], abs_obs[3]) or self.at_goal(abs_next_obs[2], abs_next_obs[3]):
            jitter = np.random.uniform(-10,10, size=(2,))
            aug_abs_obs[:2] += jitter
            aug_abs_next_obs[:2] += jitter
        else:
            delta_ball = aug_abs_next_obs[2:4] - aug_abs_obs[2:4]
            dist_ball = np.linalg.norm(delta_ball)
            if dist_ball < 1e-4:
                return None, None, None, None, None, None, None
            # print(new_pos)
            if new_pos is not None and is_in_bounds(new_pos[0], new_pos[1]):
                self._translate_to_position(aug_abs_obs, aug_abs_next_obs, new_pos)
                theta = self._sample_theta(aug_abs_obs, aug_abs_next_obs)
                self._rotate(aug_abs_obs, aug_abs_next_obs, theta)
            else:
                super()._translate(aug_abs_obs, aug_abs_next_obs)

            # aug_abs_obs[2] = 0
            # aug_abs_next_obs[2] = -500
            #
            # aug_abs_obs[3] = 0
            # aug_abs_next_obs[3] = 0


            # if np.random.random() < 0.5:
            #     self._reflect(aug_abs_obs, aug_abs_next_obs, aug_action)

        # ball_x, ball_y = aug_abs_next_obs[2], aug_abs_next_obs[3]
        # if not is_in_bounds(ball_x, ball_y) and not is_at_goal(ball_x, ball_y):
        #     aug_done = True
        # else:
        #     aug_done = False
        aug_reward, aug_done = self.env.calculate_reward_2(aug_abs_obs, aug_abs_next_obs)
        aug_obs = self._convert_to_relative_obs(aug_abs_obs)
        aug_next_obs = self._convert_to_relative_obs(aug_abs_next_obs)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_abs_obs, aug_abs_next_obs
