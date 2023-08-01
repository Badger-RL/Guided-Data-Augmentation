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


    def _sample_theta_for_kicking(self, aug_abs_obs, aug_abs_next_obs, **kwargs):

        delta_ball = aug_abs_next_obs[2:4] - aug_abs_obs[2:4]
        ball_pos = aug_abs_obs[2:4]
        # ball near corner -- change guide
        if ball_pos[0] > 2000 and np.abs(ball_pos[1]) < 1000:
            goal = np.array([4500, 0])
        else:
            goal = np.array([3400, 0])
        # goal = np.array([4400, 3000])

        delta_ball_theta = np.arctan2(delta_ball[1], delta_ball[0])

        delta_ball_to_goal = goal - aug_abs_obs[2:4]  # guide theta
        ball_to_goal_theta = np.arctan2(delta_ball_to_goal[1], delta_ball_to_goal[0])
        theta = ball_to_goal_theta - delta_ball_theta

        return theta.squeeze()

    def _sample_theta_for_walking(self, aug_abs_obs, aug_abs_next_obs, **kwargs):

        delta_robot = aug_abs_next_obs[:2] - aug_abs_obs[:2]
        robot_pos = aug_abs_obs[:2]
        ball_pos = aug_abs_obs[2:4]

        delta_robot_theta = np.arctan2(delta_robot[1], delta_robot[0])

        delta_robot_to_ball = ball_pos - robot_pos  # guide theta
        robot_to_ball_theta = np.arctan2(delta_robot_to_ball[1], delta_robot_to_ball[0])
        theta = robot_to_ball_theta - delta_robot_theta

        return theta.squeeze()

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
            return False

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

        return True

    def _rotate_agent_and_ball(self, aug_abs_obs, aug_abs_next_obs, theta):
        robot_pos = aug_abs_obs[2:4].copy()
        aug_abs_obs[:2] -= robot_pos
        aug_abs_obs[2:4] -= robot_pos
        aug_abs_next_obs[:2] -= robot_pos
        aug_abs_next_obs[2:4] -= robot_pos

        M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        aug_abs_obs[:2] = M.dot(aug_abs_obs[:2].T).T
        aug_abs_obs[2:4] = M.dot(aug_abs_obs[2:4].T).T
        aug_abs_obs[4] += theta

        aug_abs_next_obs[:2] = M.dot(aug_abs_next_obs[:2].T).T
        aug_abs_next_obs[2:4] = M.dot(aug_abs_next_obs[2:4].T).T
        aug_abs_next_obs[4] += theta

        aug_abs_obs[:2] += robot_pos
        aug_abs_obs[2:4] += robot_pos
        aug_abs_next_obs[:2] += robot_pos
        aug_abs_next_obs[2:4] += robot_pos

    def augment(self,
                abs_obs: np.ndarray,
                abs_next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                new_pos=None,
                **kwargs,
                ):

        # if not self._is_valid_input(abs_obs, abs_next_obs):
        #     return None, None, None, None, None, None, None

        aug_abs_obs, aug_abs_next_obs, aug_action, aug_reward, aug_done = \
            self._deepcopy_transition(abs_obs, abs_next_obs, action, reward, done)

        if self.at_goal(abs_obs[2], abs_obs[3]):
            new_x = np.random.uniform(4400, 4700)
            new_y = np.random.uniform(-500, 500)
            aug_abs_obs[2] = new_x
            aug_abs_obs[3] = new_y
            aug_abs_next_obs[2] = new_x
            aug_abs_next_obs[3] = new_y

        else:
            # if agent kicked the ball, rotate both agent and ball. Otherwise, only rotate the agent.
            delta_ball = abs_next_obs[2:4] - abs_obs[2:4]
            dist_ball = np.linalg.norm(delta_ball)
            if dist_ball > 1e-4:
                super()._translate(aug_abs_obs, aug_abs_next_obs)
                theta = self._sample_theta_for_kicking(aug_abs_obs, aug_abs_next_obs)
                self._rotate_agent_and_ball(aug_abs_obs, aug_abs_next_obs, theta)
            else:
                while True:
                    super()._translate_agent(aug_abs_obs, aug_abs_next_obs)
                    super()._translate_ball(aug_abs_obs, aug_abs_next_obs)
                    dist_robot_to_ball = np.linalg.norm(aug_abs_obs[:2] - aug_abs_obs[2:4])
                    if dist_robot_to_ball > (self.env.robot_radius + self.env.ball_radius) * 6:
                        break
                theta = self._sample_theta_for_walking(aug_abs_obs, aug_abs_next_obs)
                self._rotate_agent(aug_abs_obs, aug_abs_next_obs, theta)

        # assign reward and done signal
        aug_reward, ball_is_at_goal, ball_is_out_of_bounds = self.env.calculate_reward(aug_abs_next_obs)
        aug_done = ball_is_out_of_bounds

        # convert absolute observation back to relative
        robot_pos = aug_abs_obs[:2]
        ball_pos = aug_abs_obs[2:4]
        robot_angle = aug_abs_obs[4]
        aug_obs = self.env.get_obs(robot_pos, ball_pos, robot_angle)

        robot_pos = aug_abs_next_obs[:2]
        ball_pos = aug_abs_next_obs[2:4]
        robot_angle = aug_abs_next_obs[4]
        aug_next_obs = self.env.get_obs(robot_pos, ball_pos, robot_angle)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_abs_obs, aug_abs_next_obs
