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

    def _sample_theta(self, aug_abs_obs, aug_abs_next_obs, **kwargs):
        return np.random.uniform(-np.pi / 4, np.pi / 4)

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
        aug_abs_obs[4] += theta

        aug_abs_next_obs[:2] = M.dot(aug_abs_next_obs[:2].T).T
        aug_abs_next_obs[2:4] = M.dot(aug_abs_next_obs[2:4].T).T
        aug_abs_next_obs[4] += theta

        aug_abs_obs[:2] += ball_pos
        aug_abs_obs[2:4] += ball_pos
        aug_abs_next_obs[:2] += ball_pos
        aug_abs_next_obs[2:4] += ball_pos

    def _rotate_agent(self, aug_abs_obs, aug_abs_next_obs, theta):
        ball_pos = aug_abs_obs[:2].copy()
        aug_abs_obs[:2] -= ball_pos
        aug_abs_next_obs[:2] -= ball_pos

        M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        aug_abs_obs[:2] = M.dot(aug_abs_obs[:2].T).T
        aug_abs_obs[4] += theta

        aug_abs_next_obs[:2] = M.dot(aug_abs_next_obs[:2].T).T
        aug_abs_next_obs[4] += theta

        aug_abs_obs[:2] += ball_pos
        aug_abs_next_obs[:2] += ball_pos

    def _reflect(self, aug_abs_obs, aug_abs_next_obs, aug_action):
        aug_abs_obs[1] *= -1
        aug_abs_next_obs[1] *= -1
        aug_abs_obs[3] *= -1
        aug_abs_next_obs[3] *= -1
        aug_abs_obs[4] *= -1
        aug_abs_next_obs[4] *= -1

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
        if np.random.random() < 0.5:
            new_x = np.random.uniform(0, 4700)
            new_y = np.random.uniform(-3400, 3400)
            if np.random.random() < 0.25:
                new_x = np.random.uniform(2000, 4700)
                new_y = np.random.uniform(-1000, 1000)


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


    def _translate_ball(self, aug_abs_obs, aug_abs_next_obs):

        xmin = np.min([aug_abs_obs[2], aug_abs_next_obs[2]])
        ymin = np.min([aug_abs_obs[3], aug_abs_next_obs[3]])
        xmax = np.max([aug_abs_obs[2], aug_abs_next_obs[2]])
        ymax = np.max([aug_abs_obs[3], aug_abs_next_obs[3]])

        # Translate bottom left corner of the righter bounding box containing the robot and ball
        new_x = np.random.uniform(-5000, 5000 - (xmax - xmin))
        new_y = np.random.uniform(-4000, 4000 - (ymax - ymin))
        # np.random.choice(['left_half','3rd_quadrant', '4th_quadrant'], p=[0.1, 0.3, 0.4,])
        if np.random.random() < 0.5:
            new_x = np.random.uniform(0, 4300)
            new_y = np.random.uniform(-3400, 3400)
            if np.random.random() < 0.5:
                new_x = np.random.uniform(2000, 4300)
                new_y = np.random.uniform(-1000, 1000)

        delta_x = new_x - xmin
        delta_y = new_y - ymin

        aug_abs_obs[2] += delta_x
        aug_abs_obs[3] += delta_y

        aug_abs_next_obs[2] += delta_x
        aug_abs_next_obs[3] += delta_y

    def _translate_agent(self, aug_abs_obs, aug_abs_next_obs):

        xmin = np.min([aug_abs_obs[0], aug_abs_next_obs[0]])
        ymin = np.min([aug_abs_obs[1], aug_abs_next_obs[1]])
        xmax = np.max([aug_abs_obs[0], aug_abs_next_obs[0]])
        ymax = np.max([aug_abs_obs[1], aug_abs_next_obs[1]])

        # Translate bottom left corner of the righter bounding box containing the robot and ball
        new_x = np.random.uniform(-4500, 4500 - (xmax - xmin))
        new_y = np.random.uniform(-3000, 3000 - (ymax - ymin))
        # if np.random.random() < 0.2:
        #     new_x = np.random.uniform(4400, 4500)
        #     new_y = np.random.uniform(-500, 500)

        if np.random.random() < 0.5:
            new_x = np.random.uniform(0, 4300)
            new_y = np.random.uniform(-3400, 3400)
            if np.random.random() < 0.5:
                new_x = np.random.uniform(2000, 4300)
                new_y = np.random.uniform(-1000, 1000)

        delta_x = new_x - xmin
        delta_y = new_y - ymin

        aug_abs_obs[0] += delta_x
        aug_abs_obs[1] += delta_y

        aug_abs_next_obs[0] += delta_x
        aug_abs_next_obs[1] += delta_y

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


        theta = self._sample_theta(aug_abs_obs, aug_abs_next_obs)
        self._rotate(aug_abs_obs, aug_abs_next_obs, theta)
        if np.random.random() < 0.5:
            self._reflect(aug_abs_obs, aug_abs_next_obs, aug_action)
        self._translate(aug_abs_obs, aug_abs_next_obs)

        aug_reward, ball_is_at_goal, ball_is_out_of_bounds = self.env.calculate_reward(aug_abs_next_obs)
        aug_done = ball_is_at_goal or ball_is_out_of_bounds
        if ball_is_at_goal:
            aug_abs_next_obs[2:4] = self.goal
        aug_obs = self._convert_to_relative_obs(aug_abs_obs)
        aug_next_obs = self._convert_to_relative_obs(aug_abs_next_obs)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_abs_obs, aug_abs_next_obs