import gym
import numpy as np
import warnings

from custom_envs.base import BaseEnv

warnings.filterwarnings("ignore")

LENGTH = 500
TRAINING_STEPS = 1000000


class PushBallToGoalEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        observation_space_low = np.array([
            -4800,
            -3000,
            -4800,
            -3000,
            -1,
            -1,
            -1,
            -1
        ])
        observation_space_high = -observation_space_low.copy()
        self.observation_space = gym.spaces.Box(
            observation_space_low, observation_space_high
        )

        self.reset()

    def reset(self):
        self.time = 0

        self.displacement_coef = 0.2

        self.contacted_ball = False

        self.robot_x = np.random.uniform(-3500, 3500)
        self.robot_y = np.random.uniform(-2500, 2500)
        self.robot_angle = np.random.uniform(0, 2 * np.pi)

        self.target_x = np.random.uniform(-2500, 2500)
        self.target_y = np.random.uniform(-2000, 2000)

        self.goal_x = 4800
        self.goal_y = 0

        self.update_goal_value()

        robot_location = np.array([self.robot_x, self.robot_y])
        target_location = np.array([self.target_x, self.target_y])
        self.initial_distance = np.linalg.norm(target_location - robot_location)

        return self._observe_state()


    def _observe_state(self):

        self.update_target_value()
        self.update_goal_value()

        return np.array(
            [
                (self.target_x - self.robot_x),
                (self.target_y - self.robot_y),
                (self.goal_x - self.target_x),
                (self.goal_y - self.target_y),
                np.sin(self.relative_angle - self.robot_angle),
                np.cos(self.relative_angle - self.robot_angle),
                np.sin(self.goal_relative_angle - self.robot_angle),
                np.cos(self.goal_relative_angle - self.robot_angle),
            ]
        )

    def observe_global_state(self):
        return [
            self.robot_x,
            self.robot_y,
            self.target_x,
            self.target_y,
            np.sin(self.robot_angle),
            np.cos(self.robot_angle),
        ]

    def set_abstract_state(self, obs):

        self.target_x = self.goal_x - obs[6]
        self.target_y = self.goal_y - obs[7]
        self.robot_x = self.target_x - obs[4]
        self.robot_y = self.target_y - obs[5]

        relative_x = self.target_x - self.robot_x
        relative_y = self.target_y - self.robot_y
        relative_angle = np.arctan2(relative_y, relative_x)
        if relative_angle < 0:
            relative_angle += 2*np.pi

        relative_angle_minus_robot_angle = np.arctan2(obs[8], obs[9])
        if relative_angle_minus_robot_angle < 0:
            relative_angle_minus_robot_angle += 2*np.pi

        self.robot_angle = relative_angle - relative_angle_minus_robot_angle
        if self.robot_angle < 0:
            self.robot_angle += 2*np.pi

