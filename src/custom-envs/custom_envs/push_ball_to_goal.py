import gym
import numpy as np


import warnings

from custom_envs.base import BaseEnv

warnings.filterwarnings("ignore")

LENGTH = 500
TRAINING_STEPS = 1000000


class PushBallToGoalEnv(BaseEnv):
    def __init__(self, robot_x_range = [-4500,4500], robot_y_range = [-3000,3000], ball_x_range = [-4500,4500], ball_y_range = [-3000,3000]):
        
        
        super().__init__()


        """
        OBSERVATION SPACE:
            - x-cordinate of robot with respect to target
            - y-cordinate of robot with respect to target
            - sin(Angle between robot and target)
            - cos(Angle between robot and target)
        """
    
        self.robot_x_range = robot_x_range
        self.robot_y_range = robot_y_range
        self.ball_x_range = ball_x_range
        self.ball_y_range = ball_y_range


        observation_space_size = 8


        observation_space_low = -1 * np.ones(observation_space_size)
        observation_space_high = np.ones(observation_space_size)
        self.observation_space = gym.spaces.Box(
            observation_space_low, observation_space_high
        )

        self.reset()

    def reset(self):
        self.time = 0

        self.displacement_coef = 0.2

        self.contacted_ball = False


        self.robot_x = np.random.uniform(self.robot_x_range[0], self.robot_x_range[1])
        self.robot_y = np.random.uniform(self.robot_y_range[0], self.robot_y_range[1])
        self.robot_angle = np.random.uniform(0, 2 * np.pi)

        self.target_x = np.random.uniform(self.ball_x_range[0], self.ball_x_range[1])
        self.target_y = np.random.uniform(self.ball_y_range[0], self.ball_y_range[1])

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
                (self.target_x - self.robot_x) / 9000,
                (self.target_y - self.robot_y) / 6000,
                (self.goal_x - self.target_x) / 9000,
                (self.goal_y - self.target_y) / 6000,
                np.sin(self.relative_angle - self.robot_angle),
                np.cos(self.relative_angle - self.robot_angle),
                np.sin(self.goal_relative_angle - self.robot_angle),
                np.cos(self.goal_relative_angle - self.robot_angle),
            ]
        )

    def _observe_global_state(self):
        return [
            self.robot_x / 9000,
            self.robot_y / 6000,
            self.target_x / 9000,
            self.target_y / 6000,
            np.sin(self.robot_angle),
            np.cos(self.robot_angle),
        ]

    def set_abstract_state(self, obs):

        # obs[0] = tx - rx --> rx = tx - obs[0]
        # obs[2] = gx - tx --> tx = gx - obs[2]

        self.target_x = self.goal_x - obs[2]*9000
        self.target_y = self.goal_y - obs[3]*6000
        self.robot_x = self.target_x - obs[0]*9000
        self.robot_y = self.target_y - obs[1]*6000

        relative_x = self.target_x - self.robot_x
        relative_y = self.target_y - self.robot_y
        relative_angle = np.arctan2(relative_y, relative_x)
        if relative_angle < 0:
            relative_angle += 2*np.pi

        relative_angle_minus_robot_angle = np.arctan2(obs[4], obs[5])
        if relative_angle_minus_robot_angle < 0:
            relative_angle_minus_robot_angle += 2*np.pi

        self.robot_angle = relative_angle - relative_angle_minus_robot_angle
        if self.robot_angle < 0:
            self.robot_angle += 2*np.pi

