import math
import gym
import pygame
import numpy as np
import torch
import time
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import warnings
from src.utils.utils import save_vec_normalize_data

warnings.filterwarnings("ignore")

LENGTH = 500
TRAINING_STEPS = 1000000

'''
Base abstractsim environment
'''
class BaseEnv(gym.Env):

    EPISODE_LENGTH = LENGTH

    def __init__(self,robot_x_range = [-4500,4500], robot_y_range = [-3000,3000], ball_x_range = [-4500,4500], ball_y_range = [-3000,3000]):
        self.rendering_init = False

        """
        ACTION SPACE:
            - Angle to turn clockwise
            - Translation along x-axis
            - Translation along y-axis
        """
        action_space_low = np.array([-np.pi / 2, -1, -1])
        action_space_high = np.array([np.pi / 2, 1, 1])
        self.action_space = gym.spaces.Box(action_space_low, action_space_high)
        """
        OBSERVATION SPACE:
            - x-cordinate of robot with respect to target
            - y-cordinate of robot with respect to target
            - sin(Angle between robot and target)
            - cos(Angle between robot and target)
        """
        observation_space_size = 8
        observation_space_low = -1 * np.ones(observation_space_size)
        observation_space_high = np.ones(observation_space_size)
        self.observation_space = gym.spaces.Box(
            observation_space_low, observation_space_high
        )

        self.robot_x_range = robot_x_range
        self.robot_y_range = robot_y_range
        self.ball_x_range = ball_x_range
        self.ball_y_range = ball_y_range

        self.target_radius = 10
        self.robot_radius = 20


        
       

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

    def reset(self):
        self.time = 0

        self.displacement_coef = 0.2

        self.contacted_ball = False

        self.robot_x = np.random.uniform(-3500, 3500)
        self.robot_y = np.random.uniform(-2500, 2500)
        self.robot_angle = np.random.uniform(0, 2 * np.pi)

        self.target_x = np.random.uniform(-2500, 2500)
        self.target_y = np.random.uniform(-2000, 2000)

        self.goal_x = 4500
        self.goal_y = 0

        self.update_goal_value()
        self.update_target_value()

        robot_location = np.array([self.robot_x, self.robot_y])
        target_location = np.array([self.target_x, self.target_y])
        self.initial_distance = np.linalg.norm(target_location - robot_location)

        return self._observe_state()

    def out_of_bounds(self):
        # Top/Bottom -line out
        if self.robot_y < -3000 or self.robot_y > 3000:
            return True

        # Right -line out
        if self.robot_x > 4500:
            return True

        # Left -line out(upper/goal area)
        if self.robot_x < -4500 and self.robot_y < -750:
            return True

        # Left -line out(under/goal area)
        if self.robot_x < -4500 and self.robot_y > 750:
            return True

        return False

    def get_distance_target_goal(self):
        target_location = np.array([self.target_x, self.target_y])
        goal_location = np.array([self.goal_x, self.goal_y])
        return np.linalg.norm(goal_location - target_location)
    
    def check_facing_ball(self):
        # Normalize angle to 0-360
        robot_angle = math.degrees(self.robot_angle) % 360

        # Find the angle between the robot and the ball
        angle_to_ball = math.degrees(
            math.atan2(self.target_y - self.robot_y, self.target_x - self.robot_x)
        )
        if angle_to_ball < 0:
            angle_to_ball += 360
        # Check if the robot is facing the ball
        if abs(angle_to_ball - robot_angle) < 30:
            return True
        else:
            return False

    def calculate_reward(self):
        robot_location = np.array([self.robot_x, self.robot_y])
        target_location = np.array([self.target_x, self.target_y])
        # Find distance between robot and target
        distance_robot_target = np.linalg.norm(target_location - robot_location)
        reward = 0

        if self.check_facing_ball():
            reward_dist_to_ball = 1/distance_robot_target
            reward_dist_to_goal = 1/self.get_distance_target_goal()
            reward = 0.9*reward_dist_to_goal + 0.1*reward_dist_to_ball

        if self.at_goal():
            reward += 1

        return reward

    def at_goal(self):
        at_goal = False
        if self.target_x > 4500:
            if self.target_y < 750 and self.target_y > -750:
                at_goal = True

        return at_goal

    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    # Return true if line segments AB and CD intersect, for goal line
    def intersect(self, A, B, C, D):
        def ccw(A,B,C):
            return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    def kick_ball(self):
        if self.check_facing_ball():
            robot_location = np.array([self.robot_x, self.robot_y])
            target_location = np.array([self.target_x, self.target_y])

            # Find distance between robot and target
            distance_robot_target = np.linalg.norm(target_location - robot_location)

            # If robot is close enough to ball, kick ball
            if distance_robot_target < (self.robot_radius + self.target_radius) * 10:
                self.contacted_ball = True

                # Update Relative Angle
                self.delta_x = self.target_x - self.robot_x
                self.delta_y = self.target_y - self.robot_y
                self.theta_radians = np.arctan2(self.delta_y, self.delta_x)

                self.target_x = (
                    self.target_x
                    + np.cos(self.theta_radians)
                    * (self.robot_radius + self.target_radius)
                    * 80
                )
                self.target_y = (
                    self.target_y
                    + np.sin(self.theta_radians)
                    * (self.robot_radius + self.target_radius)
                    * 80
                )

                # Check if line of ball intersects goal line
                A = self.Point(self.target_x, self.target_y)
                B = self.Point(self.robot_x, self.robot_y)
                C = self.Point(-4500, -750)
                D = self.Point(-4500, 750)

                # Put ball at goal location (-4500, 0)
                if self.intersect(A, B, C, D):
                    self.target_x = -4501
                    self.target_y = 0

    def set_state_from_obs(self, obs):

        target_x = (self.goal_x - obs[2]*9000)
        target_y = (self.goal_y - obs[3]*6000)
        robot_x = (target_x - obs[0]*9000)
        robot_y = (target_y - obs[1]*6000)

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

        self.target_x = target_x
        self.target_y = target_y
        self.robot_x = robot_x
        self.robot_y = robot_y
        self.robot_angle = robot_angle        



    def move_robot(self, action):
        # Find location policy is trying to reach

        r = np.linalg.norm(action[1:])
        if r > 1:
            action[1:] /= r
        x = action[1]
        y = action[2]

        policy_target_x = self.robot_x + (
                (
                        (np.cos(self.robot_angle) * np.clip(x, -1, 1))
                        + (np.cos(self.robot_angle + np.pi / 2) * np.clip(y, -1, 1))
                )
                * 200
        )  # the x component of the location targeted by the high level action
        policy_target_y = self.robot_y + (
                (
                        (np.sin(self.robot_angle) * np.clip(x, -1, 1))
                        + (np.sin(self.robot_angle + np.pi / 2) * np.clip(y, -1, 1))
                )
                * 200
        )  # the y component of the location targeted by the high level action

        # Update robot position
        self.robot_x = (
            self.robot_x * (1 - self.displacement_coef)
            + policy_target_x * self.displacement_coef
        )  # weighted sums based on displacement coefficient
        self.robot_y = (
            self.robot_y * (1 - self.displacement_coef)
            + policy_target_y * self.displacement_coef
        )  # the idea is we move towards the target position and angle


        self.position_rule()

        # Find distance between robot and target
        robot_location = np.array([self.robot_x, self.robot_y])
        target_location = np.array([self.target_x, self.target_y])
        distance_robot_target = np.linalg.norm(target_location - robot_location)

        # push ball, if collision with robot
        if distance_robot_target < (self.robot_radius + self.target_radius) * 5:
            # changed to reached reward mode
            self.contacted_ball = True

            # Update Relative Angle
            self.delta_x = self.target_x - self.robot_x
            self.delta_y = self.target_y - self.robot_y
            self.theta_radians = np.arctan2(self.delta_y, self.delta_x)

            self.target_x = (
                self.target_x
                + np.cos(self.theta_radians)
                * (self.robot_radius + self.target_radius)
                * 5
            )
            self.target_y = (
                self.target_y
                + np.sin(self.theta_radians)
                * (self.robot_radius + self.target_radius)
                * 5
            )

            # Check if line of ball intersects goal line
            A = self.Point(self.target_x, self.target_y)
            B = self.Point(self.robot_x, self.robot_y)
            C = self.Point(-4500, -750)
            D = self.Point(-4500, 750)

            # Put ball at goal location (-4500, 0)
            if self.intersect(A, B, C, D):
                self.target_x = -4501
                self.target_y = 0

        # Turn towards the target as suggested by the policy
        self.robot_angle = self.robot_angle + self.displacement_coef * action[0]

    def step(self, action):

        self.time += 1

        self.move_robot(action)

        self.update_target_value()
        self.update_goal_value()

        done = False
        reward = self.calculate_reward()

        new_obs = self._observe_state()

        return new_obs, reward, done, {'is_success': self.at_goal()}

    # returns the state of the environment, with global angles and coordinates.

    def _observe_global_state(self):
        return [
            self.robot_x / 9000,
            self.robot_y / 6000,
            np.sin(self.robot_angle),
            np.cos(self.robot_angle),
        ]

    def get_abstract_state(self):
        return np.array(self._observe_global_state())

    def set_abstract_state(self, abstract_state):

        self.robot_x = abstract_state[0] * 9000
        self.robot_y = abstract_state[1] * 6000
        self.target_x = abstract_state[2] * 9000
        self.target_y = abstract_state[3] * 6000

        self.robot_angle = np.arctan2(abstract_state[8], abstract_state[9])
        if self.robot_angle < 0:
            self.robot_angle += 2*np.pi

    def render(self, mode="display"):
        Field_length = 1200
        time.sleep(0.01)

        if self.rendering_init == False:
            pygame.init()

            self.field = pygame.display.set_mode((Field_length, Field_length * (2 / 3)))

            self.basic_field(Field_length)
            pygame.display.set_caption("Point Targeting Environment")
            self.clock = pygame.time.Clock()

            self.rendering_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.basic_field(Field_length)

        render_robot_x = (self.robot_x / 5200 + 1) * (Field_length / 2)
        render_robot_y = (self.robot_y / 3700 + 1) * (Field_length / 3)

        render_target_x = (self.target_x / 5200 + 1) * (Field_length / 2)
        render_target_y = (self.target_y / 3700 + 1) * (Field_length / 3)

        pygame.draw.circle(
            self.field,
            pygame.Color(40, 40, 40),
            (render_target_x, render_target_y),
            self.target_radius,
        )

        pygame.draw.circle(
            self.field,
            pygame.Color(148, 17, 0),
            (render_robot_x, render_robot_y),
            self.robot_radius,
            width=5,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(50, 50, 50),
            (render_robot_x, render_robot_y),
            (
                render_robot_x + self.robot_radius * np.cos(self.robot_angle),
                render_robot_y + self.robot_radius * np.sin(self.robot_angle),
            ),
            width=5,
        )

        pygame.display.update()
        self.clock.tick(60)
    def position_rule(self):

        # 3) ball out of bounds on top/bottom of field
        if self.target_y > 3000:
            self.target_y = 3000
        if self.target_y < -3000:
            self.target_y = -3000

        # disallow robot pass through goal-net
        if self.robot_x < -4500 or self.robot_x > 4500:
            if self.robot_y < -750:
                self.robot_y = -750
            if self.robot_y > 750:
                self.robot_y = 750

        # out of bounds
        if self.robot_x > 4500:
            self.robot_x = 4500
        if self.robot_x < -4500:
            self.robot_x = -4500
        if self.robot_y > 3000:
            self.robot_y = 3000
        if self.robot_y < -3000:
            self.robot_y = -3000

    def update_target_value(self):
        # Update Relative Angle
        self.delta_x = self.target_x - self.robot_x
        self.delta_y = self.target_y - self.robot_y
        self.theta_radians = np.arctan2(self.delta_y, self.delta_x)

        if self.theta_radians >= 0:
            self.relative_angle = self.theta_radians
        else:
            self.relative_angle = self.theta_radians + 2 * np.pi

    def update_goal_value(self):
        # Update Relative Angle to goal
        self.goal_delta_x = self.goal_x - self.robot_x
        self.goal_delta_y = self.goal_y - self.robot_y
        self.goal_theta_radians = np.arctan2(self.goal_delta_y, self.goal_delta_x)

        if self.goal_theta_radians >= 0:
            self.goal_relative_angle = self.goal_theta_radians
        else:
            self.goal_relative_angle = self.goal_theta_radians + 2 * np.pi

    def basic_field(self, _Field_length=1200):
        # you can change : (l_w = line width)
        Field_length = _Field_length
        l_w = 3

        # you can't change (based on the official robocup rule book ratio)
        Field_width = Field_length * (2 / 3)
        Penalty_area_length = Field_length * (1 / 15)
        Penalty_area_width = Field_length * (22 / 90)
        Penalty_cross_distance = Field_length * (13 / 90)
        Center_circle_diameter = Field_length * (15 / 90)
        Penalry_cross_size = Field_length * (1 / 90)
        Border_strip_width = Field_length * (7 / 90)
        goal_post_size = Field_length * (1 / 90)
        goal_area_width = Field_length * (1 / 6)
        goal_area_length = Field_length * (5 / 90)

        Soccer_green = (18, 160, 0)
        self.field.fill(Soccer_green)

        # drawing goal-area(dark green-Left/Right)
        pygame.draw.rect(
            self.field,
            (56, 87, 35),
            [
                Border_strip_width - goal_area_length,
                Field_width / 2 - goal_area_width / 2 - goal_post_size / 2,
                goal_area_length,
                goal_area_width + goal_post_size,
            ],
        )
        pygame.draw.rect(
            self.field,
            (48, 190, 0),
            [
                Field_length - Border_strip_width,
                Field_width / 2 - goal_area_width / 2 - goal_post_size / 2,
                goal_area_length,
                goal_area_width + goal_post_size,
            ],
        )
        # drawing out-line
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [Border_strip_width, Border_strip_width],
            [Field_length - Border_strip_width, Border_strip_width],
            l_w,
        )
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [Border_strip_width, Border_strip_width],
            [Border_strip_width, Field_width - Border_strip_width],
            l_w,
        )
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [Border_strip_width, Field_width - Border_strip_width],
            [Field_length - Border_strip_width, Field_width - Border_strip_width],
            l_w,
        )
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [Field_length - Border_strip_width, Field_width - Border_strip_width],
            [Field_length - Border_strip_width, Border_strip_width],
            l_w,
        )
        # drawing center-line
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [Field_length / 2, Border_strip_width],
            [Field_length / 2, Field_width - Border_strip_width],
            l_w,
        )
        pygame.draw.circle(
            self.field,
            (255, 255, 255),
            [Field_length / 2, Field_width / 2],
            Center_circle_diameter / 2,
            l_w,
        )
        pygame.draw.circle(
            self.field,
            (255, 255, 255),
            [Field_length / 2, Field_width / 2],
            Penalry_cross_size / 2,
        )
        # drawing keeper_area(left-side)
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [Border_strip_width, Field_width / 2 - Penalty_area_width / 2],
            [
                Border_strip_width + Penalty_area_length,
                Field_width / 2 - Penalty_area_width / 2,
            ],
            l_w,
        )
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [Border_strip_width, Field_width / 2 + Penalty_area_width / 2],
            [
                Border_strip_width + Penalty_area_length,
                Field_width / 2 + Penalty_area_width / 2,
            ],
            l_w,
        )
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [
                Border_strip_width + Penalty_area_length,
                Field_width / 2 - Penalty_area_width / 2,
            ],
            [
                Border_strip_width + Penalty_area_length,
                Field_width / 2 + Penalty_area_width / 2,
            ],
            l_w,
        )
        # drawing keeper_area(Right-side)
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [
                Field_length - Border_strip_width,
                Field_width / 2 - Penalty_area_width / 2,
            ],
            [
                Field_length - Border_strip_width - Penalty_area_length,
                Field_width / 2 - Penalty_area_width / 2,
            ],
            l_w,
        )
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [
                Field_length - Border_strip_width,
                Field_width / 2 + Penalty_area_width / 2,
            ],
            [
                Field_length - Border_strip_width - Penalty_area_length,
                Field_width / 2 + Penalty_area_width / 2,
            ],
            l_w,
        )
        pygame.draw.line(
            self.field,
            (255, 255, 255),
            [
                Field_length - Border_strip_width - Penalty_area_length,
                Field_width / 2 - Penalty_area_width / 2,
            ],
            [
                Field_length - Border_strip_width - Penalty_area_length,
                Field_width / 2 + Penalty_area_width / 2,
            ],
            l_w,
        )
        # drawing penalty_cross(Left/Right)
        pygame.draw.circle(
            self.field,
            (255, 255, 255),
            [Penalty_cross_distance + Border_strip_width, Field_width / 2],
            Penalry_cross_size / 2,
        )
        pygame.draw.circle(
            self.field,
            (255, 255, 255),
            [
                Field_length - Penalty_cross_distance - Border_strip_width,
                Field_width / 2,
            ],
            Penalry_cross_size / 2,
        )
        # drawing goal-post(grey color-Left/Right)
        pygame.draw.circle(
            self.field,
            (110, 110, 110),
            [
                Border_strip_width,
                Field_width / 2 - goal_area_width / 2 - goal_post_size / 2,
            ],
            goal_post_size / 2,
        )
        pygame.draw.circle(
            self.field,
            (110, 110, 110),
            [
                Border_strip_width,
                Field_width / 2 + goal_area_width / 2 + goal_post_size / 2,
            ],
            goal_post_size / 2,
        )
        pygame.draw.circle(
            self.field,
            (110, 110, 110),
            [
                Field_length - Border_strip_width,
                Field_width / 2 + goal_area_width / 2 + goal_post_size / 2,
            ],
            goal_post_size / 2,
        )
        pygame.draw.circle(
            self.field,
            (110, 110, 110),
            [
                Field_length - Border_strip_width,
                Field_width / 2 - goal_area_width / 2 - goal_post_size / 2,
            ],
            goal_post_size / 2,
        )

    def get_normalized_score(self, eval_score):
        return eval_score/100
