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
        return self._augment(aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, **kwargs)

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

        self.x_scale = 9000
        self.y_scale = 6000
        self.scale = np.array([self.x_scale, self.y_scale])

        self.goal_x = 4800
        self.goal_y = 0
        self.goal = np.array([self.goal_x, self.goal_y])
        self.displacement_coef = 0.2
        self.max_dist = np.sqrt(self.x_scale**2 + self.y_scale**2)

    def _sample_robot_pos(self, n=1):
        x = np.random.uniform(-3500, 3500)
        y = np.random.uniform(-2500, 2500)
        return np.array([x, y])

    def _sample_robot_angle(self, n=1):
        return np.random.uniform(0, 2 * np.pi, size=(n,))

    def _convert_to_absolute_obs(self, obs):

        target_x = (self.goal_x - obs[2]*self.x_scale)
        target_y = (self.goal_y - obs[3]*self.y_scale)
        robot_x = (target_x - obs[0]*self.x_scale)
        robot_y = (target_y - obs[1]*self.y_scale)

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
            robot_angle
        ])

    def _convert_to_relative_obs(self, obs):
        # Get origin position
        relative_obs = []
        agent_loc = np.concatenate([obs[:2], [obs[-1]]])
        ball_loc = obs[2:4]
        origin_obs = self.get_relative_observation(agent_loc, [0, 0])
        relative_obs.extend(origin_obs)

        # Get goal position
        goal_obs = self.get_relative_observation(agent_loc, [4800, 0])
        relative_obs.extend(goal_obs)

        # Get ball position
        ball_obs = self.get_relative_observation(agent_loc, ball_loc)
        relative_obs.extend(ball_obs)

        return np.array(relative_obs, dtype=np.float32)

    def get_relative_observation(self, agent_loc, object_loc):
        # Get relative position of object to agent, returns x, y, angle
        # Agent loc is x, y, angle
        # Object loc is x, y

        # Get relative position of object to agent
        x = object_loc[0] - agent_loc[0]
        y = object_loc[1] - agent_loc[1]
        angle = np.arctan2(y, x) - agent_loc[2]

        # Rotate x, y by -agent angle
        xprime = x * np.cos(-agent_loc[2]) - y * np.sin(-agent_loc[2])
        yprime = x * np.sin(-agent_loc[2]) + y * np.cos(-agent_loc[2])

        return [xprime / 10000, yprime / 10000, np.sin(angle), np.cos(angle)]

    def calculate_reward(self, absolute_next_obs):
        robot_x = absolute_next_obs[0]
        robot_y = absolute_next_obs[1]
        target_x = absolute_next_obs[2]
        target_y = absolute_next_obs[3]
        robot_angle = absolute_next_obs[4]
        robot_angle = np.degrees(robot_angle) % 360

        at_goal = self.at_goal(target_x, target_y)

        angle_robot_ball = np.arctan2(target_y - robot_y, target_x - robot_x)
        angle_robot_ball = np.degrees(angle_robot_ball) % 360
        is_facing_ball = abs(angle_robot_ball - robot_angle) < 30

        reward = 0

        if is_facing_ball:
            robot_location = np.array([robot_x, robot_y])
            target_location = np.array([target_x, target_y])
            goal_location = np.array([self.goal_x, self.goal_y])

            distance_robot_target = np.linalg.norm(target_location - robot_location)
            distance_target_goal = np.linalg.norm(goal_location - target_location)

            reward_dist_to_ball = 1/distance_robot_target
            reward_dist_to_goal = 1/distance_target_goal
            reward = 0.9*reward_dist_to_goal + 0.1*reward_dist_to_ball

        if at_goal:
            reward += 1

        return reward, at_goal

    def at_goal(self, target_x, target_y):
        at_goal = False
        if target_x > 4400:
            if target_y < 500 and target_y > -500:
                at_goal = True

        return at_goal


