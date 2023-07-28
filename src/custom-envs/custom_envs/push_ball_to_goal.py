import copy
import math

import gym
import numpy as np
import sys

sys.path.append(sys.path[0] + "/..")

from custom_envs.base import BaseEnv

class PushBallToGoalEnv(BaseEnv):
    metadata = {'render_modes': ['human', 'rgb_array'],
                "render_fps": 30,
                }

    def __init__(self, continuous_actions=True, render_mode='rgb_array'):
        # Init base class
        super().__init__(continuous_actions=continuous_actions)

        '''
        Required:
        - possible_agents
        - action_spaces
        - observation_spaces
        '''
        self.rendering_init = False
        self.render_mode = render_mode

        # agents
        self.possible_agents = ["agent_0"]
        self.agents = self.possible_agents[:]
        self.continous_actions = continuous_actions
        self.teams = [0, 0]
        self.agent_idx = {agent: i for i, agent in enumerate(self.agents)}


        self.episode_length = 1500

        self.ball_acceleration = -3
        self.ball_velocity_coef = 4
        self.displacement_coef = 0.06
        self.angle_displacement = 0.25
        self.robot_radius = 20

        self.reward_dict = {
            "goal": 0,  # Team
            "goal_scored": False,
            "ball_to_goal": 0.0001,  # Team
            "out_of_bounds": -2000,
            "is_out_of_bounds": False,
            "agent_to_ball": 0.00001,  # Individual
            "looking_at_ball": 0.01,  # Individual
        }

    def get_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2), axis=-1)

    '''
    ego-centric observation:
        origin,
        goal,
        other robots,
        ball
    '''

    def get_obs(self, agent):
        i = self.agent_idx[agent]
        agent_loc = self.robots[i] + [self.angles[i]]

        obs = []

        # Get origin position
        origin_obs = self.get_relative_observation(agent_loc, [0, 0])
        obs.extend(origin_obs)

        # Get goal position
        goal_obs = self.get_relative_observation(agent_loc, [4800, 0])
        obs.extend(goal_obs)

        # Get other positions
        for j in range(len(self.agents)):
            if i == j:
                continue
            robot_obx = self.get_relative_observation(agent_loc, self.robots[j])
            obs.extend(robot_obx)

        # Get ball position
        ball_obs = self.get_relative_observation(agent_loc, self.ball)
        obs.extend(ball_obs)

        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, return_info=False, options=None, **kwargs):
        self.time = 0

        self.ball_velocity = 0
        self.ball_angle = 0

        self.robots = [[np.random.uniform(-4500, 4500), np.random.uniform(-3000, 3000)] for _ in
                       range(len(self.agents))]
        self.angles = [np.random.uniform(-np.pi, np.pi) for _ in range(len(self.agents))]

        self.reward_dict["goal_scored"] = False
        self.reward_dict["is_out_of_bounds"] = False

        self.previous_distances = [None for _ in range(len(self.agents))]
        self.ball = [np.random.uniform(-4500, 4500), np.random.uniform(-3000, 3000)]

        observations = {}
        for agent in self.agents:
            observations[agent] = self.get_obs(agent)
        return observations[self.agents[0]]

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        actions = {'agent_0': actions.copy()}
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        self.time += 1

        absolute_obs = np.concatenate([
            self.robots[0].copy(), self.ball.copy(), [self.angles[0]]
        ])

        previous_locations = {}
        # Save previous locations
        for agent in self.agents:
            i = self.agent_idx[agent]
            previous_locations[agent] = self.robots[i].copy()

        # Update agent locations and ball
        for agent in self.agents:
            actions[agent][3] = 0
            action = actions[agent]
            self.move_agent(agent, action)
            self.update_ball()

        absolute_next_obs = np.concatenate([
            self.robots[0], self.ball, [self.angles[0]]
        ])
        # Calculate rewards
        for agent in self.agents:
            obs[agent] = self.get_obs(agent)
            rew[agent], is_goal, is_out_of_bounds = self.calculate_reward_2(absolute_next_obs)
            terminated[agent] = is_goal or is_out_of_bounds

            truncated[agent] = False


            info[agent] = {
                'is_success': is_goal,
                'terminated': terminated[agent],
                'absolute_obs': absolute_obs,
                'absolute_next_obs': absolute_next_obs
            }

        agent = self.agents[0]
        return obs[agent], rew[agent], terminated[agent], info[agent]

    '''
    Checks if ball is in goal area
    '''
    def is_at_goal(self, ball_pos):
        if ball_pos[0] > 4400 and ball_pos[1] < 500 and ball_pos[1] > -500:
            return True
        return False

    def ball_is_in_bounds(self, ball_pos):
        if self.is_at_goal(ball_pos):
            return True
        elif np.abs(ball_pos[0]) < 4500 and np.abs(ball_pos[1]) < 3500:
            return True
        else:
            return False

    def in_opp_goal(self):
        if self.ball[0] < -4400 and self.ball[1] < 1000 and self.ball[1] > -1000:
            return True
        return False

    def calculate_reward_2(self, abs_next_obs):
        i = 0
        reward = 0

        robot_pos = abs_next_obs[:2].copy()
        robot_angle = abs_next_obs[-1].copy()

        ball_pos = abs_next_obs[2:4].copy()

        is_goal = False
        is_out_of_bounds = False

        if self.is_at_goal(ball_pos):
            reward += self.reward_dict["goal"]
            is_goal = True

        # ball to goal
        cur_ball_distance = -self.get_distance(ball_pos, [4800, 0])
        reward += self.reward_dict["ball_to_goal"] * cur_ball_distance

        # robot to ball
        cur_distance = -self.get_distance(robot_pos, ball_pos)
        reward += self.reward_dict["agent_to_ball"] * cur_distance

        if self.check_facing_ball(robot_pos, ball_pos, robot_angle):
            reward += self.reward_dict["looking_at_ball"]

        if not self.ball_is_in_bounds(ball_pos):
            reward += self.reward_dict["out_of_bounds"]
            is_out_of_bounds = True

        return reward, is_goal, is_out_of_bounds


    def calculate_reward_vec(self, abs_prev_obs, abs_obs):
        i = 0
        reward = np.zeros(len(abs_obs))

        robot_pos = abs_obs[:, :2]
        prev_robot_pos = abs_prev_obs[:, :2]
        robot_angle = abs_obs[:, -1]

        ball_pos = abs_obs[:, 2:4]
        prev_ball_pos = abs_prev_obs[:, 2:4]

        info_dict = {}
        # testing
        # Goal - Team
        mask = self.ball_is_at_goal(ball_pos)
        reward[mask] += self.reward_dict["goal"]
        self.reward_dict["goal_scored"] = True
        info_dict["goal"] = True

        # # Ball to goal - Team
        cur_ball_distance = self.get_distance(ball_pos, [4800, 0])
        prev_ball_distance = self.get_distance(prev_ball_pos, [4800, 0])
        reward += self.reward_dict["ball_to_goal"] * (prev_ball_distance - cur_ball_distance)
        info_dict["ball_to_goal"] = True

        # reward for stepping towards ball
        cur_distance = self.get_distance(robot_pos, ball_pos)
        prev_distance = self.get_distance(prev_robot_pos, prev_ball_pos)
        reward += self.reward_dict["agent_to_ball"] * (prev_distance - cur_distance)
        info_dict["agent_to_ball"] = True

        mask = self.check_facing_ball_vec(robot_pos, ball_pos, robot_angle)
        reward[mask] += self.reward_dict["looking_at_ball"]
        info_dict["looking_at_ball"] = True

        mask = ~self.ball_is_in_bounds_vec(ball_pos)
        reward[mask] += self.reward_dict["out_of_bounds"]
        self.reward_dict['is_out_of_bounds'] = True

        return reward, mask

    def get_normalized_score(self, eval_score):
        return eval_score/100

    def set_state(self, robot_pos, ball_pos, robot_angle, ):
        self.robots[0] = robot_pos.tolist()
        self.angles[0] = robot_angle
        self.ball = ball_pos.copy()
        self.ball_angle = np.arctan2(robot_pos[1] - ball_pos[1], robot_pos[0] - ball_pos[0])
        # self.ball_angle = math.atan2(self.ball[1] - robot_pos[1], self.ball[0] - robot_pos[0])