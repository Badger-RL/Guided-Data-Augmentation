import copy
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
            "goal": 3,  # Team
            "goal_scored": False,
            "ball_to_goal": 1/40,  # Team
            "out_of_bounds": 0,
            "is_out_of_bounds": False,
            "agent_to_ball": 1/20,  # Individual
            "looking_at_ball": 1/100,  # Individual
        }

        # self.reward_dict = {
        #     "goal": 100000,  # Team
        #     "goal_scored": False,
        #     "ball_to_goal": 10,  # Team
        #     "out_of_bounds": 0,
        #     "is_out_of_bounds": False,
        #     "agent_to_ball": 1,  # Individual
        #     "looking_at_ball": 0.01,  # Individual
        # }

    def get_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

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

        self.ball = []
        self.ball = [np.random.uniform(-4500, 4500), np.random.uniform(-3000, 3000)]
        # Spawn ball around edges of field for more interesting play
        field_length = 4000
        field_height = 2500
        spawn_range = 50

        # Choose a random number 0-3 to determine which edge to spawn the ball on
        # edge = np.random.randint(4)
        # if edge == 0:
        #     # Spawn on left edge
        #     ball_x = np.random.uniform(-field_length, -field_length + spawn_range)
        #     ball_y = np.random.uniform(-field_height, field_height)
        # elif edge == 1:
        #     # Spawn on top edge
        #     ball_x = np.random.uniform(-field_length, field_length)
        #     ball_y = np.random.uniform(field_height - spawn_range, field_height)
        # elif edge == 2:
        #     # Spawn on right edge
        #     ball_x = np.random.uniform(field_length - spawn_range, field_length)
        #     ball_y = np.random.uniform(-field_height, field_height)
        # else:
        #     # Spawn on bottom edge
        #     ball_x = np.random.uniform(-field_length, field_length)
        #     ball_y = np.random.uniform(-field_height, -field_height + spawn_range)
        #
        # self.ball = [ball_x, ball_y]

        # Goal is 4400, [-1000 to 1000]
        # self.ball = [100, 2900]

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
            self.robots[0].copy(), self.ball.copy(), [self.angles[0], self.ball_angle, self.ball_velocity, ]
        ])

        previous_locations = {}
        # Save previous locations
        for agent in self.agents:
            i = self.agent_idx[agent]
            previous_locations[agent] = self.robots[i].copy()

        ball_previous_location = self.ball.copy()

        # Update agent locations and ball
        # print(actions, self.agents)
        for agent in self.agents:
            actions[agent][3] = 0
            action = actions[agent]
            self.move_agent(agent, action)
            self.update_ball()

        # Calculate rewards
        for agent in self.agents:
            obs[agent] = self.get_obs(agent)
            rew[agent] = self.calculate_reward(agent, actions[agent], previous_locations[agent], ball_previous_location)
            # terminated[agent] = self.reward_dict['is_out_of_bounds'] or self.reward_dict['goal_scored']
            terminated[agent] = self.reward_dict['is_out_of_bounds']
            # terminated[agent] = self.reward_dict['goal_scored']

            truncated[agent] = False

            absolute_next_obs = np.concatenate([
                self.robots[0], self.ball, [self.angles[0], self.ball_angle, self.ball_velocity,]
            ])
            info[agent] = {
                'is_success': self.reward_dict['goal_scored'],
                'terminated': terminated[agent],
                'absolute_obs': absolute_obs,
                'absolute_next_obs': absolute_next_obs
            }

        # if self.reward_dict["goal_scored"]:
        #     # Reset ball
        #     self.ball = [np.random.uniform(-2500, 2500), np.random.uniform(-1500, 1500)]
        #     self.reward_dict["goal_scored"] = False
        agent = self.agents[0]
        return obs[agent], rew[agent], terminated[agent], info[agent]

    '''
    Checks if ball is in goal area
    '''

    def goal(self):
        if self.ball[0] > 4400 and self.ball[1] < 500 and self.ball[1] > -500:
            return True
        return False

    def ball_is_in_bounds(self):
        if self.goal():
            return True
        elif np.abs(self.ball[0]) < 4500 and np.abs(self.ball[1]) < 3500:
            return True
        else:
            return False

    def is_at_goal(self, ball_pos):
        if ball_pos[0] > 4400 and ball_pos[1] < 500 and ball_pos[1] > -500:
            return True
        return False

    def ball_is_in_bounds_2(self, ball_pos):
        if self.is_at_goal(ball_pos):
            return True
        elif np.abs(ball_pos[0]) < 4500 and np.abs(ball_pos[1]) < 3500:
            return True
        else:
            return False

    def looking_at_ball(self, agent):
        return self.check_facing_ball(agent)

    def in_opp_goal(self):
        if self.ball[0] < -4400 and self.ball[1] < 1000 and self.ball[1] > -1000:
            return True
        return False

    def calculate_reward(self, agent, action, prev_location, prev_ball_location):
        i = self.agent_idx[agent]
        reward = 0

        if self.in_opp_goal():
            return 0

        info_dict = {}
        # testing
        # Goal - Team
        if self.goal():
            reward += self.reward_dict["goal"]
            self.reward_dict["goal_scored"] = True
            info_dict["goal"] = True

        # # Ball to goal - Team
        cur_ball_distance = self.get_distance(self.ball, [4800, 0])
        prev_ball_distance = self.get_distance(prev_ball_location, [4800, 0])
        reward += self.reward_dict["ball_to_goal"] * (prev_ball_distance - cur_ball_distance)
        info_dict["ball_to_goal"] = True

        # reward for stepping towards ball
        cur_distance = self.get_distance(self.robots[i], self.ball)
        prev_distance = self.get_distance(prev_location, self.ball)
        reward += self.reward_dict["agent_to_ball"] * (prev_distance - cur_distance)
        info_dict["agent_to_ball"] = True

        if self.looking_at_ball(agent):
            reward += self.reward_dict["looking_at_ball"]
            info_dict["looking_at_ball"] = True

        if not self.ball_is_in_bounds():
            reward += self.reward_dict["out_of_bounds"]
            self.reward_dict['is_out_of_bounds'] = True

        return reward

    def calculate_reward_2(self, abs_obs, abs_prev_obs):
        i = 0
        reward = 0

        robot_pos = abs_obs[:2]
        prev_robot_pos = abs_prev_obs[:2]
        robot_angle = abs_obs[-1]

        ball_pos = abs_obs[2:4]
        prev_ball_pos = abs_prev_obs[2:4]

        info_dict = {}
        # testing
        # Goal - Team
        if self.is_at_goal(ball_pos):
            reward += self.reward_dict["goal"]
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

        if self.check_facing_ball_2(robot_pos, ball_pos, robot_angle):
            reward += self.reward_dict["looking_at_ball"]
            info_dict["looking_at_ball"] = True

        if not self.ball_is_in_bounds_2(ball_pos):
            reward += self.reward_dict["out_of_bounds"]
            self.reward_dict['is_out_of_bounds'] = True

        return reward, self.reward_dict["is_out_of_bounds"]

    def get_normalized_score(self, eval_score):
        return eval_score/100

    def set_state(self, robot_pos, robot_angle, ball_pos, ball_angle, ball_velocity):
        self.robots[0] = robot_pos.tolist()
        self.angles[0] = robot_angle
        self.ball = ball_pos.copy()
        self.ball_angle = ball_angle
        self.ball_velocity = ball_velocity