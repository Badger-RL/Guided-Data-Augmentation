from gym.envs.registration import register

register(
    id="PushBallToGoal-v0",
    entry_point="custom_envs.push_ball_to_goal:PushBallToGoalEnv",
    max_episode_steps=500,
    kwargs={
        "robot_x_range": [-3500,3500],
        "robot_y_range": [-2500,2500],
        "ball_x_range": [-2500,2500],
        "ball_y_range": [-2000,2000]
    }
)

register(
    id="PushBallToGoalCorner-v0",
    entry_point="custom_envs.push_ball_to_goal:PushBallToGoalEnv",
    max_episode_steps=500,
    kwargs={
        "robot_x_range": [1500,3500],
        "robot_y_range": [-1000,1000],
        "ball_x_range": [4000,4500],
        "ball_y_range": [2500,2750]
    }
)

register(
    id="PushBallToGoalRestricted-v0",
    entry_point="custom_envs.push_ball_to_goal:PushBallToGoalEnv",
    max_episode_steps=500,
    kwargs={
        "robot_x_range": [-400,400],
        "robot_y_range": [-400,400],
        "ball_x_range": [-400,400],
        "ball_y_range": [-400,400]
    }
)