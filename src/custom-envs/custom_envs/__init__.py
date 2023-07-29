from gym.envs.registration import register

register(
    id="PushBallToGoal-v0",
    entry_point="custom_envs.push_ball_to_goal:PushBallToGoalEnv",
    max_episode_steps=2000,
)

register(
    id="PushBallToGoalCorner-v0",
    entry_point="custom_envs.push_ball_to_goal:PushBallToGoalEnv",
    max_episode_steps=2000,
    kwargs={
        "init_robot_x_range": [1000, 4500],
        "init_robot_y_range": [-3500, 3500],
        "init_ball_x_range": [4000, 4500],
        "init_ball_y_range": [-3300, -2800],
    }
)