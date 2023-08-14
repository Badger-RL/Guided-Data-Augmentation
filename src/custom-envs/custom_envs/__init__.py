from gym.envs.registration import register

register(
    id="PushBallToGoal-v0",
    entry_point="custom_envs.push_ball_to_goal:PushBallToGoalEnv",
    max_episode_steps=2000,
)

register(
    id="PushBallToGoal-v1",
    entry_point="custom_envs.push_ball_to_goal:PushBallToGoalEnv",
    max_episode_steps=2000,
    kwargs={
        'sparse': True,
        'stochastic': False
    }
)

register(
    id="PushBallToGoal-v2",
    entry_point="custom_envs.push_ball_to_goal:PushBallToGoalEnv",
    max_episode_steps=800,
    kwargs={
        'sparse': True,
        'stochastic': False,
        'realistic': False,
        'clip_out_of_bounds': False,
        'displacement_coef': 0.20,
        # "init_ball_x_range": [-3000, 3000],
        # "init_ball_y_range": [-3000, 3000],
    }
)

register(
    id="PushBallToGoal-v3",
    entry_point="custom_envs.push_ball_to_goal:PushBallToGoalEnv",
    max_episode_steps=500,
    kwargs={
        'sparse': True,
        'stochastic': False,
        'realistic': False,
        'displacement_coef': 0.20
    }
)

register(
    id="PushBallToGoalEasy-v1",
    entry_point="custom_envs.push_ball_to_goal:PushBallToGoalEnv",
    max_episode_steps=2000,
    kwargs={
        'sparse': True,
        "init_robot_x_range": [-500, -100],
        "init_robot_y_range": [-500, 500],
        "init_ball_x_range": [0, 500],
        "init_ball_y_range": [-500, 500],
    }
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