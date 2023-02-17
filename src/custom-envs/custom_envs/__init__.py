from gym.envs.registration import register

register(
    id="PushBalltoGoal-v0",
    entry_point="custom_envs.push_ball_to_goal:PushBallToGoalEnv",
    # max_episode_steps=500,
)