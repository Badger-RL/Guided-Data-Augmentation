import gym
import custom_envs


def test_default_push_ball_to_goal():
    env = gym.make("PushBallToGoal-v0")
    env.reset()
    assert env.ball_x_range[0] == -2500
    assert env.ball_x_range[1] == 2500
    assert env.ball_y_range[0] == -2000
    assert env.ball_y_range[1] == 2000
    assert env.robot_x_range[0] == -3500
    assert env.robot_x_range[1] == 3500
    assert env.robot_y_range[0] == -2500
    assert env.robot_y_range[1] == 2500



def test_restricted_push_ball_to_goal():
    env = gym.make("PushBallToGoalRestricted-v0")
    env.reset()
    assert env.ball_x_range[0] == -400
    assert env.ball_x_range[1] == 400
    assert env.ball_y_range[0] == -400
    assert env.ball_y_range[1] == 400
    assert env.robot_x_range[0] == -400
    assert env.robot_x_range[1] == 400
    assert env.robot_y_range[0] == -400
    assert env.robot_y_range[1] == 400
