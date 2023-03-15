import numpy as np

from augment.utils import convert_to_absolute_obs, calculate_reward, convert_to_relative_obs


def translate_reflect_traj_y(obs, action, next_obs, reward, done):

    absolute_obs = convert_to_absolute_obs(obs)
    absolute_next_obs = convert_to_absolute_obs(next_obs)

    aug_absolute_obs = absolute_obs.copy()
    aug_absolute_next_obs = absolute_next_obs.copy()
    aug_action = action.copy()
    aug_done = done.copy()

    ball_at_goal_y = absolute_obs[-1, 3]

    delta_y = np.random.uniform(-(500+ball_at_goal_y), 500-ball_at_goal_y)
    aug_absolute_obs[:, 1] += delta_y
    aug_absolute_next_obs[:, 1] += delta_y

    aug_absolute_obs[:, 3] += delta_y
    aug_absolute_next_obs[:, 3] += delta_y

    # reflect about y=0 w.p. 0.5
    if np.random.random() < 0.5:
        aug_absolute_obs[:, 1] *= -1
        aug_absolute_next_obs[:, 1] *= -1
        aug_absolute_obs[:, 3] *= -1
        aug_absolute_next_obs[:, 3] *= -1
        aug_absolute_obs[:, 4] *= -1
        aug_absolute_next_obs[:, 4] *= -1

        aug_action[:, 0] *= -1
        aug_action[:, 1] *= 1
        aug_action[:, 2] *= -1

    aug_reward, _ = calculate_reward(aug_absolute_next_obs)

    aug_obs = convert_to_relative_obs(aug_absolute_obs)
    aug_next_obs = convert_to_relative_obs(aug_absolute_next_obs)

    return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done
