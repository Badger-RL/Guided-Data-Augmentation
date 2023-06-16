import numpy as np

from augment.utils import convert_to_absolute_obs, calculate_reward, convert_to_relative_obs, check_in_bounds


def translate_reflect_traj_y(obs, action, next_obs, reward, done, check_goal_post):

    absolute_obs = convert_to_absolute_obs(obs)
    absolute_next_obs = convert_to_absolute_obs(next_obs)

    aug_absolute_obs = absolute_obs.copy()
    aug_absolute_next_obs = absolute_next_obs.copy()
    aug_action = action.copy()
    aug_done = done.copy()

    # Verify that translation doesn't move the agent out of bounds.
    attempts = 0
    while attempts < 100:
        attempts += 1

        ball_at_goal_x = absolute_obs[-1, 2]
        ball_at_goal_y = absolute_obs[-1, 3]
        new_ball_final_pos_x = np.random.uniform(4510, 4510)
        new_ball_final_pos_y = np.random.uniform(-500, 500)

        traj_delta_x = new_ball_final_pos_x - ball_at_goal_x
        traj_delta_y = new_ball_final_pos_y - ball_at_goal_y

        aug_absolute_obs[:, 0] += traj_delta_x
        aug_absolute_obs[:, 1] += traj_delta_y
        aug_absolute_obs[:, 2] += traj_delta_x
        aug_absolute_obs[:, 3] += traj_delta_y

        aug_absolute_next_obs[:, 0] += traj_delta_x
        aug_absolute_next_obs[:, 1] += traj_delta_y
        aug_absolute_next_obs[:, 2] += traj_delta_x
        aug_absolute_next_obs[:, 3] += traj_delta_y

        # print(aug_absolute_obs[-1, 2], aug_absolute_obs[-1,3])

        # check if agent and ball are in bounds
        if check_in_bounds(aug_absolute_obs, check_goal_post=check_goal_post) and \
                check_in_bounds(aug_absolute_next_obs, check_goal_post=check_goal_post):
            break
        else:
            aug_absolute_obs = absolute_obs.copy()
            aug_absolute_next_obs = absolute_next_obs.copy()
    if attempts >= 100:
        print(f'Skipping trajectory after {attempts} augmentation attempts.')
        return None, None, None, None, None

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
