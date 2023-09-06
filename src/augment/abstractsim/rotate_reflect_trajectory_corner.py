import copy
import numpy as np

from src.augment.utils import convert_to_absolute_obs, calculate_reward, convert_to_relative_obs, check_valid, \
    check_in_bounds


BEHAVIORS = {
    # 'fish_out_ball': {
    #     # 'indices': (0, 1170),
    #     # 'indices': (350, 1170),
    #     'indices': (350, 1170),
    #     'max_len': 1,
    # },
    'curve_around_ball': {
        # 'indices': (0, 1170),
        # 'indices': (350, 1170),
        'indices': (0, 800),
        'max_len': 1,
    },
    # 'fish_out_ball_goal': {
    #     'indices': (600, 1170),
    #     'max_len': 1,
    # },
    'fish_out_ball_tight': {
        # 'indices': (1170, 1700),
        'indices': (1370, 1570),
        'max_len': 1,
    },
    # 'curve_before_goal': {
    #     'indices': (1950, 2995),
    #     # 'indices': (1950, 2650),
    #     'max_len': 1,
    # },
    'curve_to_goal': {
        'indices': (1950, 2995),
        # 'indices': (1950, 3295),
        # 'indices': (1950, 2650),
        'max_len': 1,
    },
    'curve_to_goal_2': {
        'indices': (1900, 2500),
        # 'indices': (1950, 3295),
        # 'indices': (1950, 2650),
        'max_len': 1,
    },
    # 'turnaround': {
    #     'indices': (1250, 1800),
    #     'max_len': 1,
    # },
    # 'curve_to_goal_2': {
    #     # 'indices': (1950, 2995),
    #     'indices': (1950, 2650),
    #     'max_len': 1,
    # },
    # 'straight_1': {
    #     'indices': (0, 400),
    #     'max_len': 1,
    # },
    # 'straight_2': {
    #     # 'indices': (1530, 1800),
    #     'indices': (0, 400),
    #     'max_len': 1,
    # },
    # 'out_of_bounds': {
    #     'indices': (2600, 2650),
    #     'max_len': 1,
    # },

    # 'straight_2': {
    #     'indices': (1530, 2200),
    #     'max_len': 1,
    # },
    # 'straight_2': {
    #     'indices': (2200, 2995),
    #     'max_len': 1,
    # },
}

def rotate_reflect_traj(env, obs, action, next_obs, reward, done, guided):
    aug_abs_obs = obs.copy()
    aug_abs_next_obs = next_obs.copy()
    aug_action = action.copy()
    aug_done = done.copy()
    keys = list(BEHAVIORS.keys())
    num_behaviors = len(BEHAVIORS)

    behavior = None

    if np.random.random() < 1:
        idx = np.random.randint(num_behaviors)
        # if np.random.random() < 0:
        #     idx = num_behaviors-1
        # idx = idx % num_behaviors # hack to sample more curve_to_ball segments
        behavior = keys[idx]
        behavior_dict = BEHAVIORS[behavior]
        start_idx, end_idx = behavior_dict['indices']
        max_len = behavior_dict['max_len']

        k = np.random.randint(1,max_len+1)  # 6 also cut off at 100 attempts
        disp = aug_abs_next_obs[end_idx-1, :2] - aug_abs_obs[start_idx, :2]

        aug_abs_obs = np.concatenate(
            [aug_abs_obs[start_idx:end_idx] for _ in range(k)]
        )
        aug_abs_next_obs = np.concatenate(
            [aug_abs_next_obs[start_idx:end_idx] for _ in range(k)]
        )
        aug_action = np.concatenate(
            [aug_action[start_idx:end_idx] for _ in range(k)]
        )
        aug_done = np.concatenate(
            [aug_done[start_idx:end_idx] for _ in range(k)]
        )

        if behavior in ['straight_1', 'straight2']:
            # put ball at final position in subtraj so it doesn't screw up rotation/
            ball_init = aug_abs_obs[:, :2] + 100
            aug_abs_obs[:, 2:4] = ball_init
            aug_abs_next_obs[:, 2:4] = ball_init

        for i in range(1,k):
            diff = end_idx - start_idx + 1
            start = diff * i
            end = diff * (i+1)
            # disp = aug_abs_next_obs[start-400, :2] - aug_abs_obs[start, :2]

            aug_abs_obs[start:end, :2] += disp*i
            aug_abs_next_obs[start:end, :2] += disp*i
            aug_abs_obs[start:end, 2:4] += disp*i
            aug_abs_next_obs[start:end, 2:4] += disp*i
            # if start_idx == 1300:
            #     aug_abs_obs[start:end, 0] += 200
            #     aug_abs_next_obs[start:end, 0] += 200

    aug_abs_obs_original = aug_abs_obs.copy()
    aug_abs_next_obs_original = aug_abs_next_obs.copy()


    ball_init_hard_x = (4200, 4400)
    ball_init_hard_y = (-2700, -2300)


    attempts = 0
    max_attempts = 100
    while attempts < max_attempts:
        attempts += 1

        if guided:
            new_ball_final_pos_x = np.random.uniform(4500, 4700)
            new_ball_final_pos_y = np.random.uniform(-10, 10)


        if behavior == 'turnaround':
            # new_ball_final_pos_x = np.random.uniform(3200, 4200)
            # new_ball_final_pos_y = np.random.uniform(-1500, -1300)
            new_ball_final_pos_x = np.random.uniform(4501, 4510)
            new_ball_final_pos_y = np.random.uniform(-300, 300)
            theta_range = (60,120)
            reflect = 0

            ball_at_goal_x = aug_abs_obs[-1, 2].copy()
            ball_at_goal_y = aug_abs_obs[-1, 3].copy()

        if behavior == 'fish_out_ball':
            # new_ball_final_pos_x = np.random.uniform(3200, 4200)
            # new_ball_final_pos_y = np.random.uniform(-1500, -1300)
            new_ball_final_pos_x = np.random.uniform(4300, 4400)
            new_ball_final_pos_y = np.random.uniform(-2500, -1500)
            theta_range = (30,120)
            reflect = 0

            ball_at_goal_x = aug_abs_obs[0, 2].copy()
            ball_at_goal_y = aug_abs_obs[0, 3].copy()

        if behavior == 'curve_around_ball':
            # new_ball_final_pos_x = np.random.uniform(3200, 4200)
            # new_ball_final_pos_y = np.random.uniform(-1500, -1300)
            new_ball_final_pos_x = np.random.uniform(4300, 4400)
            new_ball_final_pos_y = np.random.uniform(-2500, -1500)
            theta_range = (-60,120)
            reflect = 0

            ball_at_goal_x = aug_abs_obs[0, 2].copy()
            ball_at_goal_y = aug_abs_obs[0, 3].copy()

        if behavior == 'fish_out_ball_goal':
            # new_ball_final_pos_x = np.random.uniform(3200, 4200)
            # new_ball_final_pos_y = np.random.uniform(-1500, -1300)
            new_ball_final_pos_x = np.random.uniform(4501, 4600)
            new_ball_final_pos_y = np.random.uniform(-100, 100)
            theta_range = (-120, -90)
            reflect = 1

            ball_at_goal_x = aug_abs_obs[-1, 2].copy()
            ball_at_goal_y = aug_abs_obs[-1, 3].copy()

        if behavior == 'fish_out_ball_tight':
            new_ball_final_pos_x = np.random.uniform(4300, 4400)
            # new_ball_final_pos_y = np.random.uniform(-1500, -1300)
            # new_ball_final_pos_x = np.random.uniform(3500, 4400)
            new_ball_final_pos_y = np.random.uniform(2000, 1500)
            theta_range = (-90,-30)
            reflect = 1

            ball_at_goal_x = aug_abs_obs[0, 2].copy()
            ball_at_goal_y = aug_abs_obs[0, 3].copy()

        if behavior == 'curve_to_goal':
            new_ball_final_pos_x = np.random.uniform(3800, 4700)
            new_ball_final_pos_y = np.random.uniform(-300, 0)
            theta_range = (-90, -10)
            reflect = 1

            ball_at_goal_x = aug_abs_obs[-1, 2].copy()
            ball_at_goal_y = aug_abs_obs[-1, 3].copy()

        if behavior == 'curve_to_goal_2':
            new_ball_final_pos_x = np.random.uniform(4501, 4700)
            new_ball_final_pos_y = np.random.uniform(-10, 10)
            theta_range = (-15, 15)
            reflect = 1

            ball_at_goal_x = aug_abs_obs[-1, 2].copy()
            ball_at_goal_y = aug_abs_obs[-1, 3].copy()

        if behavior == 'straight_1':
            # new_ball_final_pos_x = np.random.uniform(1000, 2000)
            # new_ball_final_pos_y = np.random.uniform(-2800, -2300)
            new_ball_final_pos_x = np.random.uniform(3900, 4100)
            new_ball_final_pos_y = np.random.uniform(3500, 2000)
            theta_range = (15, 30)
            reflect = 1

            ball_at_goal_x = aug_abs_obs[-1, 0].copy()
            ball_at_goal_y = aug_abs_obs[-1, 0].copy()

        if behavior == 'straight_2':
            # new_ball_final_pos_x = np.random.uniform(100, 800)
            # new_ball_final_pos_y = np.random.uniform(-1200, -500)
            new_ball_final_pos_x = np.random.uniform(3000, 3700)
            new_ball_final_pos_y = np.random.uniform(4200, 4500)
            theta_range = (15, 30)
            # theta_range = (90, 120)
            reflect = 1

            ball_at_goal_x = aug_abs_obs[0, 0].copy()
            ball_at_goal_y = aug_abs_obs[0, 0].copy()

        if behavior == 'out_of_bounds':
            new_ball_final_pos_x = np.random.uniform(4500, 4505)
            new_ball_final_pos_y = np.random.uniform(-3000, -751)
            theta_range = (15, 80)
            reflect = 0

            ball_at_goal_x = aug_abs_obs[-1, 2].copy()
            ball_at_goal_y = aug_abs_obs[-1, 3].copy()

        # new_ball_final_pos_x = ball_at_goal_x
        # new_ball_final_pos_y = ball_at_goal_y

        traj_delta_x = new_ball_final_pos_x - ball_at_goal_x
        traj_delta_y = new_ball_final_pos_y - ball_at_goal_y

        aug_abs_obs[:, 0] += traj_delta_x
        aug_abs_obs[:, 1] += traj_delta_y
        aug_abs_obs[:, 2] += traj_delta_x
        aug_abs_obs[:, 3] += traj_delta_y

        aug_abs_next_obs[:, 0] += traj_delta_x
        aug_abs_next_obs[:, 1] += traj_delta_y
        aug_abs_next_obs[:, 2] += traj_delta_x
        aug_abs_next_obs[:, 3] += traj_delta_y

        # if behavior is None:
        # Translate origin to the final ball position
        aug_abs_obs[:, 0] -= new_ball_final_pos_x
        aug_abs_obs[:, 1] -= new_ball_final_pos_y
        aug_abs_obs[:, 2] -= new_ball_final_pos_x
        aug_abs_obs[:, 3] -= new_ball_final_pos_y
        aug_abs_next_obs[:, 0] -= new_ball_final_pos_x
        aug_abs_next_obs[:, 1] -= new_ball_final_pos_y
        aug_abs_next_obs[:, 2] -= new_ball_final_pos_x
        aug_abs_next_obs[:, 3] -= new_ball_final_pos_y

        # rotate robot and ball position about ball's final position

        theta = np.random.uniform(*theta_range) * np.pi / 180
        M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        aug_abs_obs[:, :2] = M.dot(aug_abs_obs[:, :2].T).T
        aug_abs_obs[:, 2:4] = M.dot(aug_abs_obs[:, 2:4].T).T
        aug_abs_obs[:, 4] += theta

        aug_abs_next_obs[:, :2] = M.dot(aug_abs_next_obs[:, :2].T).T
        aug_abs_next_obs[:, 2:4] = M.dot(aug_abs_next_obs[:, 2:4].T).T
        aug_abs_next_obs[:, 4] += theta

        aug_abs_obs[:, 0] += new_ball_final_pos_x
        aug_abs_obs[:, 1] += new_ball_final_pos_y
        aug_abs_obs[:, 2] += new_ball_final_pos_x
        aug_abs_obs[:, 3] += new_ball_final_pos_y
        aug_abs_next_obs[:, 0] += new_ball_final_pos_x
        aug_abs_next_obs[:, 1] += new_ball_final_pos_y
        aug_abs_next_obs[:, 2] += new_ball_final_pos_x
        aug_abs_next_obs[:, 3] += new_ball_final_pos_y


        if np.random.random() < reflect:
            aug_abs_obs[:, 1] *= -1
            aug_abs_next_obs[:, 1] *= -1
            aug_abs_obs[:, 3] *= -1
            aug_abs_next_obs[:, 3] *= -1
            aug_abs_obs[:, 4] *= -1
            aug_abs_next_obs[:, 4] *= -1

            aug_action[:, 0] *= -1
            aug_action[:, 1] *= 1
            aug_action[:, 2] *= -1

        if behavior in ['straight_1', 'straight_2']:
            origin_x = np.random.uniform(4300, 4400)
            origin_y = np.random.uniform(-3000, -2000) + 300
            ball_init = np.array([origin_x, origin_y])
            noise = np.random.uniform(-50, 50, size=(len(aug_abs_obs), 2))
            aug_abs_obs[:, 2:4] = ball_init + noise
            aug_abs_next_obs[:, 2:4] = ball_init + noise




        # Verify that translation doesn't move the agent out of bounds.
        # Only need to check y since we only translate vertically.
        # break
        # check if agent and ball are in bounds
        if behavior == 'out_of_bounds' or check_in_bounds(env, aug_abs_obs):
            break
        else:
            # print(aug_abs_obs_original[200, :2])
            aug_abs_obs = copy.deepcopy(aug_abs_obs_original)
            aug_abs_next_obs = copy.deepcopy(aug_abs_next_obs_original)
        # print('theta', is_valid_theta)
    if attempts >= max_attempts:
        print(f'{behavior} Skipping trajectory after {attempts} augmentation attempts.')
        return None, None, None, None, None, None, None

    # aug_reward, aug_done = env.calculate_reward_vec(aug_abs_next_obs, aug_abs_obs)
    n = len(aug_abs_obs)
    aug_reward = np.empty(n)
    aug_obs = np.empty((n, 12))
    aug_next_obs = np.empty((n, 12))
    for i in range(n):
        aug_reward[i], ball_is_at_goal, ball_is_out_of_bounds = env.calculate_reward(aug_abs_next_obs[i])
        # if done[i]:
        #     aug_done[i] = True
        # else:
        #     aug_done[i] = ball_is_out_of_bounds
        aug_done[i] = ball_is_at_goal
        # aug_done[i] = False

        robot_pos = aug_abs_obs[i, :2]
        ball_pos = aug_abs_obs[i, 2:4]
        robot_angle = aug_abs_obs[i, 4]
        aug_obs[i] = env.get_obs(robot_pos, ball_pos, robot_angle)

        robot_pos = aug_abs_next_obs[i, :2]
        ball_pos = aug_abs_next_obs[i, 2:4]
        robot_angle = aug_abs_next_obs[i, 4]
        aug_next_obs[i] = env.get_obs(robot_pos, ball_pos, robot_angle)
        if aug_reward[i] < 0:
            print('neg')

    # if np.any(aug_abs_obs[:,0] > 5000):
    #     stop = 0

    # aug_done[-1] = True

    return aug_obs, aug_action, aug_reward, aug_next_obs, aug_done, aug_abs_obs, aug_abs_next_obs