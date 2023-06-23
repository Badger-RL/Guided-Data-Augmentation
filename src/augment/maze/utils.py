import numpy as np


def check_valid(env, aug_obs, aug_action, aug_reward, aug_next_obs, render=False, verbose=False):

    env.reset()

    valid = True
    for i in range(len(aug_obs)):
        qpos = aug_obs[i][:2]
        qvel = aug_obs[i][2:]

        env.set_state(qpos,qvel)
        next_obs, reward, done, info = env.step(aug_action[i])

        if render:
            env.render()

        # Augmented transitions at the goal are surely not valid, but that's fine.
        if not info['is_success']:
            if not np.allclose(next_obs, aug_next_obs[i], atol=1e-5):
                valid = False
                if verbose:
                    print(f'{i}, true next obs - aug next obs', aug_next_obs[i]-next_obs)
                    print(f'{i}, true next obs', next_obs)
                    print(f'{i}, aug next obs', aug_next_obs[i])

                    # print(aug_next_obs[i, 2:4], next_obs[2:4])

            if not np.isclose(reward, aug_reward[i], atol=1e-5):
                valid = False
                if verbose:
                    print(f'{i}, aug reward: {aug_reward[i]}\ttrue reward: {reward}')

        if not valid:
            break

    return valid