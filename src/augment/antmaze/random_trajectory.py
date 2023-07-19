from collections import defaultdict

import numpy as np

from augment.antmaze.antmaze_aug_function import AntMazeAugmentationFunction


class AntMazeRandomTrajAugmentationFunction(AntMazeAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.num_aug = 0

        # precompute quaternion conversions
        self.theta_to_quat = np.array([
            [1, 0, 0, 0],
            [0.707, 0, 0, 0.707],
            [1, 0, 0, 0],
            [0.707, 0, 0, -0.707],
        ])
        if self.env.maze_arr.shape[0] == 5:
            self.guide_thetas = {
                (1, 1): [0],
                (2, 1): [0],
                (3, 1): [np.pi / 2],

                (3, 2): [np.pi / 2],

                (1, 3): [0, np.pi / 2, np.pi, np.pi * 3 / 2],
                (2, 3): [np.pi],
                (3, 3): [np.pi]
            }

    def _is_in_box(self, xlo, xhi, ylo, yhi, x, y):
        if (x > xlo and x < xhi) and (y > ylo and y < yhi):
            return True
        else:
            return False

    def _sample_umaze(self, obs, last_obs):
        x, y = obs[0], obs[1]
        probs = np.array([4.25, 4, 4.5])/12.75

        region = np.random.choice(np.arange(3), p=probs)
        # bottom
        if region == 0:
            new_pos = np.random.uniform(
                low=np.array([0, 0]),
                high=np.array([8.5, 0.5])
            )

        # right side
        if region == 1:
            new_pos = np.random.uniform(
                low=np.array([8.5, 0]),
                high=np.array([9, 8])
            )
        if region == 2:
                new_pos = np.random.uniform(
                low=np.array([0, 8]),
                high=np.array([9, 8.5])
            )
        else:
            new_pos = None

        return new_pos

    def _is_valid_umaze(self, obs):
        x, y = obs[:,0], obs[:,1]

        # bottom
        mask1 = (x > 0) & (x < 9) & (y > -1) & (y < 1)
        # right side
        mask2 = (x > 7) & (x < 9) & (y > 0) & (y < 8)
        # top
        mask3 = (x > 0) & (x < 9) & (y > 7) & (y < 9)

        return mask1 | mask2 | mask3

    def _is_in_wall(self, box_location, x, y):
        xlo, ylo = box_location*4 - self.agent_offset - 2 - self.effective_wall_width
        xhi, yhi = box_location*4 - self.agent_offset + 2 + self.effective_wall_width

        if (x > xlo and y > ylo) and (x < xhi and y < yhi):
            return True
        else:
            return False

    def _is_done(self, aug_next_obs, aug_reward):
        aug_done = ~((aug_next_obs[:,2] >= 0.2) & (aug_next_obs[:, 2] <= 1.0)) | (aug_reward > 0)
        return aug_done

    def _reward(self, next_obs):
        # Rewar dshould intuitively be computed using next_obs, but D4RL uses the current obs (D4RL bug)
        if self.env.reward_type == 'sparse':
            # print(np.linalg.norm(next_obs[0:2] - (self.target )))
            reward = (np.linalg.norm(next_obs[:,0:2] - self.target, axis=-1) <= 0.5).astype(float)
        elif self.env.reward_type == 'dense':
            reward = np.exp(-np.linalg.norm(next_obs[:,0:2] - self.target), axis=-1)
        else:
            raise ValueError('Unknown reward type %s' % self.env.reward_type)

        return reward

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs, ):

        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()

        new_pos = self._sample_umaze(obs[0])

        if new_pos is None:
            return None, None, None, None, None

        delta_pos = next_obs[:,:2] - obs[:,:2]
        delta_new_pos = new_pos - obs[0,:2]

        aug_obs[:,:2] += delta_new_pos
        aug_next_obs[:,:2] = aug_obs[:,:2] + delta_pos

        aug_action = action.copy()
        aug_reward = self._reward(aug_next_obs)
        aug_done = self._is_done(aug_next_obs, aug_reward)
        self.num_aug = 0

        mask = self._is_valid_umaze(aug_obs)

        return aug_obs[mask], aug_action[mask], aug_reward[mask], aug_next_obs[mask], aug_done[mask]


    def _get_guided_theta_umaze(self, new_pos):

        if new_pos[0] < 7.5 and new_pos[1] > -2 and new_pos[1] < 2:
            guide_theta = 0
        elif new_pos[0] > 7.5 and new_pos[0] < 9 and new_pos[1] < 7.5:
            guide_theta = np.pi / 2
        else:
            guide_theta = np.pi

        return guide_theta

    def _sample_theta(self, obs, next_obs, new_pos, new_location, **kwargs):
        guide_theta = self._get_guided_theta_umaze(new_pos)
        if guide_theta == 0:
            stop = 0
        # guide_theta = np.pi*3/2
        delta_obs = next_obs[:2] - obs[:2]
        theta = np.arctan2(delta_obs[1], delta_obs[0])
        aug_theta = -(guide_theta - theta) #+ np.random.uniform(low=-np.pi/6, high=np.pi/6)

        return aug_theta

    def augment(self,
                obs: np.ndarray,
                action: np.ndarray,
                next_obs: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                **kwargs,):

        # if not self.is_valid_input(obs, next_obs):
        #     return None, None, None, None, None
        # displacement = next_obs[-1,:2] - obs[0,:2]
        # if np.linalg.norm(displacement) < 1:
        #     return None, None, None, None, None
        #
        aug_obs = obs.copy()
        aug_next_obs = next_obs.copy()
        #
        # new_pos = self._sample_umaze(obs[0], aug_next_obs[-1, :2])
        new_pos, new_location = self._sample_pos()

        if new_pos is None:
            return None, None, None, None, None

        offset = new_pos - obs[0,:2]
        aug_obs[:,:2] += offset
        aug_next_obs[:,:2] += offset
        # aug_location = self._xy_to_rowcol(aug_obs[0,:2])
        #
        # rotate_alpha = self._sample_theta(aug_obs[0,:2], aug_next_obs[-1,:2], new_pos, aug_location)
        #
        # M = np.array([
        #     [np.cos(-rotate_alpha), -np.sin(-rotate_alpha)],
        #     [np.sin(-rotate_alpha), np.cos(-rotate_alpha)]
        # ])
        #
        # # translate initial position to 0,0 before rotation
        # origin = aug_obs[0, :2].copy()
        # aug_obs[:,:2] -= origin
        # aug_next_obs[:,:2] -= origin
        #
        # aug_obs[:,:2] = M.dot(aug_obs[:,:2].T).T
        # aug_next_obs[:,:2] = M.dot(aug_next_obs[:,:2].T).T
        #
        # # translate back to initial position after rotation
        # aug_obs[:,:2] += origin
        # aug_next_obs[:,:2] += origin
        # # delta_pos = next_obs[:,:2] - obs[:,:2]
        # # rotated_delta_obs = M.dot(delta_pos[:,:2].T).T
        # # aug_next_obs[:,:2] = aug_obs[:,:2] + rotated_delta_obs
        #
        #
        #
        # # mujoco stores quats as (qw, qx, qy, qz) internally but uses (qx, qy, qz, qw) in the observation
        # for i in range(len(obs)):
        #     sin = np.sin(rotate_alpha / 2)
        #     cos = np.cos(rotate_alpha / 2)
        #     quat_rotate_by = np.array([sin, 0, 0, cos])
        #     self._rotate_torso(aug_obs[i], quat_rotate_by)
        #     self._rotate_torso(aug_next_obs[i], quat_rotate_by)
        #
        #     sin = np.sin(-rotate_alpha)
        #     cos = np.cos(-rotate_alpha)
        #     self._rotate_vel(aug_obs[i], sin, cos)
        #     self._rotate_vel(aug_next_obs[i], sin, cos)

        aug_action = action.copy()
        # aug_reward = reward
        # aug_done = done
        aug_reward = self._reward(aug_next_obs)
        aug_done = self._is_done(aug_next_obs, aug_reward)
        mask = self._is_valid_umaze(aug_obs)
        # mask = np.ones_like(reward).astype(bool)

        return aug_obs[mask], aug_action[mask], aug_reward[mask], aug_next_obs[mask], aug_done[mask]
