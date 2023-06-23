import math

import numpy as np


class AugmentationFunctionBase:

    def __init__(self, env=None, **kwargs):
        self.env = env

    def _deepcopy_transition(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
    ):
        copy_obs = obs.copy()
        copy_next_obs = next_obs.copy()
        copy_action = action.copy()
        copy_reward = reward.copy()
        copy_done = done.copy()

        return copy_obs, copy_next_obs, copy_action, copy_reward, copy_done

    def augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 **kwargs,):
        '''
        Return an augmentation of the inputted transition.

        :param obs: observed obs
        :param next_obs: observed next_obs
        :param action: observed action
        :param reward: observed reward
        :param done:  observed done signal
        :param kwargs: augmentation function keyword arguments
        :return: augmented transition (aug
        '''

        # deepcopy input transition
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done = \
            self._deepcopy_transition(obs, next_obs, action, reward, done)

        # augment the deepcopy of the inputted transition in-place
        return self._augment(aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, **kwargs)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 **kwargs,):
        raise NotImplementedError("Augmentation function not implemented.")