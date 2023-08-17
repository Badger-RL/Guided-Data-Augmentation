# Derived from D4RL
# https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE
import argparse
import os
import time
from collections import defaultdict

import gym, d4rl
import h5py
import numpy as np
import random

from augment.maze.point_maze_aug_function import PointMazeAugmentationFunction
from generate.utils import reset_data, append_data, npify, load_dataset

if __name__ == '__main__':
    # command = 'python relabel_rewards.py --env-id antmaze-umaze-diverse-v1 --save-dir ../../datasets/antmaze-umaze-diverse-v1 --save-name no_aug_relabeled.hdf5'
    # command = 'python relabel_rewards.py --env-id antmaze-umaze-diverse-v1 --dataset-path ../../datasets/antmaze-umaze-diverse-v1/no_aug_no_collisions.hdf5 --save-dir ../../datasets/antmaze-umaze-diverse-v1 --save-name no_aug_no_collisions_relabeled.hdf5'
    # command = 'python relabel_rewards.py --env-id antmaze-umaze-diverse-v1 --dataset-path ../../datasets/antmaze-umaze-diverse-v1/no_aug_no_collisions2.hdf5 --save-dir ../../datasets/antmaze-umaze-diverse-v1 --save-name no_aug_no_collisions_relabeled.hdf5'
    # command = 'python relabel_rewards.py --env-id antmaze-medium-diverse-v1 --save-dir ../../datasets/antmaze-medium-diverse-v1 --save-name no_aug_relabeled.hdf5'
    # command = 'python relabel_rewards.py --env-id antmaze-medium-diverse-v1 --save-dir ../../datasets/antmaze-medium-diverse-v1 --save-name no_aug_relabeled.hdf5'
    # command = 'python relabel_rewards.py --env-id antmaze-medium-diverse-v1 --dataset-path ../../datasets/antmaze-medium-diverse-v1/no_aug_no_collisions.hdf5 --save-dir ../../datasets/antmaze-medium-diverse-v1 --save-name no_aug_no_collisions_relabeled.hdf5'
    # command = 'python relabel_rewards.py --env-id antmaze-large-diverse-v1 --save-dir ../../datasets/antmaze-large-diverse-v1 --save-name no_aug_relabeled.hdf5'


    command = 'python relabel_rewards.py --env-id antmaze-umaze-diverse-v1 --dataset-path ../../datasets/antmaze-umaze-diverse-v1/no_aug_no_collisions_relabeled.hdf5 --save-dir ../../datasets/antmaze-umaze-diverse-v1 --save-name no_aug_no_collisions_relabeled.hdf5'
    # command = 'python relabel_rewards.py --env-id antmaze-umaze-diverse-v2 --dataset-path ../../datasets/antmaze-umaze-diverse-v2/no_aug_no_collisions_1k.hdf5 --save-dir ../../datasets/antmaze-umaze-diverse-v2 --save-name no_aug_no_collisions_relabeled_1k.hdf5'

    os.system(command)