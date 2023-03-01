import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from utils import get_paths, plot


if __name__ == "__main__":
    seaborn.set_theme()
    # env_ids = ['PandaPush-v3', 'PandaSlide-v3', 'PandaPickAndPlace-v3', 'PandaFlip-v3']
    # algo = 'ddpg'
    # palette = seaborn.color_palette('Spectral', 12)
    # seaborn.set_palette(palette)


    for policy_type in ['expert', 'random', 'expert_augmented', 'random_augmented']:
        path_dicts = {}

        for dataset_size in [10000, 50000, 100000]:

            root_dir = f'../logdata/{policy_type}_{dataset_size}'
            path_dict = get_paths(
                results_dir=f'{root_dir}',
                key=rf'{policy_type}_{dataset_size}')

            path_dicts.update(path_dict)


        fig = plt.figure()
        plot(path_dicts)
        plt.title(f'CQL', fontsize=16)
        plt.xlabel('Timesteps', fontsize=16)
        plt.ylabel('Success Rate', fontsize=16)
        # plt.ylim(0,1.05)
        plt.tight_layout()
        plt.legend()

        save_dir = f'figures'
        save_name = f'{policy_type}'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{save_name}')