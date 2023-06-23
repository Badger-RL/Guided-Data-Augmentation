import os

import numpy as np
import seaborn

import sys
from matplotlib import pyplot as plt

from utils import get_paths, plot, get_run_paths


if __name__ == "__main__":


    

    if not(len(sys.argv) == 2):
        print("usage: python3 ./plot.py <results path>")
        exit()


    paths = get_run_paths(sys.argv[1])
    plot(paths, success_rate = True)
    plt.title(f'CQL', fontsize=16)
    plt.xlabel('Timesteps', fontsize=16)
    plt.ylabel('Success Rate', fontsize=16)
    plt.tight_layout()    
    plt.legend()
    plt.savefig(f"./figures/{sys.argv[1].replace('/','*').replace('.','*')}_success_rate")

    fig = plt.figure()
    paths = get_run_paths(sys.argv[1])
    plot(paths, success_rate = False)    
    plt.title(f'CQL', fontsize=16)
    plt.axhline(y=3.44, color='k', linestyle='--')
    plt.xlabel('Timesteps', fontsize=16)
    plt.ylabel('Return', fontsize=16)
    plt.tight_layout()    
    plt.legend()





    plt.savefig(f"./figures/{sys.argv[1].replace('/','*').replace('.','*')}_return")


    



    """
    if not(len(sys.argv) == 2):
        print("usage: python3 ./plot.py <experiment type>")
        exit()






    seaborn.set_theme()
    # env_ids = ['PandaPush-v3', 'PandaSlide-v3', 'PandaPickAndPlace-v3', 'PandaFlip-v3']
    # algo = 'ddpg'
    # palette = seaborn.color_palette('Spectral', 12)
    # seaborn.set_palette(palette)


    path_dicts = {}

    for dataset_size in [50]:

        root_dir = f'../logdata/Exp_expert_no_aug_{dataset_size}k'
        path_dict = get_paths(
            results_dir=f'{root_dir}',
            key=rf'Exp_expert_no_aug_{dataset_size}k')

        path_dicts.update(path_dict)

    for dataset_size in ["50k_100k"]:

                root_dir = f'../logdata/Exp_expert_aug_uniform_{dataset_size}'
                path_dict = get_paths(
                    results_dir=f'{root_dir}',
                    key=rf'Exp_expert_aug_uniform_{dataset_size}')

                path_dicts.update(path_dict)

    for run_index in [0,1,2,3,4]:
                #for dataset_size in [100, 200]:

                root_dir = f'../logdata/ExpGuided_expert_aug_guided_{run_index}'#_{dataset_size}k'
                path_dict = get_paths(
                    results_dir=f'{root_dir}',
                    key=rf'ExpGuided_expert_aug_guided_{run_index}')#_{dataset_size}k')

                path_dicts.update(path_dict)


    fig = plt.figure()
    plot(path_dicts)
    plt.title(f'CQL', fontsize=16)
    plt.axhline(y=3.44, color='k', linestyle='--')
    plt.xlabel('Timesteps', fontsize=16)
    plt.ylabel('Return', fontsize=16)
    # plt.ylim(0,1.05)
    plt.tight_layout()
    plt.legend()

    save_dir = f'figures'
    save_name = f'{sys.argv[1]}_return'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}')

    # Plot success rate
    fig = plt.figure()
    plot(path_dicts, success_rate=True)
    plt.title(f'CQL', fontsize=16)
    plt.axhline(y=1, color='k', linestyle='--')
    plt.xlabel('Timesteps', fontsize=16)
    plt.ylabel('Success Rate', fontsize=16)
    plt.ylim(-0.05,1.05)
    plt.tight_layout()
    plt.legend()

    save_dir = f'figures'
    save_name = f'{sys.argv[1]}_success_rate'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}')




    exit()

    if sys.argv[1] == "Exp_expert_no_aug":

        path_dicts = {}

        for dataset_size in [10, 100, 50]:

            root_dir = f'../logdata/Exp_expert_no_aug_{dataset_size}k'
            path_dict = get_paths(
                results_dir=f'{root_dir}',
                key=rf'Exp_expert_no_aug_{dataset_size}k')

            path_dicts.update(path_dict)


        fig = plt.figure()
        plot(path_dicts)
        plt.title(f'CQL', fontsize=16)
        plt.axhline(y=3.44, color='k', linestyle='--')
        plt.xlabel('Timesteps', fontsize=16)
        plt.ylabel('Return', fontsize=16)
        # plt.ylim(0,1.05)
        plt.tight_layout()
        plt.legend()

        save_dir = f'figures'
        save_name = f'{sys.argv[1]}_return'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{save_name}')

        # Plot success rate
        fig = plt.figure()
        plot(path_dicts, success_rate=True)
        plt.title(f'CQL', fontsize=16)
        plt.axhline(y=1, color='k', linestyle='--')
        plt.xlabel('Timesteps', fontsize=16)
        plt.ylabel('Success Rate', fontsize=16)
        plt.ylim(-0.05,1.05)
        plt.tight_layout()
        plt.legend()

        save_dir = f'figures'
        save_name = f'{sys.argv[1]}_success_rate'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{save_name}')


    elif sys.argv[1] == "Exp_expert_aug_uniform":

        path_dicts = {}

        for dataset_size in ["10k_100k","10k_200k","50k_100k","50k_200k","100k_200k"]:

            root_dir = f'../logdata/Exp_expert_aug_uniform_{dataset_size}'
            path_dict = get_paths(
                results_dir=f'{root_dir}',
                key=rf'Exp_expert_aug_uniform_{dataset_size}')

            path_dicts.update(path_dict)


        fig = plt.figure()
        plot(path_dicts)
        plt.title(f'CQL', fontsize=16)
        plt.axhline(y=3.44, color='k', linestyle='--')
        plt.xlabel('Timesteps', fontsize=16)
        plt.ylabel('Return', fontsize=16)
        # plt.ylim(0,1.05)
        plt.tight_layout()
        plt.legend()

        save_dir = f'figures'
        save_name = f'{sys.argv[1]}_return'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{save_name}')

        # Plot success rate
        fig = plt.figure()
        plot(path_dicts, success_rate=True)
        plt.title(f'CQL', fontsize=16)
        plt.axhline(y=1, color='k', linestyle='--')
        plt.xlabel('Timesteps', fontsize=16)
        plt.ylabel('Success Rate', fontsize=16)
        plt.ylim(-0.05,1.05)
        plt.tight_layout()
        plt.legend()

        save_dir = f'figures'
        save_name = f'{sys.argv[1]}_success_rate'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{save_name}')


    elif sys.argv[1] == "ExpGuided_expert_aug_guided":

        path_dicts = {}

        for run_index in [0,1,2,3,4]:
            #for dataset_size in [100, 200]:

            root_dir = f'../logdata/ExpGuided_expert_aug_guided_{run_index}'#_{dataset_size}k'
            path_dict = get_paths(
                results_dir=f'{root_dir}',
                key=rf'ExpGuided_expert_aug_guided_{run_index}')#_{dataset_size}k')

            path_dicts.update(path_dict)


        fig = plt.figure()
        plot(path_dicts)
        plt.title(f'CQL', fontsize=16)
        plt.axhline(y=3.44, color='k', linestyle='--')
        plt.xlabel('Timesteps', fontsize=16)
        plt.ylabel('Return', fontsize=16)
        # plt.ylim(0,1.05)
        plt.tight_layout()
        plt.legend()

        save_dir = f'figures'
        save_name = f'{sys.argv[1]}_return'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{save_name}')

        # Plot success rate
        fig = plt.figure()
        plot(path_dicts, success_rate=True)
        plt.title(f'CQL', fontsize=16)
        plt.axhline(y=1, color='k', linestyle='--')
        plt.xlabel('Timesteps', fontsize=16)
        plt.ylabel('Success Rate', fontsize=16)
        plt.ylim(-0.05,1.05)
        plt.tight_layout()
        plt.legend()

        save_dir = f'figures'
        save_name = f'{sys.argv[1]}_success_rate'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{save_name}')

    """
    """
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
        plt.axhline(y=3.44, color='k', linestyle='--')
        plt.xlabel('Timesteps', fontsize=16)
        plt.ylabel('Return', fontsize=16)
        # plt.ylim(0,1.05)
        plt.tight_layout()
        plt.legend()

        save_dir = f'figures'
        save_name = f'{policy_type}_return'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{save_name}')

        # Plot success rate
        fig = plt.figure()
        plot(path_dicts, success_rate=True)
        plt.title(f'CQL', fontsize=16)
        plt.axhline(y=1, color='k', linestyle='--')
        plt.xlabel('Timesteps', fontsize=16)
        plt.ylabel('Success Rate', fontsize=16)
        plt.ylim(-0.05,1.05)
        plt.tight_layout()
        plt.legend()

        save_dir = f'figures'
        save_name = f'{policy_type}_success_rate'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{save_name}')
    """