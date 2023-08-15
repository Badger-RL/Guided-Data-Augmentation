from condor.utils import MEMDISK

if __name__ == "__main__":
    all_commands = ""

    for env_id in ['PushBallToGoal-v2']:
        for aug in ['no_aug', 'random_traj', 'guided_traj']:
            aug = f'{aug}'
            # for aug in ['guided_transition']:
            for lr in [3e-4, 3e-5]:
                for lmbda in [0.5, 1, 2]:
                    for n_layers in [1,2]:
                        for hidden_dims in [128, 256]:

                            if n_layers == 1 and hidden_dims == 256: continue

                            dataset_name = f'/staging/ncorrado/datasets/{env_id}/physical/{aug}.hdf5'
                            save_dir = f'results/{env_id}/{aug}/awac/nl_{n_layers}/hd_{hidden_dims}/lr_{lr}/l_{lmbda}'

                            max_timesteps = int(1e6) if n_layers == 2 else int(1e6)
                            eval_freq = int(20e3)
                            command = f'python -u algorithms/awac.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
                                      f' --save_dir {save_dir} ' \
                                      f' --env {env_id}' \
                                      f' --learning_rate {lr}' \
                                      f' --awac_lambda {lmbda}' \
                                      f' --n_layers {n_layers}' \
                                      f' --hidden_dim {hidden_dims} '\
                                      f' --tau {5e-3}' \
                                      f' --gamma 0.99 --save-policy 1'

                            if dataset_name:
                                command += f' --dataset_name {dataset_name}'

                            mem, disk = MEMDISK[1][env_id]
                            command = f'{mem},{disk},' + command.replace(' ', '*')
                            print(command)
                            # print(command + f' --device cuda')
                            all_commands += command + '\n'

    # for aug in ['no_aug', 'random', 'guided']:
    #     for lr in [3e-4]:
    #         for lmbda in [0.5, 1, 2]:
    #             for n_layers in [2]:
    #                 for hidden_dims in [256]:
    #                     dataset_name = f'/staging/ncorrado/datasets/{env_id}/physical/{aug}.hdf5'
    #
    #                     save_dir = f'results/{aug}/{env_id}/physical/awac/l_{lmbda}'
    #
    #                     max_timesteps = int(1e6)
    #                     eval_freq = int(20e3)
    #                     command = f'python -u algorithms/awac.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
    #                               f' --save_dir {save_dir} ' \
    #                               f' --env {env_id}' \
    #                               f' --learning_rate {lr}' \
    #                               f' --awac_lambda {lmbda}' \
    #                               f' --n_layers {n_layers}' \
    #                               f' --hidden_dim {hidden_dims}'
    #
    #                     if dataset_name:
    #                         command += f' --dataset_name {dataset_name}'
    #
    #                     mem, disk = MEMDISK[1][env_id]
    #                     command = f'{mem},{disk},' + command.replace(' ', '*')
    #                     print(command)
    #                     # print(command + f' --device cuda')
    #                     all_commands += command + '\n'