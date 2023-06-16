from condor.common import gen_cql, gen_awac

MEMDISK = {
    1: {
        'maze2d-umaze-v1': (1.6, 9),
        'maze2d-medium-v1': (1.6, 9),
        'maze2d-large-v1': (2.9, 9),

        'antmaze-umaze-diverse-v1': (2.2, 9),
        'antmaze-medium-diverse-v1': (2.2, 9),
        'antmaze-large-diverse-v1': (3.3, 9),
    },
    4: {
        'maze2d-umaze-v1': (2.2, 9),
        'maze2d-medium-v1': (2.2, 9),
        'maze2d-large-v1': (3.5, 9),
    }
}

if __name__ == "__main__":
    all_commands = ""

    for m in[1]:
        # for env_id in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']:
        for env_id in ['antmaze-medium-diverse-v1', 'antmaze-large-diverse-v1', 'antmaze-umaze-diverse-v1']:
            for aug in ['no_aug', 'random', 'guided']:
                for lr in [3e-4]:
                    for lmbda in [0.5, 1, 2]:
                        for n_layers in [1]:
                            for hidden_dims in [128]:

                                if aug == 'no_aug':
                                    dataset_name = None
                                    save_dir = f'results/{aug}/{env_id}/awac/l_{lmbda}'
                                else:
                                    save_dir = f'results/{aug}/m_{m}/{env_id}/awac/l_{lmbda}'
                                    dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}/m_{m}.hdf5'

                                # command = gen_awac(
                                #     save_dir=save_dir,
                                #     max_timesteps=int(1e6),
                                #     eval_freq=int(10000),
                                #     dataset_name=dataset_name,
                                #     env_id=env_id,
                                #     lr=lr,
                                #     lmbda=lmbda,
                                # )

                                max_timesteps = int(1e6)
                                eval_freq = int(10e3)
                                command = f'python -u algorithms/awac.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
                                          f' --save_dir {save_dir} ' \
                                          f' --env {env_id}' \
                                          f' --learning_rate {lr}' \
                                          f' --awac_lambda {lmbda}' \
                                          f' --hidden_dim {hidden_dims}'

                                if dataset_name:
                                    command += f' --dataset_name {dataset_name}'

                                mem, disk = MEMDISK[m][env_id]
                                command = f'{mem},{disk},' + command.replace(' ', '*')
                                print(command)
                                # print(command + f' --device cuda')
                                all_commands += command + '\n'