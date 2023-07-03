
from src.condor.td3_bc import MEMDISK

if __name__ == "__main__":
    all_commands = ""
    n_layerss = [1, 2]
    hidden_dimss = [128, 256]
    for m in[1]:
        for env_id in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']:
            for aug in ['no_aug', 'random', 'guided']:
                for lr in [3e-4]:
                    for lmbda in [0.5, 1, 2]:
                        for arch_index in range(2):
                            n_layers = n_layerss[arch_index]
                            hidden_dims = hidden_dimss[arch_index]


                            if aug == 'no_aug':
                                dataset_name = None
                                save_dir = f'results/{aug}/{env_id}/awac/nl_{n_layers}/hd_{hidden_dims}/l_{lmbda}'
                            else:
                                save_dir = f'results/{aug}/{env_id}/awac/nl_{n_layers}/hd_{hidden_dims}/l_{lmbda}'
                                dataset_name = f'/staging/qu45/GuDA/datasets/{env_id}/{aug}/m_{m}.hdf5'

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
                            eval_freq = int(20e3)
                            command = f'python -u algorithms/awac.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
                                        f' --save_dir {save_dir} ' \
                                        f' --env {env_id}' \
                                        f' --learning_rate {lr}' \
                                        f' --awac_lambda {lmbda}' \
                                        f' --n_layers {n_layers}' \
                                        f' --hidden_dim {hidden_dims}'

                            if dataset_name:
                                command += f' --dataset_name {dataset_name}'

                            mem, disk = MEMDISK[m][env_id]
                            command = f'{mem},{disk},' + command.replace(' ', '*')
                            print(command)
                            # print(command + f' --device cuda')
                            all_commands += command + '\n'