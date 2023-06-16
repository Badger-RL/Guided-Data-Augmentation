from condor.common import gen_cql, gen_awac

PARAMS128 = {
    'maze2d-umaze-v1': {
        'no_aug': 0.5,
        'random': 0.5,
        'guided': 0.5,
    },
    'maze2d-medium-v1': {
        'no_aug': 0.5,
        'random': 0.5,
        'guided': 2,
    },
    'maze2d-large-v1': {
        'no_aug': 2,
        'random': 2,
        'guided': 1,
    },
}

MEMDISK = {
    1: {
        'maze2d-umaze-v1': (1.6, 9),
        'maze2d-medium-v1': (1.6, 9),
        'maze2d-large-v1': (2.9, 9),
    },
    4: {
        'maze2d-umaze-v1': (2.2, 9),
        'maze2d-medium-v1': (2.2, 9),
        'maze2d-large-v1': (3.5, 9),
    }
}



if __name__ == "__main__":
    all_commands = ""

    for env_id in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']:
        for aug in ['no_aug', 'random', 'guided']:
            lr = 3e-4
            n_layers = 1
            hidden_dims = 128
            lmbda = PARAMS128[env_id][aug]

            m = 1

            if aug == 'no_aug':
                dataset_name = None
            else:
                dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}/m_{m}.hdf5'
            if aug == 'no_aug':
                save_dir = f'results/{aug}/{env_id}/awac/'
                batch_size = 256
            else:
                save_dir = f'results/{aug}/m_{m}/{env_id}/awac/'
                batch_size = 256 * (m + 1)


            max_timesteps = int(1e6)
            eval_freq = int(10e3)
            command = f'python -u algorithms/awac.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
                      f' --save_dir {save_dir} ' \
                      f' --env {env_id}' \
                      f' --learning_rate {lr}' \
                      f' --batch_size {batch_size}' \
                      f' --awac_lambda {lmbda}' \
                      f' --hidden_dim {hidden_dims}'

            if dataset_name:
                command += f' --dataset_name {dataset_name}'

            mem, disk = MEMDISK[m][env_id]
            command = f'{mem},{disk},' + command.replace(' ', '*')
            print(command)
            # print(command + f' --device cuda')
            all_commands += command + '\n'