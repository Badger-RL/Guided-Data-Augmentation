from condor.common import gen_cql, gen_awac

PARAMS = { # nl = 1, hd = 128
    'maze2d-umaze-v1': {
        'no_aug': 0.5,
        'random': 0.5,
        'guided': 0.5,
    },
    'maze2d-medium-v1': {
        'no_aug': 0.25,
        'random': 0.25,
        'guided': 0.25,
    },
    'maze2d-large-v1': {
        'no_aug': 0.25,
        'random': 2,
        'guided': 2,
    },
}

PARAMS_LARGE = { # nl = 2, hd = 256
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
        'no_aug': 0.5,
        'random': 1,
        'guided': 2,
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
        # for env_id in ['antmaze-medium-diverse-v1', 'antmaze-large-diverse-v1', 'antmaze-umaze-diverse-v1']:
        for aug in ['no_aug', 'random', 'guided']:

            m = 1
            n_layers = 2
            hidden_dims = 256
            lr = 3e-4
            lmbda = PARAMS_LARGE[env_id][aug]
            save_dir = f'results/{aug}/{env_id}/awac/'

            if aug == 'no_aug':
                dataset_name = None
            else:
                dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}/m_{m}.hdf5'

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