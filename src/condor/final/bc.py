from condor.utils import MEMDISK

PARAMS = {
    'maze2d-umaze-v1': {
        'no_aug': 3e-6,
        'random': 3e-6,
        'guided': 3e-6,
    },
    'maze2d-medium-v1': {
        'no_aug': 3e-6,
        'random': 3e-6,
        'guided': 3e-6,
    },
    'maze2d-large-v1': {
        'no_aug': 3e-6,
        'random': 3e-6,
        'guided': 3e-6,
    },
    'antmaze-umaze-diverse-v1': {
        'no_aug': 3e-6,
        'random': 3e-6,
        'guided': 3e-6,
    },
    'antmaze-medium-diverse-v1': {
        'no_aug': 3e-6,
        'random': 3e-6,
        'guided': 3e-6,
    },
    'antmaze-large-diverse-v1': {
        'no_aug': 3e-6,
        'random': 3e-6,
        'guided': 3e-6,
    },
    # 'PushBallToGoal-v0': {
    #     'no_aug': 3e-6,
    #     'random': 3e-6,
    #     'guided': 3e-6,
    # },
}

if __name__ == "__main__":
    all_commands = ""

    i = 0
    for env_id in PARAMS.keys():
        for aug in ['no_aug', 'random', 'guided']:

            if '2d' in env_id:
                dataset_name = f'/staging/ncorrado/datasets/{env_id}/small/{aug}.hdf5'
            else:
                dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}.hdf5'

            save_dir = f'results/{env_id}/{aug}/bc/'

            # nl = 2
            # hd = 256
            bs = 256
            lr = PARAMS[env_id][aug]
            max_timesteps = int(1e6)
            eval_freq = int(20e3)

            command = f'python -u algorithms/any_percent_bc.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
                      f' --save_dir {save_dir} ' \
                      f' --env {env_id} ' \
                      f' --dataset_name {dataset_name}' \
                      f' --learning_rate {lr}' \
                      f' --batch_size {bs}' \
                      # f' --n_layers {nl}' \
                      # f' --hidden_dims {hd}'



            mem, disk = MEMDISK[1][env_id]
            memdisk = f'{mem},{disk},'
            command = memdisk + command.replace(' ', '*')
            print(command)
            i += 1
            all_commands += command + '\n'
