from condor.utils import MEMDISK

PARAMS = {
    'maze2d-umaze-v1': {
        'no_aug': 5,
        'random': 1,
        'guided': 10,
    },
    'maze2d-medium-v1': {
        'no_aug': 10,
        'random': 10,
        'guided': 10,
    },
    'maze2d-large-v1': {
        'no_aug': 10,
        'random': 5,
        'guided': 10,
    },
    'antmaze-umaze-diverse-v1': {
        'no_aug': 1,
        'random': 10,
        'guided': 5,
    },
    'antmaze-medium-diverse-v1': {
        'no_aug': 1,
        'random': 10,
        'guided': 5,
    },
    'antmaze-large-diverse-v1': {
        'no_aug': 1,
        'random': 10,
        'guided': 5,
    },
    # 'PushBallToGoal-v0': {
    #     'no_aug': 3e-6,
    #     'random': 3e-6,
    #     'guided': 3e-6,
    # },
}

if __name__ == "__main__":
    all_commands = ""

    # for env_id in ['antmaze-umaze-diverse-v1']:
    for env_id in PARAMS.keys():
        for aug in ['no_aug', 'random', 'guided']:

            if '2d' in env_id:
                dataset_name = f'/staging/ncorrado/datasets/{env_id}/small/{aug}.hdf5'
            else:
                dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}.hdf5'

            # for lrs in [(3e-5, 3e-4),]:
            #     for beta in [1, 5, 10]:
            #         for nl in [2]:
            #             if nl == 1:
            #                 hd = 128
            #                 bs = 64
            #             if nl == 2:
            #                 hd = 64
            #                 bs = 64
            save_dir = f'results/{env_id}/{aug}/iql/'

            # nl = 2
            # hd = 256
            bs = 64
            actor_lr, critic_lr = 3e-5, 3e-4
            beta = PARAMS[env_id][aug]

            max_timesteps = int(1e6)
            eval_freq = int(20e3)

            if '2d' in env_id:
                dataset_name = f'/staging/ncorrado/datasets/{env_id}/small/{aug}.hdf5'
            else:
                dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}.hdf5'
            save_dir = f'results/{aug}/{env_id}/iql/lr_{actor_lr}/lr_{critic_lr}/b_{beta}'


            max_timesteps = int(1e6)
            eval_freq = int(20e3)
            iql_tau = 0.7 if '2d' in env_id else 0.9

            command = f'python -u algorithms/iql.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
                      f' --save_dir {save_dir} ' \
                      f' --env {env_id}' \
                      f' --actor_lr {actor_lr}' \
                      f' --qf_lr {critic_lr}' \
                      f' --beta {beta}' \
                      f' --batch_size 64 --reward_bias -1' \
                      # f' --n_layers {nl}' \
                      # f' --hidden_dim {hd}'
            # f' --vf_lr 3e-5' \

            # if 'ant' in env_id:
            #     command += ' --normalize_reward 1'

            if dataset_name:
                command += f' --dataset_name {dataset_name}'

            mem, disk = MEMDISK[1][env_id]
            command = f'{mem},{disk},' + command.replace(' ', '*')
            print(command)
            # print(command + f' --device cuda')
            all_commands += command + '\n'

