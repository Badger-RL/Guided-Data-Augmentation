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

    for env_id in ['PushBallToGoal-v0']:
    # for env_id in PARAMS.keys():
        for aug in ['no_aug', 'random', 'guided']:

            if '2d' in env_id:
                dataset_name = f'/staging/ncorrado/datasets/{env_id}/small/{aug}.hdf5'
            else:
                dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}.hdf5'

            for actor_lr in [3e-4, 3e-5]:
                for critic_lr in [3e-4, 3e-5]:
                    if actor_lr !=  critic_lr: continue
                    for vf_lr in [3e-4, 3e-5]:
                        for beta in [1, 5, 10]:
                            for tau in [0.7, 0.9]:
                                # for nl in [2]:
                                #     if nl == 1:
                                #         hd = 128
                                #         bs = 64
                                #     if nl == 2:
                                #         hd = 64
                                #         bs = 64
                                nl = 2
                                hd = 256
                                bs = 256

                                save_dir = f'results/{env_id}/{aug}/iql/lr_{actor_lr}/lr_{critic_lr}/lr_{vf_lr}/b_{beta}/t_{tau}'


                                max_timesteps = int(1e6)
                                eval_freq = int(20e3)

                                if '2d' in env_id:
                                    dataset_name = f'/staging/ncorrado/datasets/{env_id}/small/{aug}.hdf5'
                                else:
                                    dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}.hdf5'


                                max_timesteps = int(1e6)
                                eval_freq = int(20e3)

                                command = f'python -u algorithms/iql.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
                                          f' --save_dir {save_dir} ' \
                                          f' --env {env_id}' \
                                          f' --actor_lr {actor_lr}' \
                                          f' --qf_lr {critic_lr}' \
                                          f' --vf_lr {vf_lr}' \
                                          f' --beta {beta}' \
                                          f' --iql_tau {tau}' \
                                          f' --n_layers 2 --hidden_dims 64 --batch_size {bs}' \
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

