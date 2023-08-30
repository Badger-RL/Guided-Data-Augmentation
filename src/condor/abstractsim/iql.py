from condor.utils import MEMDISK

if __name__ == "__main__":
    all_commands = ""

    # for env_id in ['antmaze-umaze-diverse-v1']:
    for env_id in ['antmaze-umaze-diverse-v1', 'antmaze-medium-diverse-v1', 'antmaze-large-diverse-v1',
                   'maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1',
                   ]:
        for aug in ['no_aug', 'random', 'guided']:
            for lrs in [(3e-5, 3e-4),]:
                for beta in [1, 5, 10]:
                    for nl in [2]:
                        if nl == 1:
                            hd = 128
                            bs = 64
                        if nl == 2:
                            hd = 64
                            bs = 64

                        actor_lr, critic_lr = lrs

                        dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}.hdf5'
                        save_dir = f'results/{aug}/{env_id}/iql/lr_{actor_lr}/lr_{critic_lr}/b_{beta}'


                        max_timesteps = int(1e6)
                        eval_freq = int(20e3)

                        command = f'python -u algorithms/iql.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
                                  f' --save_dir {save_dir} ' \
                                  f' --env {env_id}' \
                                  f' --actor_lr {actor_lr}' \
                                  f' --qf_lr {critic_lr}' \
                                  f' --vf_lr 3e-5' \
                                  f' --beta {beta}' \
                                  f' --batch_size 64' \
                                  # f' --n_layers {nl}' \
                                  # f' --hidden_dim {hd}'

                        # if 'ant' in env_id:
                        #     command += ' --normalize_reward 1'

                        if dataset_name:
                            command += f' --dataset_name {dataset_name}'

                        mem, disk = MEMDISK[1][env_id]
                        # command = f'{mem},{disk},' + command.replace(' ', '*')
                        print(command)
                        # print(command + f' --device cuda')
                        all_commands += command + '\n'

