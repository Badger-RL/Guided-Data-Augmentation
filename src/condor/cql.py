
from src.condor.td3_bc import MEMDISK

if __name__ == "__main__":
    all_commands = ""
    i = 0

    for m in [1]:
        for env_id in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1',]:# 'antmaze-umaze-diverse-v1', 'antmaze-medium-diverse-v1', 'antmaze-large-diverse-v1']:
        # for env_id in ['antmaze-umaze-diverse-v1', 'antmaze-medium-diverse-v1', 'antmaze-large-diverse-v1']:
            for aug in ['no_aug', 'random', 'guided', 'mixed']:
                for policy_lr in [3e-4, 3e-5]:
                    for qf_lr in [3e-4]:
                        for gap in [5]:
                            # for n_layers in [2]:
                            #     for hidden_dims in [256]:
                            m = 1

                            if aug == 'no_aug':
                                save_dir = f'results/{aug}/{env_id}/cql/lr_{policy_lr}/lr_{qf_lr}/g_{gap}'
                                dataset_name = None
                                batch_size = 256
                            else:
                                save_dir = f'results/{aug}/{env_id}/cql/lr_{policy_lr}/lr_{qf_lr}/g_{gap}'
                                dataset_name = f'/staging/qu45/GuDA/datasets/{env_id}/{aug}/m_{m}.hdf5'
                                batch_size = 256

                            max_timesteps = int(1e6)
                            eval_freq = int(20e3)
                            command = f'python -u algorithms/cql.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
                                      f' --save_dir {save_dir} ' \
                                      f' --env {env_id}' \
                                      f' --qf_lr {qf_lr}' \
                                      f' --policy_lr {policy_lr}' \
                                      f' --cql_target_action_gap {gap}' \
                                      f' --batch_size {batch_size}' \
                                      # f' --cql_min_q_weight 5'

                            if dataset_name:
                                command += f' --dataset_name {dataset_name}'

                            # command = '4,9,' + command + f' --device cuda'
                            # command = command.replace(' ', '*')

                            mem, disk = MEMDISK[m][env_id]
                            command = f'{mem},{disk},' + command.replace(' ', '*')
                            # print(command)
                            print(command)
                            i+=1
                            all_commands += command + '\n'