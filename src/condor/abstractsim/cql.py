from condor.common import gen_td3_bc, gen_cql
from condor.utils import MEMDISK

if __name__ == "__main__":
    all_commands = ""
    i = 0
    for env_id in ['PushBallToGoal-v0']:
        for aug in ['guided', 'no_aug', 'random', ]:
            for policy_lr, qf_lr in [(3e-5, 3e-4)]:
                for gap in [-5, -3, -1, 1]:
                    # for n_layers in [2]:
                    #     for hidden_dims in [256]:
                            m = 1

                            dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}.hdf5'

                            save_dir = f'results/{aug}/{env_id}/cql/lr_{policy_lr}/lr_{qf_lr}/g_{gap}'
                            batch_size = 64
                            max_timesteps = int(1e6)
                            eval_freq = int(20e3)
                            command = f'python -u algorithms/cql.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
                                      f' --save_dir {save_dir} ' \
                                      f' --env {env_id}' \
                                      f' --qf_lr {qf_lr}' \
                                      f' --policy_lr {policy_lr}' \
                                      f' --cql_target_action_gap {gap}' \
                                      f' --batch_size {batch_size}' \
                                      f' --n_layers 2 --hidden_dims 64 --batch_size 64 ' \
                                      f' --cql_min_q_weight 5 --cql_n_actions 10' \
                                      # f' --cql_max_target_backup 1 --cql_clip_diff_min -200'

                # cql_max_target_backup: bool = True  # Use max target backup
                # cql_clip_diff_min: float = -200  # Q-function lower loss clipping

                            if dataset_name:
                                command += f' --dataset_name {dataset_name}'

                            # command = '4,9,' + command + f' --device cuda'
                            mem, disk = MEMDISK[1][env_id]
                            command = f'{mem},{disk},' + command.replace(' ', '*')
                            print(command)
                            i+=1
                            all_commands += command + '\n'