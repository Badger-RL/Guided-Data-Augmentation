from condor.common import gen_td3_bc
from condor.utils import MEMDISK

if __name__ == "__main__":
    all_commands = ""

    i = 0
    for env_id in ['PushBallToGoal-v0']:
        for aug in ['no_aug', 'random', 'guided']:
            # for expert in [50, 85]:
            #     aug = f'{aug}_{expert}'
                aug = f'{aug}'
                for gap in [5]:
                        for nl in [1,2]:
                            dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}.hdf5'

                            save_dir = f'results/{env_id}/{aug}/td3bc/nl_{nl}/gap_{gap}'
                            hidden_dims = 256

                            max_timesteps = int(1e6) if nl == 2 else int(1e6)
                            eval_freq = int(20e4)

                            command = f'python -u algorithms/cql.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
                                      f' --save_dir {save_dir} ' \
                                      f' --env {env_id}' \
                                      f' --cql_target_action_gap {gap}' \
                                      f' --cql_min_q_weight 5'
                                      # f' --qf_lr {qf_lr}' \
                                      # f' --policy_lr {policy_lr}' \

                                      # f' --batch_size {batch_size}' \

                            if dataset_name:
                                command += f' --dataset_name {dataset_name}'

                            # command = command + f' --device cuda'
                            # command = '4,9,' + command.replace(' ', '*')
                            print(command)
                            i+=1
                            all_commands += command + '\n'