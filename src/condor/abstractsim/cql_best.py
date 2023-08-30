from condor.common import gen_td3_bc
from condor.utils import MEMDISK

PARAMS = {
    'umaze': {
        'no_aug': (64, 64, 3e-4, 3e-4, 5),
        'random': (256, 64, 3e-5, 3e-5, 2.5),
        'guided': (256, 64, 3e-4, 3e-4, 5),
    },
    'medium': {
        'no_aug': (64, 64, 3e-5, 3e-5, 2.5),
        'random': (64, 64, 3e-4, 3e-4, 10),
        'guided': (64, 64, 3e-5, 3e-5, 2.5),
    },
    'large': {
        'no_aug': (64, 64, 3e-4, 3e-4, 2.5),
        'random': (64, 64, 3e-5, 3e-5, 10),
        'guided': (64, 64, 3e-4, 3e-4, 10),
    },
}

if __name__ == "__main__":
    all_commands = ""

    i = 0
    env_id = f'PushBallToGoal-v0'
    for aug in ['no_aug', 'random', 'guided']:

        dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}.hdf5'
        save_dir = f'results/{aug}/{env_id}/cql/'

        nl = 2
        policy_lr = 3e-5
        qf_lr = 3e-4
        gap = 1

        max_timesteps = int(1e6)
        eval_freq = int(20e3)

        command = f'python -u algorithms/cql.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
                  f' --save_dir {save_dir} ' \
                  f' --datset_name {dataset_name}' \
                  f' --env {env_id}' \
                  f' --qf_lr {qf_lr}' \
                  f' --policy_lr {policy_lr}' \
                  f' --cql_target_action_gap {gap}' \
                  f' --n_layers 2 --hidden_dims 64 --batch_size 64 ' \
                  f' --cql_min_q_weight 5  --cql_n_actions 5' \



        mem, disk = MEMDISK[1][env_id]
        memdisk = f'{mem},{disk},'
        command = memdisk + command.replace(' ', '*')
        print(command)
        i += 1
        all_commands += command + '\n'
