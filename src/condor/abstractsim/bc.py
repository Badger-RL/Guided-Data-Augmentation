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
    for env_id in ['PushBallToGoal-v0']:
        for aug in ['mocoda']:
                for lr in [3e-4, 3e-5, 3e-6]:
                    # for hd in [64, 256]:
                        dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}.hdf5'
                        nl = 2
                        bs = 256
                        save_dir = f'results/{env_id}/{aug}/bc/lr_{lr}'

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
