from condor.common import gen_td3_bc, gen_cql

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
'''
'maze2d-medium-v1': {
    'no_aug': f'data/td3_bc_smaller/results/no_aug/maze2d-medium-v1/td3_bc/alr_0.0003/clr_0.0003/a_5/t_0.005/',
    'random': f'data/td3_bc_smaller/results/random/maze2d-medium-v1/td3_bc/alr_0.0003/clr_0.0003/a_10/t_0.001/',
    'guided': f'data/td3_bc_smaller/results/guided/maze2d-medium-v1/td3_bc/alr_0.0003/clr_0.0003/a_10/t_0.001/',
},
'maze2d-large-v1': {
    'no_aug': f'data/td3_bc_smaller/results/no_aug/maze2d-large-v1/td3_bc/alr_0.0003/clr_0.0003/a_5/t_0.001/',
    'random': f'data/td3_bc_smaller/results/random/maze2d-large-v1/td3_bc/alr_0.0003/clr_0.0003/a_10/t_0.001/',
    'guided': f'data/td3_bc_smaller/results/guided/maze2d-large-v1/td3_bc/alr_0.0003/clr_0.0003/a_7.5/t_0.001/',
},
'''

PARAMS = {
    'maze2d-umaze-v1': {
        'no_aug': (10, 1e-3),
        'random': (7.5, 1e-3),
        'guided': (10, 1e-3),
    },
    'maze2d-medium-v1': {
        'no_aug': (5, 1e-3),
        'random': (7.5, 1e-3),
        'guided': (7.5, 1e-3),
    },
    'maze2d-large-v1': {
        'no_aug': (10, 1e-3),
        'random': (10, 1e-3),
        'guided': (5, 1e-3), # or 10
    },
}

PARAMS_LARGE = {
    'maze2d-umaze-v1': {
        'no_aug': (10, 1e-3),
        'random': (10, 1e-3),
        'guided': (10, 1e-3),
    },
    'maze2d-medium-v1': {
        'no_aug': (10, 1e-3),
        'random': (5, 1e-3),
        'guided': (7.5, 1e-3),
    },
    'maze2d-large-v1': {
        'no_aug': (10, 1e-3),
        'random': (7.5, 1e-3),
        'guided': (5, 1e-3), # or 10
    },
}

if __name__ == "__main__":
    all_commands = ""

    for env_id in ['maze2d-medium-v1', 'maze2d-umaze-v1', 'maze2d-large-v1']:
        for aug in ['random', 'guided', 'no_aug']:
            for hidden_dims in [256]:

                actor_lr = 3e-4
                critic_lr = 3e-4
                n_layers = 2
                m = 1
                alpha, tau = PARAMS_LARGE[env_id][aug]


                if aug == 'no_aug':
                    dataset_name = None
                else:
                    dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}/m_{m}.hdf5'
                if aug == 'no_aug':
                    save_dir = f'results/{aug}/{env_id}/td3bc/'
                else:
                    save_dir = f'results/{aug}/m_{m}/{env_id}/td3bc/'

                batch_size = 256
                # if aug == 'no_aug':
                #     batch_size = 256
                # else:
                #     batch_size = 256*(m+1)
                command = gen_td3_bc(
                    save_dir=save_dir,
                    max_timesteps=int(1e6),
                    eval_freq=int(20e3),
                    dataset_name=dataset_name,
                    env_id=env_id,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    alpha=alpha,
                    tau=tau,
                    n_layers=n_layers,
                    hidden_dims=hidden_dims,
                    batch_size=batch_size
                )
                mem, disk = MEMDISK[m][env_id]
                memdisk = f'{mem},{disk},'
                command = memdisk + command.replace(' ', '*')
                print(command)
                all_commands += command + '\n'