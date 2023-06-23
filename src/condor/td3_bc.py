from condor.common import gen_td3_bc, gen_cql

MEMDISK = {
    1: {
        'maze2d-umaze-v1': (1.6, 9),
        'maze2d-medium-v1': (1.6, 9),
        'maze2d-large-v1': (2.9, 9),

        'antmaze-umaze-diverse-v1': (2.2, 9),
        'antmaze-medium-diverse-v1': (2.2, 9),
        'antmaze-large-diverse-v1': (3.3, 9),
    },
    4: {
        'maze2d-umaze-v1': (2.2, 9),
        'maze2d-medium-v1': (2.2, 9),
        'maze2d-large-v1': (3.5, 9),
    }
}

if __name__ == "__main__":
    all_commands = ""

    i = 0
    for m in [1]:

        # for env_id in ['antmaze-medium-diverse-v1', 'antmaze-large-diverse-v1', 'antmaze-umaze-diverse-v1']:
        for env_id in ['maze2d-medium-v1', 'maze2d-large-v1', 'maze2d-umaze-v1']:
            for aug in ['random', 'guided', 'no_aug']:
                for lr in [3e-4]:
                    for alpha in [2.5, 5, 7.5, 10]:
                        for tau in [1e-3]:
                            if aug == 'no_aug':
                                dataset_name = None
                                save_dir = f'results/{aug}/{env_id}/td3bc/lr_{lr}/a_{alpha}/t_{tau}'
                                batch_size = 256
                            else:
                                save_dir = f'results/{aug}/m_{m}/{env_id}/td3bc/lr_{lr}/a_{alpha}/t_{tau}'
                                dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}/m_{m}.hdf5'
                                batch_size = 256*(m+1)
                            actor_lr, critic_lr = lr, lr
                            n_layers = 1
                            hidden_dims = 128

                            command = gen_td3_bc(
                                save_dir=save_dir,
                                max_timesteps=int(1e6),
                                eval_freq=int(10000),
                                dataset_name=dataset_name,
                                env_id=env_id,
                                actor_lr=actor_lr,
                                critic_lr=critic_lr,
                                alpha=alpha,
                                tau=tau,
                                n_layers=n_layers,
                                hidden_dims=hidden_dims
                            )
                            mem, disk = MEMDISK[m][env_id]
                            memdisk = f'{mem},{disk},'
                            # command = memdisk + command.replace(' ', '*')
                            print(command)
                            i += 1
                            all_commands += command + '\n'