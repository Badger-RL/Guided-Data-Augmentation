from src.condor.common import gen_td3_bc

MEMDISK = {
    1: {
        'maze2d-umaze-v1': (1.6, 9),
        'maze2d-medium-v1': (1.6, 9),
        'maze2d-large-v1': (2.9, 9),

        'antmaze-umaze-diverse-v1': (2.2, 9),
        'antmaze-medium-diverse-v1': (2.2, 9),
        'antmaze-large-diverse-v1': (3.3, 9),
    },
    2: {
        'maze2d-umaze-v1': (1.9, 9),
        'maze2d-medium-v1': (1.9, 9),
        'maze2d-large-v1': (3.2, 9),
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

    actor_lr, critic_lr = 3e-4, 3e-4
    lr = 3e-4
    n_layers = 2
    hd = 256
    tau = 1e-3

    for m in [1]:
        # for env_id in ['antmaze-medium-diverse-v1', 'antmaze-large-diverse-v1', 'antmaze-umaze-diverse-v1']:
        for env_id in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']:
            for aug in ['random', 'guided', 'no_aug']:
                for alpha in [2.5, 5, 7.5, 10]:
                    if aug == 'no_aug':
                        dataset_name = None
                        save_dir = f'results/{aug}/{env_id}/td3bc/nl_{n_layers}/hd_{hd}/lr_{lr}/a_{alpha}/t_{tau}'
                        batch_size = 256
                    else:
                        save_dir = f'results/{aug}/m_{m}/{env_id}/td3bc/nl_{n_layers}/hd_{hd}/lr_{lr}/a_{alpha}/t_{tau}'
                        dataset_name = f'/staging/qu45/GuDA/datasets/{env_id}/{aug}/m_{m}.hdf5'
                        batch_size = 256


                    command = gen_td3_bc(
                        save_dir=save_dir,
                        max_timesteps=int(1e6),
                        eval_freq=int(20e3),
                        # max_timesteps=int(1e4),
                        # eval_freq=int(1e3),
                        dataset_name=dataset_name,
                        env_id=env_id,
                        actor_lr=actor_lr,
                        critic_lr=critic_lr,
                        alpha=alpha,
                        tau=tau,
                        n_layers=n_layers,
                        hidden_dims=hd
                    )
                    mem, disk = MEMDISK[m][env_id]
                    # disk += 7
                    memdisk = f'{mem},{disk},'
                    command = memdisk + command.replace(' ', '*')
                    print(command)
                    i += 1
                    all_commands += command + '\n'