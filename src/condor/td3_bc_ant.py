from condor.common import gen_td3_bc, gen_cql

MEMDISK = {
    'antmaze-umaze-v1': (1.7, 9),
    'antmaze-umaze-diverse-v1': (1.7, 9),
    'antmaze-medium-diverse-v1': (1.7, 9),
    'antmaze-large-diverse-v1': (3, 9),
}

if __name__ == "__main__":
    all_commands = ""

    for env_id in ['antmaze-large-diverse-v1', 'antmaze-medium-diverse-v1', 'antmaze-umaze-diverse-v1', 'antmaze-umaze-v1',]:
        for aug in ['no_aug' ]:
            for actor_lr in [3e-4]:
                for critic_lr in [3e-4]:
                    if actor_lr != critic_lr: continue
                    for alpha in [2.5, 5, 7.5, 10]:
                        for tau in [1e-3, 5e-3]:
                            save_dir = f'results/{aug}/{env_id}/td3_bc/alr_{actor_lr}/clr_{critic_lr}/a_{alpha}/t_{tau}'

                            if aug == 'no_aug':
                                dataset_name = None
                            else:
                                # if 'umaze' in env_id:
                                #     m = 16
                                # else:
                                #     m = 4
                                m = 4
                                dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}/m_{m}.hdf5'

                            command = gen_td3_bc(
                                save_dir=save_dir,
                                max_timesteps=int(1e6),
                                eval_freq=int(10000),
                                dataset_name=dataset_name,
                                env_id=env_id,
                                actor_lr=actor_lr,
                                critic_lr=critic_lr,
                                alpha=alpha,
                                tau=tau
                            )
                            mem, disk = MEMDISK[env_id]
                            memdisk = f'{mem},{disk},'
                            command = memdisk + command.replace(' ', '*')
                            print(command)
                            all_commands += command + '\n'