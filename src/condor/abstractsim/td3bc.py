from condor.common import gen_td3_bc
from condor.utils import MEMDISK

if __name__ == "__main__":
    all_commands = ""

    i = 0
    for env_id in ['PushBallToGoal-v0']:
        for aug in ['mocoda']:
            # for expert in [50, 85]:
            #     aug = f'{aug}_{expert}'
                aug = f'{aug}'
                for alpha in [2.5, 5, 7.5, 10]:
                    for lr in [3e-4, 3e-5]:
                        for nl in [2]:
                            dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}.hdf5'

                            save_dir = f'results/{env_id}/{aug}/td3bc/nl_{nl}/lr_{lr}/a_{alpha}'
                            hidden_dims = 256
                            tau = 1e-3
                            actor_lr = lr
                            critic_lr = lr

                            max_timesteps = int(1e6) if nl == 2 else int(1e6)

                            command = gen_td3_bc(
                                save_dir=save_dir,
                                max_timesteps=max_timesteps,
                                eval_freq=int(20e3),
                                dataset_name=dataset_name,
                                env_id=env_id,
                                actor_lr=actor_lr,
                                critic_lr=critic_lr,
                                alpha=alpha,
                                tau=tau,
                                n_layers=nl,
                                hidden_dims=hidden_dims,
                            )
                            mem, disk = MEMDISK[1][env_id]
                            memdisk = f'{mem},{disk},'
                            command = memdisk + command.replace(' ', '*')
                            print(command)
                            i += 1
                            all_commands += command + '\n'
