from condor.common import gen_td3_bc
from condor.utils import MEMDISK

if __name__ == "__main__":
    all_commands = ""

    i = 0
    env_id = 'PushBallToGoal-v0'
    for expert_success_rate in [40,60,72]:
        aug = f'no_aug_{expert_success_rate}'
        # for aug in ['guided_transition']:
        for alpha in [1, 2.5, 5]:
            for lr in [3e-4, 3e-5]:
                for nl in [1,2]:
                    dataset_name = f'/staging/ncorrado/datasets/{env_id}/{aug}.hdf5'

                    save_dir = f'results/{env_id}/{aug}/td3bc/nl_{nl}/lr_{lr}/a_{alpha}'
                    hidden_dims = 256
                    tau = 1e-3
                    actor_lr = lr
                    critic_lr = lr

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
                        n_layers=nl,
                        hidden_dims=hidden_dims
                    )
                    mem, disk = MEMDISK[1][env_id]
                    memdisk = f'{mem},{disk},'
                    command = memdisk + command.replace(' ', '*')
                    print(command)
                    i += 1
                    all_commands += command + '\n'
    #
    # for aug in ['random', 'guided', 'no_aug']:
    #     for alpha in [2.5, 5, 7.5, 10]:
    #         for actor_lr in [3e-5]:
    #             for critic_lr in [3e-4]:
    #                 for tau in [1e-3]:
    #                     dataset_name = f'/staging/ncorrado/datasets/{env_id}/physical/{aug}.hdf5'
    #
    #                     save_dir = f'results/{aug}/{env_id}/physical/td3bc/lr_{actor_lr}/lr_{critic_lr}/a_{alpha}/t_{tau}'
    #                     n_layers = 2
    #                     hidden_dims = 256
    #
    #                     command = gen_td3_bc(
    #                         save_dir=save_dir,
    #                         max_timesteps=int(1e6),
    #                         eval_freq=int(10000),
    #                         dataset_name=dataset_name,
    #                         env_id=env_id,
    #                         actor_lr=actor_lr,
    #                         critic_lr=critic_lr,
    #                         alpha=alpha,
    #                         tau=tau,
    #                         n_layers=n_layers,
    #                         hidden_dims=hidden_dims
    #                     )
    #                     mem, disk = MEMDISK[1][env_id]
    #                     memdisk = f'{mem},{disk},'
    #                     command = memdisk + command.replace(' ', '*')
    #                     print(command)
    #                     i += 1
    #                     all_commands += command + '\n'