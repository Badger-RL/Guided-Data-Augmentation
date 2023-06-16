from condor.common import gen_td3_bc, gen_cql

if __name__ == "__main__":
    all_commands = ""

    for env_id in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']:
        for lr in [3e-4]:
            for alpha in [1,2.5,5,7.5, 10, 12.5, 15]:

                save_dir = f'results/no_aug/{env_id}/td3_bc/lr_{lr}/a_{alpha}'
                command = gen_td3_bc(
                    save_dir=save_dir,
                    max_timesteps=int(1e6),
                    # eval_freq=1000,
                    env_id=env_id,
                    actor_lr=lr,
                    critic_lr=lr,
                    alpha=alpha
                )

                # command = command.replace(' ', '*')
                print(command)
                all_commands += command + '\n'
        #
        # for gap in [1, 2.5, 5]:
        #
        #     save_dir = f'results/{env_id}/cql/lr_{lr}/g_{gap}'
        #     command = gen_cql(
        #         save_dir=save_dir,
        #         max_timesteps=int(1e6),
        #         # eval_freq=1000,
        #         env_id=env_id,
        #         # policy_lr=lr,
        #         # qf_lr=lr,
        #         gap=gap
        #     )
        #
        #     command = command.replace(' ', '*')
        #     print(command)
        #     all_commands += command + '\n'