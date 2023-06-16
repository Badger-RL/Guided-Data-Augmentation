from condor.common import gen_td3_bc, gen_cql

if __name__ == "__main__":
    all_commands = ""

    for env_id in ['maze2d-umaze-v1', 'maze2d-medium-v1']:
        for aug in ['no_aug', 'random', 'guided']:
            for policy_lr in [3e-5]:
                for qf_lr in [3e-4]:
                    for gap in [2.5, 5, 10]:
                        save_dir = f'results/{aug}/{env_id}/cql/gap_{gap}'

                        if aug == 'no_aug':
                            dataset_name = None
                        else:
                            dataset_name = f'{env_id}/{aug}/m_1.hdf5'

                        command = gen_cql(
                            save_dir=save_dir,
                            max_timesteps=int(1e6),
                            eval_freq=int(10000),
                            dataset_name=dataset_name,
                            env_id=env_id,
                            policy_lr=policy_lr,
                            qf_lr=qf_lr,
                            gap=gap
                        )

                        command = command.replace(' ', '*')
                        print(command)
                        all_commands += command + '\n'