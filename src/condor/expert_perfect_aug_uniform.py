import os


def expert(
        name,
        dataset_name,
        max_timesteps=int(100e3),
        eval_freq=5000,
):
    command = f'python algorithms/cql.py --name {name}' \
              f' --dataset_name {dataset_name}' \
              f' --max_timesteps {max_timesteps} --eval_freq {eval_freq}'
    return command

if __name__ == "__main__":

    all_commands = ""
    for dataset_dir in ['expert_perfect/aug_uniform']:
        for observed_dataset_size in [10, 50, 100]:
            for aug_dataset_size in [100, 200]:
                if observed_dataset_size >= aug_dataset_size: continue

                name = f"ExpPerfect_{dataset_dir.replace('/', '_')}_{observed_dataset_size}k_{aug_dataset_size}k"
                dataset_name = f'{dataset_dir}/{observed_dataset_size}k_{aug_dataset_size}k.hdf5'
                command = expert(
                    name=name,
                    dataset_name=dataset_name,
                )

                print(command)
                command = command.replace(' ', '*')
                all_commands += command + '\n'

    save_dir = 'commands'
    os.makedirs(save_dir, exist_ok=True)
    f = open(f'{save_dir}/expert_perfect_aug_uniform.txt', "w",)

    f.write(all_commands[:-1])
    f.close()


