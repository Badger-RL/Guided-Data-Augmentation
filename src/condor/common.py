def gen_command(
        save_dir,
        dataset_name,
        max_timesteps=int(100e3),
        eval_freq=5000,
):
    command = f'python algorithms/cql.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
              f' --save_dir {save_dir} ' \
              f' --dataset_name {dataset_name}'

    return command