def gen_command(
        save_dir,
        dataset_name,
        max_timesteps=int(100e3),
        eval_freq=5000,
        qf_lr=3e-06,
        policy_lr=3e-06,
        soft_target_update_rate=0.05,
        target_update_period=10
):
    command = f'python algorithms/cql.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
              f' --save_dir {save_dir} ' \
              f' --dataset_name {dataset_name}' \
              f' --qf_lr {qf_lr}' \
              f' --policy_lr {policy_lr}' \
              f' --soft_target_update_rate {soft_target_update_rate}' \
              f' --target_update_period {target_update_period}' 
    return command