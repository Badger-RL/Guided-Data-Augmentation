

def gen_cql(
        save_dir,
        env_id,
        dataset_name=None,
        max_timesteps=int(100e3),
        eval_freq=5000,
        qf_lr=3e-4,
        policy_lr=3e-5,
        gap=5,
):
    command = f'python -u algorithms/cql.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
              f' --save_dir {save_dir} ' \
              f' --env {env_id}' \
              f' --qf_lr {qf_lr}' \
              f' --policy_lr {policy_lr}' \
              f' --cql_target_action_gap {gap}'

    if dataset_name:
        command += f' --dataset_name {dataset_name}'

    return command

def gen_td3_bc(
        save_dir,
        env_id,
        dataset_name,
        max_timesteps=int(100e3),
        eval_freq=5000,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha=2.5,
        tau=5e-3,
        n_layers=1,
        hidden_dims=2,
        batch_size=256,
):
    command = f'python -u algorithms/td3_bc.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
              f' --save_dir {save_dir} ' \
              f' --env {env_id}' \
              f' --actor_lr {actor_lr}' \
              f' --critic_lr {critic_lr}' \
              f' --batch_size {batch_size}' \
              f' --alpha {alpha}' \
              f' --tau {tau}' \
              f' --n_layers {n_layers}' \
              f' --hidden_dims {hidden_dims}'

    if dataset_name:
        command += f' --dataset_name {dataset_name}'

    return command


def gen_awac(
        save_dir,
        env_id,
        dataset_name,
        max_timesteps=int(100e3),
        eval_freq=10000,
        lr=3e-4,
        lmbda=1,
):
    command = f'python -u algorithms/awac.py --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
              f' --save_dir {save_dir} ' \
              f' --env {env_id}' \
              f' --learning_rate {lr}' \
              f' --awac_lambda {lmbda}' \

    if dataset_name:
        command += f' --dataset_name {dataset_name}'

    return command