import os
import itertools

def gen_command(
        name,
        dataset_name,
        qf_lr,
        soft_target_update_rate,
        policy_lr,
        target_update_period,
        max_timesteps=int(100e3),
        eval_freq=5000,
):
    command = f'python algorithms/cql.py --name {name}' \
              f' --dataset_name {dataset_name}' \
              f' --max_timesteps {max_timesteps} --eval_freq {eval_freq}' \
              f' --qf_lr {qf_lr} --soft_target_update_rate {soft_target_update_rate}'\
              f' --policy_lr {policy_lr} --target_update_period {target_update_period}'
    return command

if __name__ == "__main__":



    hyperparams = {"qf_lr": [ 3e-6, 3e-5, 3e-4],
                "soft_target_update_rate":[0.005, 0.05],
                 "policy_lr": [ 3e-6, 3e-5, 3e-4],
                 "target_update_period": [1,10] }



    ordered_entries = hyperparams.items()

    order = [key for key, _ in ordered_entries]


    settings = [set(entry) for _, entry in ordered_entries ]
    settings = [entry for entry in itertools.product(*settings)]


    print(settings)
    print(order)


    all_commands = ""


    dataset_dir = "physical"

    for setting in settings:

            name = "ExpPhysicalSweep"
            dataset_name = f'{dataset_dir}/10_episodes.hdf5'
            command = gen_command(
                name=name,
                dataset_name=dataset_name,
                qf_lr= setting[order.index("qf_lr")],
                soft_target_update_rate = setting[order.index("soft_target_update_rate")],
                policy_lr =setting[order.index("policy_lr")] ,
                target_update_period = setting[order.index("target_update_period")],

            )

            print(command)
            command = command.replace(' ', '*')
            all_commands += command + '\n'

    save_dir = 'commands'
    os.makedirs(save_dir, exist_ok=True)
    f = open(f'{save_dir}/physical_sweep.txt', "w",)

    f.write(all_commands[:-1])
    f.close()


