import os
from condor.common import gen_command

def expert():
    all_commands = ""
    dataset_dir = 'expert/no_aug'
    for dataset_size in [10, 50, 100]:
        name = f"Exp_{dataset_dir.replace('/', '_')}_{dataset_size}k"
        save_dir = f'results/PushBallToGoal/expert/no_aug/{dataset_size}k'
        dataset_name = f'{dataset_dir}/{dataset_size}k.hdf5'

        command = gen_command(
            save_dir=save_dir,
            dataset_name=dataset_name,
        )

        print(command)
        command = command.replace(' ', '*')
        all_commands += command + '\n'

    return all_commands

def expert_traj():
    all_commands = ""
    dataset_dir = 'expert/trajectories'
    for i in range(5):
        save_dir = f'results/PushBallToGoal/expert/no_aug/traj_{i}'
        dataset_name = f'{dataset_dir}/{i}.hdf5'
        command = gen_command(
            save_dir=save_dir,
            dataset_name=dataset_name,
        )

        print(command)
        command = command.replace(' ', '*')
        all_commands += command + '\n'

    return all_commands

def expert_restricted():

    all_commands = ""
    dataset_dir = 'expert_restricted/no_aug'
    for dataset_size in [10, 50, 100]:
        save_dir = f'results/PushBallToGoal/expert_restricted/no_aug/{dataset_size}k'
        dataset_name = f'{dataset_dir}/{dataset_size}k.hdf5'
        command = gen_command(
            save_dir=save_dir,
            dataset_name=dataset_name,
        )

        print(command)
        command = command.replace(' ', '*')
        all_commands += command + '\n'
    return all_commands

if __name__ == "__main__":

    all_commands = ""
    all_commands += expert()
    all_commands += expert_traj()
    all_commands += expert_restricted()

    save_dir = 'commands'
    os.makedirs(save_dir, exist_ok=True)
    f = open(f'{save_dir}/expert.txt', "w",)

    f.write(all_commands[:-1])
    f.close()


