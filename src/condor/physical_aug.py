import os

from src.condor.common import gen_command
def guided_single_traj():
    all_commands = ""
    dataset_dir = 'physical/aug_guided'

    # guided datasets generated from a single expert trajectory
    dataset_size = 200
    for i in range(10):
        save_dir = f'results/PushBallToGoal/physical/aug_guided/{i}'
        dataset_name = f'{dataset_dir}/{i}_{dataset_size}k.hdf5'
        command = gen_command(
            save_dir=save_dir,
            dataset_name=dataset_name,
        )

        print(command)
        command = command.replace(' ', '*')
        all_commands += command + '\n'

    return all_commands

def guided():

    all_commands = ""
    dataset_dir = 'physical/aug_guided'

    # guided datasets generated from 10k and 50k expert datasets
    save_dir = f'results/PushBallToGoal/physical/aug_guided/10_episodes_200k'
    dataset_name = f'{dataset_dir}/10_episodes_200k.hdf5'
    command = gen_command(
        save_dir=save_dir,
        dataset_name=dataset_name,
    )
    print(command)
    command = command.replace(' ', '*')
    all_commands += command + '\n'

    return all_commands

def random():

    all_commands = ""
    dataset_dir = 'physical/aug_uniform'

    save_dir = f'results/PushBallToGoal/physical/aug_uniform/10_episodes_200k'
    dataset_name = f'{dataset_dir}/10_episodes_200k.hdf5'
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
    all_commands += guided()
    all_commands += random()
                
    save_dir = 'commands'
    os.makedirs(save_dir, exist_ok=True)
    f = open(f'{save_dir}/physical_aug.txt', "w",)

    f.write(all_commands[:-1])
    f.close()


