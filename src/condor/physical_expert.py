import os

from src.condor.common import gen_command
def physical_expert():
    all_commands = ""
    dataset_dir = 'physical'

    # guided datasets generated from a single expert trajectory
    save_dir = f'results/PushBallToGoal/physical/expert/'
    dataset_name = f'{dataset_dir}/10_episodes.hdf5'
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
    all_commands += physical_expert()
                
    save_dir = 'commands'
    os.makedirs(save_dir, exist_ok=True)
    f = open(f'{save_dir}/physical_expert.txt', "w",)

    f.write(all_commands[:-1])
    f.close()


