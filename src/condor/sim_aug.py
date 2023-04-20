import os

from condor.common import gen_command

def guided_single_traj():

    all_commands = ""
    dataset_dir = 'expert/aug_guided'

    # guided datasets generated from a single expert trajectory
    for dataset_size in [100]:
        for i in range(5):

            save_dir = f'results/PushBallToGoal/sim/aug_guided/{i}'
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
    dataset_dir = 'expert/aug_guided'

    # guided datasets generated from 10k and 50k expert datasets
    for expert_size in [10, 50]:
        save_dir = f'results/PushBallToGoal/sim/aug_guided/{expert_size}k_100k'
        dataset_name = f'{dataset_dir}/{expert_size}_100k.hdf5'
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
    dataset_dir = 'expert/aug_uniform'
    for observed_dataset_size in [10, 50]:
        for aug_dataset_size in [100]:
            if observed_dataset_size >= aug_dataset_size: continue

            save_dir = f'results/PushBallToGoal/sim/aug_uniform'
            dataset_name = f'{dataset_dir}/{observed_dataset_size}k_{aug_dataset_size}k.hdf5'
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
    f = open(f'{save_dir}/sim_aug.txt', "w",)

    f.write(all_commands[:-1])
    f.close()


