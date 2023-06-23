import os

if __name__ == "__main__":

    # os.chdir('generate')
    for i in range(10):
        os.system(
            f'python ./generate_single_trajectory_dataset.py --seed {i+42}'
            f' --save-dir ../datasets/expert/trajectories'
            f' --save-name {i}.hdf5')