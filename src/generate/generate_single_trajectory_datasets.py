import os

if __name__ == "__main__":

    # os.chdir('generate')
    for i in range(5):
        os.system(
            f'python ./generate_single_trajectory_dataset.py --seed {i+412}'
            f' --save-dir ../datasets/expert/trajectories'
            f' --save-name {i}.hdf5')