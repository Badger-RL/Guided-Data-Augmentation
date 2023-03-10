import os

if __name__ == "__main__":

    # os.chdir('generate')
    for i in range(5):
        os.system(
            f'python ./generate_guided_trajectory_dataset.py --seed {i} --aug-ratio 100'
            f' --observed-dataset-path ../datasets/expert/trajectories/{i}.hdf5'
            f' --save-dir ../datasets/expert/trajectories/guided'
            f' --save-name {i}.hdf5')