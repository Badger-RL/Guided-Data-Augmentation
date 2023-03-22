import os

if __name__ == "__main__":

    for i in range(5):
        for aug_ratio in [99, 199]:
            aug_dataset_size = 500 + aug_ratio*500
            os.system(
                f'python ./generate_guided_trajectory_dataset.py --seed {i} --aug-ratio {aug_ratio}'
                f' --observed-dataset-path ../datasets/expert/trajectories/{i}.hdf5'
                f' --save-dir ../datasets/expert/trajectories/guided'
                f' --save-name {i}_{aug_dataset_size}.hdf5')