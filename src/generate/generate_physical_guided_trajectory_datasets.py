import os

if __name__ == "__main__":

    for i in range(10):
        for aug_ratio in [99]:
            aug_dataset_size = 2000 + aug_ratio*2000
            os.system(
                f'python ./generate_guided_trajectory_dataset.py --seed {i} --aug-ratio {aug_ratio}'
                f' --observed-dataset-path ../datasets/physical/trajectories/{i}.hdf5'
                f' --save-dir ../datasets/physical/aug_guided'
                f' --save-name {i}_{int(aug_dataset_size/1e3)}k.hdf5')