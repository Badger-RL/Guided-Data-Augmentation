import os

if __name__ == "__main__":

    # os.chdir('generate')

    aug_ratio = 9

    os.system(
        f'python ./generate_augmented_dataset.py '
        f' --observed-dataset-path ../datasets/physical/10_episodes.hdf5 '
        f' --augmentation-ratio {aug_ratio}'
        f' --save-dir ../datasets/physical/aug_uniform/'
        f' --save-name 10_episodes_200k.hdf5'
        f' --check-valid 0'
    )