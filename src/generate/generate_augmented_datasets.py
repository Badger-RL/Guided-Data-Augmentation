import os

if __name__ == "__main__":

    # os.chdir('generate')
    for policy in ['expert', 'random']:
        for observed_dataset_size in [1e3, 5e3, 10e3, 50e3, 100e3]:
            observed_dataset_size = int(observed_dataset_size)
            for aug_ratio in [1,4]:
                os.system(
                    f'python ./generate_augmented_dataset.py '
                    f' --policy {policy} '
                    f' --observed-dataset-size {observed_dataset_size} '
                    f' --augmentation-ratio {aug_ratio}')