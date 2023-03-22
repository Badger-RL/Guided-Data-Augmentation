import os

if __name__ == "__main__":

    # os.chdir('generate')
    for policy in ['expert', 'random']:
        for observed_dataset_size in [10e3, 50e3, 100e3]:
            observed_dataset_size = int(observed_dataset_size)
            for aug_dataset_size in [100e3, 200e3]:

                aug_ratio = int(aug_dataset_size // observed_dataset_size - 1)
                if aug_ratio <= 0: continue

                os.system(
                    f'python ./generate_augmented_dataset.py '
                    f' --policy {policy} '
                    f' --observed-dataset-path ../datasets/{policy}/{observed_dataset_size}.hdf5 '
                    f' --augmentation-ratio {aug_ratio}'
                    f' --save-dir ../datasets/{policy}/aug_uniform/'
                    f' --save-name {observed_dataset_size//1e3:.0f}k_{aug_dataset_size//1e3:.0f}k.hdf5'
                )