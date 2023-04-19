import os

if __name__ == "__main__":

    # os.chdir('generate')
    for policy in ['expert', 'random']:
        for observed_dataset_size in [10, 50]:
            observed_dataset_size = int(observed_dataset_size)
            for aug_dataset_size in [100]:

                aug_ratio = int(aug_dataset_size // observed_dataset_size - 1)
                if aug_ratio <= 0: continue

                os.system(
                    f'python ./aug_random.py '
                    f' --policy {policy} --seed 0'
                    f' --observed-dataset-path ../datasets/{policy}/no_aug/{observed_dataset_size}k.hdf5 '
                    f' --augmentation-ratio {aug_ratio}'
                    f' --save-dir ../datasets/{policy}/aug_uniform/'
                    f' --save-name {observed_dataset_size}k_{aug_dataset_size}k.hdf5'
                    f' --check-valid 1'
                )