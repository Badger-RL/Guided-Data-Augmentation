import os

if __name__ == "__main__":

    # os.chdir('generate')
    for i in range(5):
        for dataset_size in [100, 200]:
            os.system(
                f'python ./visualize_dataset.py  '
                f' --dataset-path ../datasets/expert/aug_guided/{i}_{dataset_size}k.hdf5'
                f' --save-dir ./figures/aug_guided/'
                f' --save-name {i}_{dataset_size}k.png'
            )
