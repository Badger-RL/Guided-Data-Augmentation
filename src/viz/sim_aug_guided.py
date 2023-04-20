import os

if __name__ == "__main__":

    # os.chdir('generate')
    for size in [10, 50]:
        os.system(
            f'python ./visualize_dataset.py  '
            f' --dataset-path ../datasets/expert/aug_guided/{size}k_100k.hdf5'
            f' --save-dir ./figures/sim/aug_guided'
            f' --save-name {size}k_100k.png'
        )

    # guided datasets generated form a single expert trajectory
    for i in range(5):
        for dataset_size in [100]:
            os.system(
                f'python ./visualize_dataset.py  '
                f' --dataset-path ../datasets/expert/aug_guided/{i}_{dataset_size}k.hdf5'
                f' --save-dir ./figures/sim/aug_guided'
                f' --save-name {i}_{dataset_size}k.png'
            )
