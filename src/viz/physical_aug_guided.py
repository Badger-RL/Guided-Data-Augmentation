import os

if __name__ == "__main__":

    # os.chdir('generate')
    os.system(
        f'python ./visualize_dataset.py  '
        f' --dataset-path ../datasets/physical/aug_guided/10_episodes_200k.hdf5'
        f' --save-dir ./figures/physical/aug_guided/'
        f' --save-name 10_episodes_200k.png'
    )

    for i in range(10):
        os.system(
            f'python ./visualize_dataset.py  '
            f' --dataset-path ../datasets/physical/aug_guided/{i}_200k.hdf5'
            f' --save-dir ./figures/physical/aug_guided'
            f' --save-name {i}.png'
        )