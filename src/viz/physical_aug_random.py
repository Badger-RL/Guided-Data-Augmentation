import os

if __name__ == "__main__":

    # os.chdir('generate')
    os.system(
        f'python ./visualize_dataset.py  '
        f' --dataset-path ../datasets/physical/aug_uniform/10_episodes_200k.hdf5'
        f' --save-dir ./figures/physical/aug_random/'
        f' --save-name 10_episodes_200k.png'
    )
