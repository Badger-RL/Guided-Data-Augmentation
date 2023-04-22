import os

if __name__ == "__main__":

    # os.chdir('generate')
    for size in[10, 50]:
        os.system(
            f'python ./visualize_dataset.py  '
            f' --dataset-path ../datasets/expert/aug_uniform/{size}k_100k.hdf5'
            f' --save-dir ./figures/sim/aug_random'
            f' --save-name {size}k_100k.png'
        )
