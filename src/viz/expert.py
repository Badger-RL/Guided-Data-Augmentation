import os

if __name__ == "__main__":

    # os.chdir('generate')
    for size in [1, 5, 10, 50, 100]:
        os.system(
            f'python ./visualize_dataset.py  '
            f' --dataset-path ../datasets/expert/no_aug/{size}k.hdf5'
            f' --save-dir ./figures/sim/no_aug'
            f' --save-name {size}k.png'
        )
