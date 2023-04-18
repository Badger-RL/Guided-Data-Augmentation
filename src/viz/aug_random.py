import os

if __name__ == "__main__":

    # os.chdir('generate')
    os.system(
        f'python ./visualize_dataset.py  '
        f' --dataset-path ../datasets/expert/aug_uniform/50k_100k.hdf5'
        f' --save-dir ./figures/expert/'
        f' --save-name aug_uniform.png'
    )
