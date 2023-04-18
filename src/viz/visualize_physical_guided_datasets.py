import os

if __name__ == "__main__":

    # os.chdir('generate')
    for i in range(10):
        os.system(
            f'python ./visualize_dataset.py  '
            f' --dataset-path ../datasets/physical/aug_guided/{i}_200k.hdf5'
            f' --save-dir ./figures/expert_physical/aug_guided'
            f' --save-name {i}.png'
        )
