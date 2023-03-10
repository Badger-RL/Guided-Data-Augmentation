import os

if __name__ == "__main__":

    # os.chdir('generate')
    for i in range(5):
        os.system(
            f'python ./visualize_dataset.py  '
            f' --dataset-path ../datasets/expert/trajectories/guided/{i}.hdf5'
            f' --save-dir ./figures/guided/'
            f' --save-name {i}.png'
        )
