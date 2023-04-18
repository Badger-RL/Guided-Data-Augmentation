import os

if __name__ == "__main__":

    # os.chdir('generate')
    for i in range(10):
        os.system(
            f'python ./visualize_dataset.py  '
            f' --dataset-path ../datasets/expert/trajectories/{i}.hdf5'
            f' --save-dir ./figures/expert/'
            f' --save-name {i}.png'
        )
