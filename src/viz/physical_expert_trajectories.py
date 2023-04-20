import os

if __name__ == "__main__":

    for i in range(10):
        os.system(
            f'python ./visualize_dataset.py  '
            f' --dataset-path ../datasets/physical/trajectories/{i}.hdf5'
            f' --save-dir ./figures/expert_physical/'
            f' --save-name {i}.png'
        )
