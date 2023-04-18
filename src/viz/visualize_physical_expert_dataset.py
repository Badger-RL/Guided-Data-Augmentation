import os

if __name__ == "__main__":

    os.system(
        f'python ./visualize_dataset.py  '
        f' --dataset-path ../datasets/physical/10_episodes.hdf5'
        f' --save-dir ./figures/expert_physical/'
        f' --save-name expert.png'
    )
