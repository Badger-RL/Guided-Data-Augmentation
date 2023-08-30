import os

if __name__ == "__main__":

    # os.chdir('generate')
    os.system(
        f'python ./visualize_dataset.py  '
        f' --dataset-path ../datasets/PushBallToGoal-v0/random_100.hdf5'
        f' --save-dir ./figures/PushBallToGoal-v0/random'
        f' --save-name random_100.png'
    )
