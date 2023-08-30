import os

if __name__ == "__main__":

    # os.chdir('generate')
    os.system(
        f'python ./visualize_dataset.py  '
        f' --dataset-path ../datasets/PushBallToGoal-v0/physical/random.hdf5'
        f' --save-dir figures/PushBallToGoal-v0/physical'
        f' --save-name random.png'
    )
