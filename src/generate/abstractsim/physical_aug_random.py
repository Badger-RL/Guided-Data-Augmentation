import os

if __name__ == "__main__":

    # os.chdir('generate')

    aug_ratio = 9

    os.system(
        f'python ./aug_random.py '
        f' --observed-dataset-path ../../datasets/PushBallToGoal-v0/physical/no_aug.hdf5 '
        f' --augmentation-ratio {aug_ratio} --seed 0'
        f' --save-dir ../../datasets/PushBallToGoal-v0/physical/'
        f' --save-name random.hdf5'
        f' --check-valid 0'
    )