import os

if __name__ == "__main__":

    os.system(
        f'python ./visualize_dataset.py  '
        f' --dataset-path ../datasets/PushBallToGoal-v0/physical/trajectories.hdf5'
        f' --save-dir figures/PushBallToGoal-v0/physical'
        f' --save-name no_aug.png'
    )

    for i in range(10):
        os.system(
            f'python ./visualize_dataset.py  '
            f' --dataset-path ../datasets/PushBallToGoal-v0/physical/trajectories.hdf5'
            f' --save-dir figures/PushBallToGoal-v0/physical'
            f' --save-name no_aug.png'
        )