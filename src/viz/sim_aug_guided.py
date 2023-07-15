import os

if __name__ == "__main__":

    # os.chdir('generate')
    os.system(
        f'python ./visualize_dataset.py  '
        f' --dataset-path ../datasets/PushBallToGoal-v0/guided_100.hdf5'
        f' --save-dir ./figures/PushBallToGoal-v0/guided'
        f' --save-name guided_100.png'
    )

    # guided datasets generated form a single expert trajectory
    for i in range(5):
        os.system(
            f'python ./visualize_dataset.py  '
            f' --dataset-path ../datasets/PushBallToGoal-v0/guided_traj_{i}.hdf5'
            f' --save-dir ./figures/PushBallToGoal-v0/guided'
            f' --save-name traj_{i}.png'
        )
