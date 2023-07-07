import os

if __name__ == "__main__":

    # os.chdir('generate')
    os.system(
        f'python ./visualize_dataset.py  '
        f' --dataset-path ../datasets/PushBallToGoal-v0/physical/guided.hdf5'
        f' --save-dir ./figures/PushBallToGoal-v0/physical'
        f' --save-name guided.png'
    )

    for i in range(10):
        os.system(
            f'python ./visualize_dataset.py  '
            f' --dataset-path ../datasets/PushBallToGoal-v0/physical/guided_traj_{i}.hdf5'
            f' --save-dir ./figures/PushBallToGoal-v0/physical'
            f' --save-name guided_traj_{i}.png'
        )