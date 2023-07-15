import os

if __name__ == "__main__":

    for i in range(10):
        os.system(
            f'python ./visualize_dataset.py  '
            f' --dataset-path ../datasets/PushBallToGoal-v0/physical/trajectories/traj_{i}.hdf5'
            f' --save-dir ./figures/PushBallToGoal-v0/physical/'
            f' --save-name traj_{i}.png'
        )
