import os

if __name__ == "__main__":

    aug_size = int(200e3)
    os.system(
        f'python ./aug_guided.py --seed {0} --aug-size {aug_size} --validate 0 --check-goal-post 1'
        f' --observed-dataset-path ../../datasets/PushBallToGoal-v0/physical/no_aug.hdf5'
        f' --save-dir ../../datasets/PushBallToGoal-v0/physical'
        f' --save-name guided.hdf5')

    for i in range(10):
        os.system(
            f'python ./aug_guided.py --seed {0} --aug-size {aug_size} --validate 0 --check-goal-post 1'
            f' --observed-dataset-path ../../datasets/PushBallToGoal-v0/physical/trajectories/traj_{i}.hdf5'
            f' --save-dir ../../datasets/PushBallToGoal-v0/physical'
            f' --save-name guided_traj_{i}.hdf5')