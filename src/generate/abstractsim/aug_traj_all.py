import os

if __name__ == "__main__":

    for aug_size in [200e3]:
        for aug in ['guided_traj', 'random_traj', ]:
            os.system(
                f'python ./aug_traj.py --seed {0} --aug-size {int(aug_size)} --validate 0'
                f' --observed-dataset-path ../../datasets/PushBallToGoal-v1/no_aug.hdf5'
                f' --aug-func {aug}'
                f' --save-dir ../../datasets/PushBallToGoal-v1'
                f' --save-name {aug}.hdf5')