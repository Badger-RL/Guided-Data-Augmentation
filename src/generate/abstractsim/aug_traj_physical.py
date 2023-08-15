import os

if __name__ == "__main__":


    for env_id in ['PushBallToGoal-v2']:
        for aug_size in [300e3]:
            for aug in ['guided_traj', 'random_traj']:
                os.system(
                    f'python ./aug_traj.py --seed {0} --aug-size {int(aug_size)} --validate 0'
                    f' --observed-dataset-path ../../datasets/{env_id}/physical/no_aug.hdf5'
                    f' --aug-func {aug}'
                    f' --save-dir ../../datasets/{env_id}/physical'
                    f' --save-name {aug}.hdf5')

    # for aug_size in [300e3]:
    #     for aug in ['guided_traj']:
    #         os.system(
    #             f'python ./aug_traj.py --seed {0} --aug-size {int(aug_size)} --validate 0'
    #             f' --observed-dataset-path ../../generate/abstractsim/export_physical_trajectories/curated_kicks.hdf5'
    #             f' --aug-func {aug}'
    #             f' --save-dir ../../datasets/PushBallToGoal-v1/physical'
    #             f' --save-name {aug}.hdf5')