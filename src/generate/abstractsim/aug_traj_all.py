import os

if __name__ == "__main__":

    for expert_success_rate in [40]:
        for observed_size in [5]:
            for aug_size in [200e3]:
                for aug in ['random_traj', 'guided_traj']:
                    os.system(
                        f'python ./aug_traj.py --seed {0} --aug-size {int(aug_size)} --validate 0'
                        f' --observed-dataset-path ../../datasets/PushBallToGoal-v0/no_aug_{expert_success_rate}_{observed_size}k.hdf5'
                        f' --aug-func {aug}'
                        f' --save-dir ../../datasets/PushBallToGoal-v0'
                        f' --save-name {aug}_{expert_success_rate}_{observed_size}k.hdf5')