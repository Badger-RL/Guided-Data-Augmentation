import os

if __name__ == "__main__":

    for expert_success_rate in [50]:
        for aug in ['random', 'guided']:
            os.system(
                f'python ./aug_guided.py --seed {0} --guided {int(aug == "guided")} --aug-size {int(200e3)} --validate 0'
                f' --observed-dataset-path ../../datasets/PushBallToGoal-v0/no_aug_{expert_success_rate}.hdf5'
                f' --save-dir ../../datasets/PushBallToGoal-v0'
                f' --save-name {aug}.hdf5')

        # guided aug for single trajectory datasets
        # for i in range(5):
        #     os.system(
        #         f'python ./aug_guided.py --seed {i} --aug-size {int(200e3)} --validate 0'
        #         f' --observed-dataset-path ../../datasets/PushBallToGoal-v0/trajectories/{i}.hdf5'
        #         f' --save-dir ../../datasets/PushBallToGoal-v0'
        #         f' --save-name guided_traj_{i}.hdf5')