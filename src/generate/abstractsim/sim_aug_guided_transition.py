import os

if __name__ == "__main__":

    # os.chdir('generate')
    for expert_success_rate in [50, 73, 85, 100]:
        os.system(
            f'python ./aug_guided_transition.py --seed 0'
            f' --observed-dataset-path ../../datasets/PushBallToGoal-v0/no_aug_{expert_success_rate}.hdf5'
            f' --augmentation-ratio 1'
            f' --save-dir ../../datasets/PushBallToGoal-v0/'
            f' --save-name guided_transition_{expert_success_rate}.hdf5'
            f' --check-valid 0'
        )