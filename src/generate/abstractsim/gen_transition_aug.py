import os

if __name__ == "__main__":

    for expert_success_rate in [72]:
        for aug in ['guided']:
            os.system(
                f'python ./transition_aug_guided.py --seed 0'
                f' --observed-dataset-path ../../datasets/PushBallToGoal-v0/no_aug_{expert_success_rate}_10k.hdf5'
                f' --augmentation-ratio 9'
                f' --save-dir ../../datasets/PushBallToGoal-v0/'
                f' --save-name {aug}_{expert_success_rate}.hdf5'
                f' --check-valid 0'
            )

    # for expert_success_rate in [72, 40]:
    #     for aug in ['guided']:
    #         os.system(
    #             f'python ./transition_aug_guided.py --seed 0'
    #             f' --observed-dataset-path ../../datasets/PushBallToGoal-v0/no_aug_{expert_success_rate}_deterministic.hdf5'
    #             f' --augmentation-ratio 1'
    #             f' --save-dir ../../datasets/PushBallToGoal-v0/'
    #             f' --save-name {aug}_{expert_success_rate}_deterministic.hdf5'
    #             f' --check-valid 0'
    #         )