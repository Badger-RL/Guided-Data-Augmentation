import os


# 40 893.5071630557165
# 60 1195.693313336531
# 72 1353.9820733017361
for success_rate in [40, ]:
    for n in [5]:
        os.system(f'python3 expert.py --num_traj {n} --seed 42' # v0 seed = 42
                  f' --policy-path ../../../src/policies/PushBallToGoal-v1/policy_{success_rate}.zip'
                  # f' --policy-path ../../../src/results/PushBallToGoal-v0/rl_model_1600000_steps.zip'
                  f' --save-dir ../../datasets/PushBallToGoal-v1'
                  f' --save-name no_aug.hdf5 '
                  # f' --render 1'
                  )

# for success_rate in [60, 40]:
#     os.system('python3 expert.py --num_samples 500000 '
#               f' --policy-path ../../../src/policies/PushBallToGoal-v0/policy_{success_rate}.zip'
#               f' --save-dir ../../datasets/PushBallToGoal-v0'
#               f' --save-name no_aug_{success_rate}_deterministic.hdf5')