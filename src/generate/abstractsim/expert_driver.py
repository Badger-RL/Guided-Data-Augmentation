import os


# 40 893.5071630557165
# 60 1195.693313336531
# 72 1353.9820733017361
for success_rate in [40,72]:
    for n in [10]:
        os.system(f'python3 expert.py --num_traj {n} '
                  f' --policy-path ../../../src/policies/PushBallToGoal-v0/policy_{success_rate}.zip'
                  f' --save-dir ../../datasets/PushBallToGoal-v0'
                  f' --save-name no_aug_{success_rate}_{n}k.hdf5')

# for success_rate in [60, 40]:
#     os.system('python3 expert.py --num_samples 500000 '
#               f' --policy-path ../../../src/policies/PushBallToGoal-v0/policy_{success_rate}.zip'
#               f' --save-dir ../../datasets/PushBallToGoal-v0'
#               f' --save-name no_aug_{success_rate}_deterministic.hdf5')