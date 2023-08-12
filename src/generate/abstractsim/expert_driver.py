import os

for expert_success_rate in [50]:
    os.system('python3 expert.py --num_samples 1500 '
              # ' --render 1'
              f' --policy-path ../../policies/PushBallToGoal-v0/policy_{expert_success_rate}'
              f' --norm-path ../../policies/PushBallToGoal-v0/vector_normalize_{expert_success_rate}'
              f' --save-dir ../../datasets/PushBallToGoal-v0'
              f' --save-name no_aug_{expert_success_rate}.hdf5')