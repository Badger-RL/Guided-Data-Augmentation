import os

for env_id in ['intersection-v0']:
    for num_samples in [10000]:
        num_samples = int(num_samples)
        for skip_terminated_episodes in [0]:

            save_dir = f'../../datasets/{env_id}/'
            save_name = f'no_aug_{int(num_samples/1e3)}k'
            if skip_terminated_episodes:
                save_name = f'no_aug_{int(num_samples / 1e3)}k_skip.hdf5'
            else:
                save_name = f'no_aug_{int(num_samples / 1e3)}k.hdf5'

            save_name = f'no_aug.hdf5'

            command = f'python simulate.py ' \
                      f' --env_id {env_id} ' \
                      f' --policy_path ../../results/{env_id}/dqn/rl_model_10000_steps.zip ' \
                      f' --save_dir {save_dir} --save_name {save_name}' \
                      f' --num_samples {num_samples} --skip_terminated_episodes {skip_terminated_episodes}'
            os.system(command)