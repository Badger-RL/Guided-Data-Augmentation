import os

for env_id in ['highway-v0']:
    for num_samples in [10e3, 20e3]:
        num_samples = int(num_samples)
        for skip_terminated_episodes in [0, 1]:

            save_dir = f'../../datasets/{env_id}/'
            save_name = f'no_aug_{int(num_samples/1e3)}k'
            if skip_terminated_episodes:
                save_name = f'no_aug_{int(num_samples / 1e3)}k_skip.hdf5'
            else:
                save_name = f'no_aug_{int(num_samples / 1e3)}k.hdf5'

            command = f'python simulate.py ' \
                      f' --env_id {env_id} ' \
                      f' --policy_path ../../policies/{env_id}/best_model.zip ' \
                      f' --save_dir {save_dir} --save_name {save_name}' \
                      f' --num_samples {num_samples} --skip_terminated_episodes {skip_terminated_episodes}'
            os.system(command)