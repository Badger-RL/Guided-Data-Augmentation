import os

for maze in ['umaze', 'medium', 'large']:
    env_id = f'maze2d-{maze}-v1'
    for aug_func in ['random', 'guided']:
        for aug_size in [10e3, 100e3, 250e3]:
            aug_size = int(aug_size)
            print(env_id)
            # os.system(f'python ../augment_dataset.py --observed-dataset-path ../../datasets/{env_id}/small/no_aug.hdf5  --env-id {env_id} --aug-size {int(1e6)} --aug-func {aug_func} --save-dir ../../datasets/{env_id}/small/{aug_func}')
            os.system(f'python ../augment_dataset.py --observed-dataset-path ../../datasets/{env_id}/small/no_aug.hdf5  --env-id {env_id} --aug-size {aug_size} --aug-func {aug_func} --save-dir ../../datasets/{env_id}')