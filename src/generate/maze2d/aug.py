import os

for m in [1,2,4]:
    for maze in ['umaze', 'medium', 'large']:
        env_id = f'maze2d-{maze}-v1'
        for aug_func in ['random', 'guided']:
            os.system(f'python ../augment_dataset.py --env-id {env_id} --aug-func {aug_func} --save-dir ../../datasets/{env_id}/{aug_func}') #--observed-dataset-path ../../datasets/{env_id}/no_aug-sparse.hdf5')