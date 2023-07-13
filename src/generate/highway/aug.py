import os

# for maze in ['large']:
#     env_id = f'maze2d-{maze}-v1'
#     for aug_func in ['random', 'guided', 'guided_mix']:
#     # for aug_func in ['guided']:
#         os.system(f'python ../augment_dataset.py --env-id {env_id} --aug-func {aug_func} --save-dir ../../datasets/{env_id}/{aug_func}') #--observed-dataset-path ../../datasets/{env_id}/no_aug-sparse.hdf5')
#
#     import os

for env_id in ['highway-v0']:
    for aug_func in ['random', 'guided']:
        for m in [1]:
            os.system(
                f'python ../augment_dataset.py --env-id {env_id} --aug-func {aug_func} -m {m} --load- --save-dir ../../datasets/{env_id}/{aug_func}')