import os

for m in [1]:
    for maze in ['umaze', 'medium', 'large']:
        env_id = f'maze2d-{maze}-v1'
        for aug_func in ['no_aug', 'random', 'guided', 'mixed']:
            print(env_id, m)
            os.system(f'python ../augment_dataset.py -size 10000 -m {m} --env-id {env_id} --aug-func {aug_func} --save-dir ../../datasets/{env_id}/{aug_func}')