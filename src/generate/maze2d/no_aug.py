import os

for maze in ['umaze', 'medium']:
    env_id = f'maze2d-{maze}-v1'
    # os.system(f'python ../../D4RL/scripts/generation/generate_maze2d_datasets.py --save_dir ../../datasets/{env_id} --save_name no_aug.hdf5 --env_name {env_id} --num_samples {int(1e6)}')

    for relabel_type in ['sparse']:
        os.system(f'python ../../D4RL/scripts/generation/relabel_maze2d_rewards.py --maze {maze} --filename ../../datasets/{env_id}/no_aug.hdf5 --relabel_type {relabel_type}')

