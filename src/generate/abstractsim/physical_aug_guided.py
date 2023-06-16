import os

if __name__ == "__main__":

    aug_size = int(200e3)
    os.system(
        f'python ./aug_guided.py --seed {0} --aug-size {aug_size} --validate 0 --check-goal-post 1'
        f' --observed-dataset-path ../datasets/physical/10_episodes.hdf5'
        f' --save-dir ../datasets/physical/aug_guided'
        f' --save-name 10_episodes_{int(aug_size/1e3)}k.hdf5')

    for i in range(10):
        for aug_ratio in [99]:
            aug_dataset_size = 2000 + aug_ratio*2000
            os.system(
                f'python ./aug_guided.py --seed {0} --aug-size {aug_size} --validate 0 --check-goal-post 1'
                f' --observed-dataset-path ../datasets/physical/trajectories/{i}.hdf5'
                f' --save-dir ../datasets/physical/aug_guided'
                f' --save-name {i}_{int(aug_dataset_size/1e3)}k.hdf5')