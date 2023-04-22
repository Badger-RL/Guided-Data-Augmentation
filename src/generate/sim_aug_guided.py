import os

if __name__ == "__main__":

    for size in [10, 50]:
        aug_ratio = int((100-1)/size)
        aug_size = size*(aug_ratio+1)
        print(aug_size, aug_ratio)
        os.system(
            f'python ./aug_guided.py --seed {0} --aug-size {aug_size*1000} --validate 1'
            f' --observed-dataset-path ../datasets/expert/no_aug/{size}k.hdf5'
            f' --save-dir ../datasets/expert/aug_guided'
            f' --save-name {size}k_{aug_size}k.hdf5')

    # guided aug for single trajectory datasets
    for i in range(5):
        aug_size = int(100e3)
        os.system(
            f'python ./aug_guided.py --seed {i} --aug-size {aug_size} --validate 0'
            f' --observed-dataset-path ../datasets/expert/trajectories/{i}.hdf5'
            f' --save-dir ../datasets/expert/aug_guided'
            f' --save-name {i}_{int(aug_size/1000)}k.hdf5')