import os

if __name__ == "__main__":

    # os.chdir('generate')

    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # disable pygame welcome message

    dataset_dir = '../datasets/'
    all_dataset_paths = []
    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        for fname in filenames:
            if fname.endswith('.hdf5'):
                all_dataset_paths.append(f'{dirpath}/{fname}')

    for dataset_path in all_dataset_paths:
        print(f'Validating {dataset_path}...')
        os.system(
            f'python ./validate_dataset.py --dataset-path {dataset_path}'
        )
