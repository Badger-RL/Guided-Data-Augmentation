import argparse
import numpy as np
from src.generate.utils import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default=None)
    args = parser.parse_args()
    dataset = load_dataset(args.dataset_path)
    rewards = dataset['rewards']

    print(f'Number of rewards: {len(rewards)}')
    print(f'Number of non-zero rewards: {np.sum(rewards > 0)}')
    print(f'Proportion of non-zero rewards: {np.sum(rewards > 0) * 1.0 / len(rewards)}')
