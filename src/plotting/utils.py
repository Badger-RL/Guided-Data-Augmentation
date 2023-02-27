import os

import numpy as np
import seaborn as sns
# from plotting.plot import plot
from matplotlib import pyplot as plt


def get_paths(results_dir, key, **kwargs):

    path_dict = {}
    path_dict[key] = {
        'paths': []
    }
    for dirpath, dirnames, filenames in os.walk(results_dir):
        for fname in filenames:
            path_dict[key]['paths'].append(f'{dirpath}/{fname}')

    return path_dict



def load_data(paths):
    t = None
    avg_returns = []

    for path in paths:
        data = np.load(path, allow_pickle=True).item()
        # t = np.array(data['t'])
        r = data['r']
        print(t)

        if r is not None:
            avg_returns.append(r)
        else:
            print(f'Could not load data at {path}. Skipping.')

    return t, np.array(avg_returns)

def plot(path_dict):

    for agent, info in path_dict.items():
        paths = info['paths']

        t, avgs = load_data(paths)
        assert len(avgs) > 0

        avg_of_avgs = np.average(avgs, axis=0)

        # compute 95% confidence interval
        std = np.std(avgs, axis=0)
        N = len(avgs)
        ci = 1.96 * std / np.sqrt(N)
        q05 = avg_of_avgs + ci
        q95 = avg_of_avgs - ci

        style_kwargs = {}
        t = np.arange(len(avg_of_avgs)) * 500
        plt.plot(t, avg_of_avgs, label=agent, **style_kwargs)
        plt.fill_between(t, q05, q95, alpha=0.2)


