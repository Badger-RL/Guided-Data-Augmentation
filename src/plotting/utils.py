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



def load_data(paths, success_rate=False):
    t = None
    avgs = []

    for path in paths:

        data = dict(np.load(path))
        if success_rate:
            avg = data['success_rate']
        else:
            avg = data['r']
        avgs.append(avg)
        
        if t is None:
            t = data['t']

    return t, np.array(avgs)

def plot(path_dict, success_rate=False):

    for agent, info in path_dict.items():
        paths = info['paths']

        t, avgs = load_data(paths, success_rate=success_rate)
        assert len(avgs) > 0

        avg_of_avgs = np.average(avgs, axis=0)

        # compute 95% confidence interval
        std = np.std(avgs, axis=0)
        N = len(avgs)
        ci = 1.96 * std / np.sqrt(N)
        q05 = avg_of_avgs + ci
        q95 = avg_of_avgs - ci

        style_kwargs = {}
        #t = np.arange(len(avg_of_avgs)) * 5000
        plt.plot(t, avg_of_avgs, label=agent, **style_kwargs)
        plt.fill_between(t, q05, q95, alpha=0.2)