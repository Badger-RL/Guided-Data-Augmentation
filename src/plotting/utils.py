import os

import numpy as np
import seaborn as sns
# from plotting.plot import plot
from matplotlib import pyplot as plt


def get_paths(results_dir, key, file_name='evaluations.npz', **kwargs):
    path_dict = {}
    path_dict[key] = {
        'paths': []
    }
    for dirpath, dirnames, filenames in os.walk(results_dir):
        for fname in filenames:
            if fname == file_name:
                path_dict[key]['paths'].append(f'{dirpath}/{fname}')

    return path_dict


def load_data(paths, field_name='return'):
    t = None
    avgs = []

    for path in paths:

        data = np.load(path, allow_pickle=True)
        avg = data[field_name]
        avgs.append(avg)

        if t is None:
            t = data['timestep']

    return t, np.array(avgs)


def plot(path_dict, field_name='return'):
    for agent, info in path_dict.items():
        paths = info['paths']

        t, avgs = load_data(paths, field_name=field_name)
        assert len(avgs) > 0

        avg_of_avgs = np.average(avgs, axis=0)

        l = min(len(avg_of_avgs), len(t))
        t = t[:l]
        avg_of_avgs = avg_of_avgs[:l]

        # compute 95% confidence interval
        std = np.std(avgs, axis=0)
        N = len(avgs)
        ci = 1.96 * std / np.sqrt(N)
        q05 = avg_of_avgs + ci
        q95 = avg_of_avgs - ci
        print(agent, avg_of_avgs[-1], ci[-1])

        style_kwargs = {}
        style_kwargs['linewidth'] = 3

        if 'no_aug' in agent.lower():
            style_kwargs['linestyle'] = ':'
        elif 'random' in agent.lower():
            style_kwargs['linestyle'] = '--'
        elif 'guided_neg' in agent.lower():
            style_kwargs['linestyle'] = '-.'
        # t = np.arange(len(avg_of_avgs)) * 5000

        plt.plot(t, avg_of_avgs, label=agent, **style_kwargs)
        plt.fill_between(t, q05, q95, alpha=0.2)