import h5py
import numpy as np

def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'next_observations': [],
            'truncateds': [],
            }

def append_data(data, s, a, r, ns, done):
    data['observations'].append(s)
    data['next_observations'].append(ns)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)

def append_data2(data, s, a, r, ns, done, truncated):
    data['observations'].append(s)
    data['next_observations'].append(ns)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)
    data['truncateds'].append(truncated)

def extend_data(data, s, a, r, ns, done):
    data['observations'].extend(s)
    data['next_observations'].extend(ns)
    data['actions'].extend(a)
    data['rewards'].extend(r)
    data['terminals'].extend(done)

def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts', 'truncateds']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def load_dataset(dataset_path):
    data_hdf5 = h5py.File(f"{dataset_path}", "r")

    dataset = {}
    for key in data_hdf5.keys():
        dataset[key] = np.array(data_hdf5[key])

    return dataset

def unpack_dataset(dataset):
    obs = dataset['observations']
    action = dataset['actions']
    reward = dataset['rewards']
    next_obs = dataset['next_observations']
    done = dataset['terminals']