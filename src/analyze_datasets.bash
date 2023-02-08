#!/bin/bash

source env/bin/activate

python3 ./algorithms/cql.py --dataset_name dataset_expert_1000.hdf5
python3 ./algorithms/cql.py --dataset_name dataset_expert_5000.hdf5
python3 ./algorithms/cql.py --dataset_name dataset_expert_10000.hdf5
python3 ./algorithms/cql.py --dataset_name dataset_expert_50000.hdf5
python3 ./algorithms/cql.py --dataset_name dataset_expert_100000.hdf5
python3 ./algorithms/cql.py --dataset_name dataset_random_1000.hdf5
python3 ./algorithms/cql.py --dataset_name dataset_random_5000.hdf5
python3 ./algorithms/cql.py --dataset_name dataset_random_10000.hdf5
python3 ./algorithms/cql.py --dataset_name dataset_random_50000.hdf5
python3 ./algorithms/cql.py --dataset_name dataset_random_100000.hdf5