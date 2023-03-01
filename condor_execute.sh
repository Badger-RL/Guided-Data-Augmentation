#!/bin/bash


echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"

: '
while sleep 60; do
    ps aux --sort=-%mem | head 
    echo "________________________________" 
done & 
'


mkdir ./workspace

# copy in bundle

cp ./bundle.zip ./workspace
cd workspace
unzip -qq ./bundle.zip

ls

cd GuidedDataAugmentationForRobotics

python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements/requirements.txt
#python -c "import mujoco_py"

python3 -m pip install -e . 
python3 -m pip install -e src/custom-envs

cd src


export D4RL_DATASET_DIR=$(pwd)/.d4rl
export WANDB_CONFIG_DIR=$(pwd)/.config/wandb

python3 ./condor_control.py $1 # we go to our python control script, and tell it our index number

#python3 ./algorithms/cql.py --dataset_name dataset_expert_1000.hdf5

cd ../../..

rm -rf ./workspace

