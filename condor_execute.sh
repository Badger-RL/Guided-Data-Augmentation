#!/bin/bash

mkdir workspace
cd workspace
cp /staging/ncorrado/bundle.zip .
unzip -qq ./bundle.zip

python3 -m venv env
source env/bin/activate

cd GuidedDataAugmentationForRobotics
pip install -r requirements/requirements.txt
export D4RL_DATASET_DIR=$(pwd)/.d4rl
export WANDB_CONFIG_DIR=$(pwd)/.config/wandb

python3 -m pip install -e .
python3 -m pip install -e src/custom-envs

wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
mkdir .mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz -C .mujoco
export MUJOCO_PY_MUJOCO_PATH="$PWD/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/.mujoco/mujoco210/bin"

cd src
pid=$1 # command index
step=$2 # index within different runs of the same command
command=`tr '*' ' ' <<< $3` # replace * with space in command
echo $command

$($command --seed $step --run_id $step)

tar -czvf results_${pid}.tar.gz results/*
mv results_${pid}.tar.gz ../../..

cd ../../..
rm -rf ./workspace

