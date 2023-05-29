#!/bin/bash


#tar -xzf python310.tar.gz
#export PATH=$PWD/python/bin:$PATH

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

#pip install d4rl
#pip install gym==0.23.1
#pip install torch
#pip install wandb
#pip install pyrallis


wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
mkdir .mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz -C .mujoco
export MUJOCO_PY_MUJOCO_PATH="$PWD/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/.mujoco/mujoco210/bin"

######################################
#python3 -m pip install torch
#python3 -m pip install wandb
#git clone https://github.com/Farama-Foundation/d4rl.git
#cd d4rl
#python3 -m pip install -e .
#
## Fetch mujoco_py dependencies
#mkdir -p rpm
#yumdownloader --assumeyes --destdir rpm --resolve \
#  libglvnd-glx.x86_64 \
#  mesa-libGL.x86_64 \
#  mesa-libOSMesa-devel.x86_64 \
#  glew-devel-2.0.0-6.el8.x86_64 \
#  patchelf.x86_64
#
## Install mujoco_py dependencies
#cd rpm
#for rpm in `ls *.rpm`; do rpm2cpio $rpm | cpio -id ; done
#ln -sf $PWD/rpm/usr/lib64/libGL.so.1.7.0 $PWD/rpm/usr/lib64/libGL.so
#export PATH="$PATH:$PWD/rpm/usr/bin"
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/rpm/usr/lib:$PWD/rpm/usr/lib64"
#export LDFLAGS="-L$PWD/rpm/usr/lib -L$PWD/rpm/usr/lib64"
#export CPATH="$CPATH:$PWD/rpm/usr/include"
#cd ..
#
## Install mujoco
#wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
#mkdir .mujoco
#tar -xzvf mujoco210-linux-x86_64.tar.gz -C .mujoco
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/.mujoco/mujoco210/bin"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
#
## Install mujoco-py
#git clone https://github.com/openai/mujoco-py.git
#pip install -e mujoco-py
#export MUJOCO_PY_MUJOCO_PATH="$PWD/.mujoco/mujoco210"
#
#
#cd GuidedDataAugmentationForRobotics
##
#git clone https://github.com/Farama-Foundation/d4rl.git
#cd d4rl
#python3 -m pip install -e .
#python3 -m pip install -e src/d4rl
#python3 -m pip install -e src/custom-envs
#
#cd src
#
#
#export D4RL_DATASET_DIR=$(pwd)/.d4rl
#export WANDB_CONFIG_DIR=$(pwd)/.config/wandb

pid=$1 # command index
step=$2 # index within different runs of the same command
command=`tr '*' ' ' <<< $3` # replace * with space in command
echo $command

$($command --seed $step --run_id $step)

#python3 ./algorithms/cql.py --dataset_name dataset_expert_1000.hdf5

tar -czvf results_${pid}.tar.gz results/*
mv results_${pid}.tar.gz ../../..

cd ../../..
rm -rf ./workspace

