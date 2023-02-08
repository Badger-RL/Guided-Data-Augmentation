#!/bin/bash


source ../env/bin/activate

python3 ./generate/generate_D4RL_dataset.py --use_policy --num_samples 1000 --path push_ball_to_goal
python3 ./generate/generate_D4RL_dataset.py --use_policy --num_samples 5000 --path push_ball_to_goal
python3 ./generate/generate_D4RL_dataset.py --use_policy --num_samples 10000 --path push_ball_to_goal
python3 ./generate/generate_D4RL_dataset.py --use_policy --num_samples 50000 --path push_ball_to_goal
python3 ./generate/generate_D4RL_dataset.py --use_policy --num_samples 100000 --path push_ball_to_goal
python3 ./generate/generate_D4RL_dataset.py  --num_samples 1000 --path push_ball_to_goal
python3 ./generate/generate_D4RL_dataset.py  --num_samples 5000 --path push_ball_to_goal
python3 ./generate/generate_D4RL_dataset.py  --num_samples 10000 --path push_ball_to_goal
python3 ./generate/generate_D4RL_dataset.py  --num_samples 50000 --path push_ball_to_goal
python3 ./generate/generate_D4RL_dataset.py  --num_samples 100000 --path push_ball_to_goal
