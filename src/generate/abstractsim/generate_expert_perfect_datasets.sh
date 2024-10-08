#!/bin/bash

source ../env/bin/activate

python3 ./generate_expert_perfect_dataset.py --random_actions 0 --num_samples 1000 --path push_ball_to_goal
python3 ./generate_expert_perfect_dataset.py --random_actions 0 --num_samples 5000 --path push_ball_to_goal
python3 ./generate_expert_perfect_dataset.py --random_actions 0 --num_samples 10000 --path push_ball_to_goal
python3 ./generate_expert_perfect_dataset.py --random_actions 0 --num_samples 50000 --path push_ball_to_goal
python3 ./generate_expert_perfect_dataset.py --random_actions 0 --num_samples 100000 --path push_ball_to_goal