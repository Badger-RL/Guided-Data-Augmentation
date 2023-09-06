import argparse
import copy
import importlib
import json
import os

import numpy as np
import torch

import discrete_BCQ
import DQN
import utils
import highway_env
import sys
from collections import defaultdict

from src.algorithms.utils import ReplayBuffer, load_dataset

def save_log(eval_scores, eval_successes, env, t, log_evaluations, config):
	eval_score = eval_scores.mean()
	if len(eval_successes) > 0:
		eval_success_rate = eval_successes.mean()
	else:
		eval_success_rate = -np.inf
	try:
		normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
	except:
		normalized_eval_score = eval_score
	print("---------------------------------------", file=sys.stderr)
	print(
		f"Iteration {t}: "
		f"Return: {eval_score:.3f}, "
		f"Normalized return: {normalized_eval_score:.3f}, "
		f"Success rate: {eval_success_rate:.3f}",
		file=sys.stderr
	)
	print("---------------------------------------", file=sys.stderr)

	# log evaluations
	log_evaluations['timestep'].append(t)
	log_evaluations['return'].append(eval_score)
	log_evaluations['normalized_return'].append(normalized_eval_score)
	log_evaluations['success_rate'].append(eval_success_rate)
	np.savez(os.path.join(config.save_dir, "evaluations.npz"), **log_evaluations)

	# TODO: log training stats
	# log_stats['timestep'].append(t)
	# for key, val in stats_dict.items():
	# 	log_stats[key].append(val)
	# np.savez(os.path.join(config.save_dir, "stats.npz"), **log_stats)

	# if config.save_policy:
	# 	# save current model
	# 	torch.save(
	# 		trainer.state_dict(),
	# 		os.path.join(config.save_dir, f"model.pt"),
	# 	)

	# 	# save best model
	# 	if eval_score > best_eval_score:
	# 		best_eval_score = eval_score
	# 		torch.save(
	# 			trainer.state_dict(),
	# 			os.path.join(config.save_dir, f"best_model.pt"),
	# 		)

def interact_with_environment(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters):
	# For saving files
	setting = f"{args.env}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"
	# Initialize and load policy
	policy = DQN.DQN(
		is_atari,
		num_actions,
		state_dim,
		device,
		parameters["discount"],
		parameters["optimizer"],
		parameters["optimizer_parameters"],
		parameters["polyak_target_update"],
		parameters["target_update_freq"],
		parameters["tau"],
		parameters["initial_eps"],
		parameters["end_eps"],
		parameters["eps_decay_period"],
		parameters["eval_eps"],
	)

	if args.generate_buffer: policy.load(f"./models/behavioral_{setting}")
	
	evaluations = []
	state, _ = env.reset()
	episode_start = True
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	low_noise_ep = np.random.uniform(0,1) < args.low_noise_p

	# Interact with the environment for max_timesteps
	log_evaluations = defaultdict(lambda: [])
	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# If generating the buffer, episode is low noise with p=low_noise_p.
		# If policy is low noise, we take random actions with p=eval_eps.
		# If the policy is high noise, we take random actions with p=rand_action_p.
		if args.generate_buffer:
			if not low_noise_ep and np.random.uniform(0,1) < args.rand_action_p - parameters["eval_eps"]:
				action = env.action_space.sample()
			else:
				action = policy.select_action(np.array(state), eval=True)

		if args.train_behavioral:
			if t < parameters["start_timesteps"]:
				action = env.action_space.sample()
			else:
				action = policy.select_action(np.array(state))

		# next_state, reward, done, info = env.step(action)

		next_state, reward, terminated, truncated, info = env.step(action)
		done = terminated

		episode_reward += reward

		# Only consider "done" if episode terminates due to failure condition
		done_float = float(terminated)

		# For atari, info[0] = clipped reward, info[1] = done_float
		if is_atari:
			reward = info[0]
			done_float = info[1]
			
		# Store data in replay buffer
		# print("add to replay_buffer: ", state)
		replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
		state = copy.copy(next_state)
		episode_start = False

		# Train agent after collecting sufficient data
		if args.train_behavioral and t >= parameters["start_timesteps"] and (t+1) % parameters["train_freq"] == 0:
			policy.train(replay_buffer, args.batch_size)

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, _ = env.reset()
			episode_start = True
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			low_noise_ep = np.random.uniform(0,1) < args.low_noise_p

		# Evaluate episode
		if args.train_behavioral and (t + 1) % parameters["eval_freq"] == 0:
			eval_scores, eval_successes = eval_policy(policy, args.env, args.seed)
			save_log(eval_scores, eval_successes, env, t, log_evaluations, args)

			# evaluations.append(eval_policy(policy, args.env, args.seed))
			# TODO: save the model
			np.save(f"./results/behavioral_{setting}", evaluations)
			policy.save(f"./models/behavioral_{setting}")

	# Save final policy
	if args.train_behavioral:
		policy.save(f"./models/behavioral_{setting}")

	# Save final buffer and performance
	else:
		eval_scores, eval_successes = eval_policy(policy, args.env, args.seed)
		save_log(eval_scores, eval_successes, env, t, log_evaluations, args)
		# evaluations.append(eval_policy(policy, args.env, args.seed))
		np.save(f"./results/buffer_performance_{setting}", evaluations)
		replay_buffer.save(f"./buffers/{buffer_name}")


# Trains BCQ offline
def train_BCQ(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters):
	# For saving files
	setting = f"{args.env}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"

	# Initialize and load policy
	policy = discrete_BCQ.discrete_BCQ(
		args,
		is_atari,
		num_actions,
		state_dim,
		device,
		args.BCQ_threshold,
		parameters["discount"],
		parameters["optimizer"],
		parameters["optimizer_parameters"],
		parameters["polyak_target_update"],
		parameters["target_update_freq"],
		parameters["tau"],
		parameters["initial_eps"],
		parameters["end_eps"],
		parameters["eps_decay_period"],
		parameters["eval_eps"]
	)

	# Load replay buffer	
	# replay_buffer.load(f"./buffers/{buffer_name}")
	
	evaluations = []
	episode_num = 0
	done = True 
	training_iters = 0
	log_evaluations = defaultdict(lambda: [])
	while training_iters < args.max_timesteps: 
		
		for _ in range(int(parameters["eval_freq"])):
			policy.train(replay_buffer, args.batch_size)

		if (training_iters) % args.eval_freq == 0 or training_iters == 0:
			eval_scores, eval_successes = eval_policy(policy, args.env, args.seed)
			save_log(eval_scores, eval_successes, env, training_iters, log_evaluations, args)

			# evaluations.append(eval_policy(policy, args.env, args.seed))
			# np.save(f"BCQ_{setting}", evaluations)

			training_iters += int(parameters["eval_freq"])
			# print(f"Training iterations: {training_iters}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=50):
	eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing)

	# eval_env.seed(seed + 100)

	# avg_reward = 0.0
	episode_rewards = []
	successes = []
	for _ in range(eval_episodes):
		state, _ = env.reset()
		state, _ = eval_env.reset()
		done = False
		episode_reward = 0.0
		while not done:
			action = policy.select_action(np.array(state), eval=True)
			if args.env == 'roundabout-v0' or args.env == 'two-way-v0':
				state, reward, terminated, truncated, info = env.step(action)
				done = terminated or truncated
				if 'is_success' in info:
					successes.append(info['is_success'])
			else:
				state, reward, done, _ = eval_env.step(action)
			episode_reward += reward
		episode_rewards.append(episode_reward)
	return np.asarray(episode_rewards), np.array(successes)


if __name__ == "__main__":

	# Atari Specific
	atari_preprocessing = {
		"frame_skip": 4,
		"frame_size": 84,
		"state_history": 4,
		"done_on_life_loss": False,
		"reward_clipping": True,
		"max_episode_timesteps": 27e3
	}

	atari_parameters = {
		# Exploration
		"start_timesteps": 2e4,
		"initial_eps": 1,
		"end_eps": 1e-2,
		"eps_decay_period": 25e4,
		# Evaluation
		"eval_freq": 5e4,
		"eval_eps": 1e-3,
		# Learning
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 32,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 0.0000625,
			"eps": 0.00015
		},
		"train_freq": 4,
		"polyak_target_update": False,
		"target_update_freq": 8e3,
		"tau": 1
	}

	regular_parameters = {
		# Exploration
		"start_timesteps": 2e2,
		"initial_eps": 0.1,
		"end_eps": 0.1,
		"eps_decay_period": 1,
		# Evaluation
		"eval_freq": 5e4,
		"eval_eps": 0,
		# Learning
		"discount": 0.99,
		"buffer_size": 15000,
		"batch_size": 32,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 5e-4
		},
		"train_freq": 1,
		"polyak_target_update": True,
		"target_update_freq": 1,
		"tau": 0.005
	}

	# Load parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default='two-way-v0')     # OpenAI gym environment name
	# parser.add_argument("--env", default='CartPole-v1')
	parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_name", default="Default")        # Prepends name to filename
	parser.add_argument("--max_timesteps", default=1000000, type=int)  # Max time steps to run environment or train for
	parser.add_argument("--BCQ_threshold", default=0.3, type=float)# Threshold hyper-parameter for BCQ
	parser.add_argument("--low_noise_p", default=0.2, type=float)  # Probability of a low noise episode when generating buffer
	parser.add_argument("--rand_action_p", default=0.2, type=float)# Probability of taking a random action when generating buffer, during non-low noise episode
	parser.add_argument("--train_behavioral", action="store_true") # If true, train behavioral policy
	parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
	parser.add_argument("--save_dir", default="./results", type=str) # Where to save results

	# DQN parameters
	parser.add_argument("--num_layers", default=1, type=int)
	parser.add_argument("--hidden_dim", default=256, type=int)
	parser.add_argument("--dataset_name", default="../../datasets/two-way-v0/no_aug.hdf5", type=str)

	# Extended arguments from `regular_parameters`
	parser.add_argument("--start_timesteps", default=2e2, type=float)
	parser.add_argument("--initial_eps", default=0.1, type=float)
	parser.add_argument("--end_eps", default=0.1, type=float)
	parser.add_argument("--eps_decay_period", default=1, type=int)
	parser.add_argument("--eval_freq", default=10e3, type=float)
	parser.add_argument("--eval_eps", default=0, type=int)
	parser.add_argument("--discount", default=0.99, type=float)
	parser.add_argument("--buffer_size", default=15000, type=int)
	parser.add_argument("--batch_size", default=32, type=int)
	parser.add_argument("--optimizer", default="Adam", type=str)
	parser.add_argument("--lr", default=3e-5, type=float) # for "optimizer_parameters" -> "lr"
	parser.add_argument("--train_freq", default=1, type=int)
	parser.add_argument("--polyak_target_update", action="store_true") # assuming it's boolean
	parser.add_argument("--target_update_freq", default=1, type=int)
	parser.add_argument("--tau", default=0.005, type=float)
	parser.add_argument("--normalize", default=1, type=int)
	parser.add_argument("--normalize_reward", default=0, type=int)

	args = parser.parse_args()

	regular_parameters = {
		"start_timesteps": args.start_timesteps,
		"initial_eps": args.initial_eps,
		"end_eps": args.end_eps,
		"eps_decay_period": args.eps_decay_period,
		"eval_freq": args.eval_freq,
		"eval_eps": args.eval_eps,
		"discount": args.discount,
		"buffer_size": args.buffer_size,
		"batch_size": args.batch_size,
		"optimizer": args.optimizer,
		"optimizer_parameters": {
			"lr": args.lr
		},
		"train_freq": args.train_freq,
		"polyak_target_update": args.polyak_target_update,
		"target_update_freq": args.target_update_freq,
		"tau": args.tau
	}

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./models"):
		os.makedirs("./models")

	if not os.path.exists("./buffers"):
		os.makedirs("./buffers")

	
	print("---------------------------------------")	
	if args.train_behavioral:
		print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
	elif args.generate_buffer:
		print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
	else:
		print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if args.train_behavioral and args.generate_buffer:
		print("Train_behavioral and generate_buffer cannot both be true.")
		exit()

	os.makedirs(args.save_dir, exist_ok=True)

	# Make env and determine properties
	env, is_atari, state_dim, num_actions = utils.make_env(args.env, atari_preprocessing)

	parameters = atari_parameters if is_atari else regular_parameters

	# Set seeds
	# env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Initialize buffer
	dataset, state_mean, state_std = load_dataset(config=args, env=env)
	# create replay buffer

	state_dim = env.observation_space.shape[0]
	if args.env == "two-way-v0":
		action_dim = env.action_space.n
	else:
		action_dim = env.action_space.shape[0]
	# action_dim = 1
	if args.buffer_size is None:
		args.buffer_size = len(dataset['observations'])
	replay_buffer = ReplayBuffer(
    	state_dim,
    	1,
        args.buffer_size,
        # args.device,
    )
	dataset['actions'] = dataset['actions'].reshape(-1, 1)
	replay_buffer.load_d4rl_dataset(dataset)
	replay_buffer._actions = replay_buffer._actions.type(torch.int64)

	# .astype(np.int64)
	# self._actions = self._actions.type(torch.int64)

	# replay_buffer = utils.ReplayBuffer(state_dim, is_atari, atari_preprocessing, parameters["batch_size"], parameters["buffer_size"], device)

	if args.train_behavioral or args.generate_buffer:
		interact_with_environment(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)
	else:
		train_BCQ(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)