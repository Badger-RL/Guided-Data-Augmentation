import copy
import dataclasses
import glob
import json
import os
import random
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List

import d4rl
import gym, custom_envs
import h5py
import numpy as np
import torch
import wandb
from torch import nn

TensorBatch = List[torch.Tensor]
DATASET_SIZES = {
    'maze2d-umaze-v1': 987540,
    'maze2d-medium-v1': 1988111,
    'maze2d-large-v1': 3983273,
    'antmaze-umaze-diverse-v1': 1,
    'antmaze-medium-diverse-v1': 1,
    'antmaze-large-diverse-v1': 1,

}

### SAVE UTILS #########################################################################################################

def get_latest_run_id(save_dir: str) -> int:
    max_run_id = 0
    for path in glob.glob(os.path.join(save_dir, 'run_[0-9]*')):
        filename = os.path.basename(path)
        ext = filename.split('_')[-1]
        if ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id

def load_config(config):

    with open(os.path.join(config.config_dir, "config.json"), "w") as f:
        config_dict = json.load(f)
    return config_dict

def make_save_dir(config):
    # create save directories
    if config.run_id is not None:
        config.save_dir += f"/run_{config.run_id}"
    else:
        run_id = get_latest_run_id(save_dir=config.save_dir) + 1
        config.save_dir += f"/run_{run_id}"
    os.makedirs(config.save_dir, exist_ok=True)
    print(f"Results will be saved to: {config.save_dir}")

    # save config
    with open(os.path.join(config.save_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=4)

def wandb_init(config):
    # wandb logging
    wandb.login(key='7313077863c8908c24cc6058b99c2b2cc35d326b')

    if config.use_wandb and config.name is None:
        config.name = config.save_dir.replace('/', '_')

    config = dataclasses.asdict(config)
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

### TRAINING UTILS #####################################################################################################

def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)

def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    # else:
        # mean = dataset["rewards"].mean()
        # std = dataset["rewards"].std()
        # dataset["rewards"] = (dataset["rewards"] - mean)/std
        # print(mean, std)

        # mean = dataset["rewards"].mean()
        # dataset["rewards"] =  (dataset["rewards"] - mean)/1000


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

def load_dataset(config, env):
    dataset = {}
    if config.dataset_name:
        # local dataset
        data_hdf5 = h5py.File(f"{config.dataset_name}", "r")
        for key in data_hdf5.keys():
            dataset[key] = np.array(data_hdf5[key])
            print(dataset[key].shape)
    else:
        # remote dataset
        dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    return dataset, state_mean, state_std

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)

class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError

### TRAINING AND EVAL ##################################################################################################



@dataclasses.dataclass
class TrainConfigBase:
    # Experiment
    device: str = "cpu"
    env: str = "maze2d-umaze-v1"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(10e3)  # How often (time steps) we evaluate
    n_episodes: int = 50  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    load_model: str = ""  # Model load file name, "" doesn't load
    dataset_name: str = None
    deterministic_torch: bool = True
    # Normalization
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    # Augmentation
    aug_ratio: int = 1
    # Wandb logging
    use_wandb: bool = False
    project: str = "td3bc"
    group: str = "no_aug"
    name: str = None
    save_dir: str = "results"
    run_id: str = None
    save_policy: bool = False

    def __post_init__(self):
        self.name = self.name


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> (np.ndarray, np.ndarray):
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    successes = []
    info = {}
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            env.render()
        episode_rewards.append(episode_reward)
        if 'is_success' in info:
            successes.append(info['is_success'])

    return np.asarray(episode_rewards), np.array(successes)

def train_base(config, env, trainer):

    # create save directories
    make_save_dir(config=config)

    # setup wandb logging
    if config.use_wandb:
        wandb_init(config)

    # load dataset
    dataset, state_mean, state_std = load_dataset(config=config, env=env)

    # wrap env
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    # create replay buffer
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if config.buffer_size is None:
        config.buffer_size = len(dataset['observations'])
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    # load policy if applicable
    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
    actor = trainer.actor

    # local logging
    log_evaluations = defaultdict(lambda: [])
    log_stats = defaultdict(lambda: [])
    best_eval_score = -np.inf

    print("---------------------------------------")
    print(f"Training {trainer}, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # f = PointMazeGuidedAugmentationFunction(env)

    # for t in trange(int(config.max_timesteps), ncols=100):
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        stats_dict = trainer.update(batch)

        # log training statistics
        if config.use_wandb and t % 100 == 0:
            wandb.log(stats_dict, step=t)

        # evalutate agent
        if (t) % config.eval_freq == 0 or t == 0:
            print(f"Time steps: {t}")
            eval_scores, eval_successes = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            if len(eval_successes) > 0:
                eval_success_rate = eval_successes.mean()
            else:
                eval_success_rate = -np.inf
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
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

            # log training stats
            log_stats['timestep'].append(t)
            for key, val in stats_dict.items():
                log_stats[key].append(val)
            np.savez(os.path.join(config.save_dir, "stats.npz"), **log_stats)

            if config.save_policy:
                # save current model
                try:
                    test = config.cql_n_actions
                    torch.save(
                        trainer.actor.state_dict(),
                        os.path.join(config.save_dir, f"model_{t}.pt"),
                    )
                    torch.save(
                        trainer.state_dict(),
                        os.path.join(config.save_dir, f"model.pt"),
                    )
                    # save best model
                    if eval_score > best_eval_score:
                        best_eval_score = eval_score
                        torch.save(
                            trainer.actor.state_dict(),
                            os.path.join(config.save_dir, f"best_model.pt"),
                        )
                except:
                    is_cql = False
                    torch.save(
                        trainer.state_dict(),
                        os.path.join(config.save_dir, f"model_{t}.pt"),
                    )
                    torch.save(
                        trainer.state_dict(),
                        os.path.join(config.save_dir, f"model.pt"),
                    )
                    # save best model
                    if eval_score > best_eval_score:
                        best_eval_score = eval_score
                        torch.save(
                            trainer.state_dict(),
                            os.path.join(config.save_dir, f"best_model.pt"),
                        )

                torch.save(
                    trainer.actor.state_dict(),
                    os.path.join(config.save_dir, f"model.pt"),
                )

                # save best model
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    torch.save(
                        trainer.actor.state_dict(),
                        os.path.join(config.save_dir, f"best_model.pt"),
                    )

            if config.use_wandb:
                wandb.log(
                    {"return": eval_score,
                     "normalized_return": normalized_eval_score,
                     "success_rate": eval_success_rate
                     },
                    step=t,
                )
