import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gymnasium as gym
import highway_env
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from src.algorithms.utils import TensorBatch, TrainConfigBase, train_base, ReplayBuffer, eval_actor, bc_train_base

@dataclass
class TrainConfig(TrainConfigBase):
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    # BC
    buffer_size: int = None  # Replay buffer size
    frac: float = 1  # Best data fraction to use
    max_traj_len: int = 1000  # Max trajectory length
    learning_rate: float = 3e-5
    normalize: bool = True  # Normalize states
    n_layers: int = 2
    hidden_dims: int = 64


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)



class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 n_layers: int, hidden_dims: int):
        super(Actor, self).__init__()

        if n_layers == 1:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, action_dim),
                nn.Tanh(),
            )
        elif n_layers == 2:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, action_dim),
                nn.Tanh(),
            )
        elif n_layers == 3:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, action_dim),
                nn.Tanh(),
            )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()

def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    env = gym.wrappers.TransformObservation(env, normalize_state)
    return env


class BC:
    def __init__(
        self,
        max_action: np.ndarray,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, _, _, _ = batch

        # Compute actor loss
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    if isinstance(env.observation_space, gym.spaces.Dict):
        state_dim = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0]
    else:
        state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    actor = Actor(state_dim, action_dim, max_action, n_layers=config.n_layers, hidden_dims=config.hidden_dims).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "device": config.device,
    }

    # Initialize policy
    trainer = BC(**kwargs)
    bc_train_base(config, env, trainer)

if __name__ == "__main__":
    train()