from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from dataclasses import asdict, dataclass
import os
import random
import uuid

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional

from src.algorithms.utils import TrainConfigBase, train_base, TensorBatch


@dataclass
class TrainConfig(TrainConfigBase):
    buffer_size: int = None
    batch_size: int = 256
    eval_freq: int = 1000
    n_episodes: int = 10

    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    awac_lambda: float = 1.0

class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.min_action = min_action
        self.max_action = max_action

    def _get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
        mean = self.mlp(state)
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return log_prob

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(self.min_action, self.max_action)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if self.mlp.training:
            action_t = policy.sample()
        else:
            action_t = policy.mean
        action = action_t[0].cpu().numpy()
        return action


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_value = self.mlp(torch.cat([state, action], dim=-1))
        return q_value


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class AdvantageWeightedActorCritic:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 5e-3,  # parameter for the soft target update,
        awac_lambda: float = 1.0,
        exp_adv_max: float = 100.0,
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer

        self.critic_1 = critic_1
        self.critic_1_optimizer = critic_1_optimizer
        self.target_critic_1 = deepcopy(critic_1)

        self.critic_2 = critic_2
        self.critic_2_optimizer = critic_2_optimizer
        self.target_critic_2 = deepcopy(critic_2)

        self.gamma = gamma
        self.tau = tau
        self.awac_lambda = awac_lambda
        self.exp_adv_max = exp_adv_max

    def _actor_loss(self, states, actions):
        with torch.no_grad():
            pi_action, _ = self.actor(states)
            v = torch.min(
                self.critic_1(states, pi_action), self.critic_2(states, pi_action)
            )

            q = torch.min(
                self.critic_1(states, actions), self.critic_2(states, actions)
            )
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self.awac_lambda), self.exp_adv_max
            )

        action_log_prob = self.actor.log_prob(states, actions)
        loss = (-action_log_prob * weights).mean()
        return loss

    def _critic_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_actions, _ = self.actor(next_states)

            q_next = torch.min(
                self.target_critic_1(next_states, next_actions),
                self.target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)

        q1_loss = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        loss = q1_loss + q2_loss

        critic_stats = {
            # 'avg_q1': q1.mean().item(),
            # 'avg_q2': q1.mean().item(),
            'avg_target_q': q_target.mean().item(),
        }

        return loss, critic_stats

    def _update_critic(self, states, actions, rewards, dones, next_states):
        loss, critic_stats = self._critic_loss(states, actions, rewards, dones, next_states)
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        return loss.item(), critic_stats

    def _update_actor(self, states, actions):
        loss = self._actor_loss(states, actions)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss.item()

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        critic_loss, critic_stats = self._update_critic(states, actions, rewards, dones, next_states)
        actor_loss = self._update_actor(states, actions)

        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)

        result = {"critic_loss": critic_loss, "actor_loss": actor_loss}
        result.update(critic_stats)
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_2.load_state_dict(state_dict["critic_2"])


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor_critic_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.hidden_dim,
    }

    actor = Actor(**actor_critic_kwargs)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_1 = Critic(**actor_critic_kwargs)
    critic_2 = Critic(**actor_critic_kwargs)
    critic_1.to(config.device)
    critic_2.to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate)

    trainer = AdvantageWeightedActorCritic(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        awac_lambda=config.awac_lambda,
    )
    train_base(config, env, trainer)
if __name__ == "__main__":
    train()
