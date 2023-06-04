# source: https://github.com/sfujim/TD3_BC
# https://arxiv.org/pdf/2106.06860.pdf
from typing import Any, Dict, List, Tuple, Union
import copy
from dataclasses import dataclass

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.algorithms.utils import TrainConfigBase, train_base, return_reward_range, soft_update

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig(TrainConfigBase):
    # TD3
    buffer_size: int = None  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    gamma: float = 0.95  # gamma for
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    beta: float = 1
    # TD3 + BC
    alpha: float = 2.5  # Coefficient for Q function in actor loss


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa), None, None

class CriticVIB(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(CriticVIB, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256,256),
        )

        self.logstd = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256,256),
        )

        self.q = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        sa = torch.cat([state, action], 1)
        mu = self.mu(sa)
        logstd = self.logstd(sa)
        std = torch.exp(logstd)

        phi = mu + std * torch.randn_like(std)
        return self.q(phi), mu, std

def compute_kl_loss(mean, std):
    return -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()

class TD3_BC:  # noqa
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
        beta: float = 1e-2,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.beta = beta

        self.total_it = 0
        self.device = device

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, next_state, done = batch
        not_done = 1 - done

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_q1, _, _ = self.critic_1_target(next_state, next_action)
            target_q2, _, _ = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.gamma * target_q
            # log_dict["avg_target_q1"] = target_q1.mean().item()
            # log_dict["avg_target_q2"] = target_q2.mean().item()
            log_dict["avg_target_q"] = target_q.mean().item()


        # Get current Q estimates
        current_q1, mu1, std1 = self.critic_1(state, action)
        current_q2, mu2, std2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()

        # Compute KL loss
        # kl_loss = self.beta*(compute_kl_loss(mu1, std1) + compute_kl_loss(mu1, std1))
        # critic_loss += kl_loss
        # log_dict["kl_loss"] = kl_loss.item()

        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)
            q, mu, std = self.critic_1(state, pi)
            lmbda = self.alpha / q.abs().mean().detach()

            q_loss = q.mean()
            q_loss_lmbda = lmbda * q_loss
            bc_loss = F.mse_loss(pi, action)
            # actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
            actor_loss = -q_loss_lmbda + bc_loss

            log_dict["actor_loss"] = actor_loss.item()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

            self.last_lmbda = lmbda.item()
            self.last_q_loss = q_loss.item()
            self.last_q_loss_lmbda = q_loss_lmbda.item()
            self.last_bc_loss = bc_loss.item()
            self.last_actor_loss = actor_loss.item()

        elif self.total_it >= self.policy_freq:
            # These values exist only if at least one update was performed
            log_dict["lambda"] = self.last_lmbda
            log_dict["q_loss"] = self.last_q_loss
            log_dict["q_loss_lmbda"] = self.last_q_loss_lmbda
            log_dict["bc_loss"] = self.last_bc_loss
            log_dict["actor_loss"] = self.last_actor_loss

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):

    # make env
    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    critic_1 = Critic(state_dim, action_dim).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.critic_lr)
    critic_2 = Critic(state_dim, action_dim).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.critic_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "gamma": config.gamma,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # TD3 + BC
        "alpha": config.alpha,
    }

    # Initialize actor
    trainer = TD3_BC(**kwargs)
    train_base(config, env, trainer)

if __name__ == "__main__":
    train()
