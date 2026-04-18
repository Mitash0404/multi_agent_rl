"""Independent DQN agent for discrete-action environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    min_buffer_size: int = 1_000
    target_update_interval: int = 250
    gradient_clip_norm: float = 10.0
    hidden_dim: int = 256


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: DQNConfig,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.config = config
        self.device = torch.device(device)

        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.q_network = QNetwork(obs_dim, action_dim, hidden_dim=config.hidden_dim).to(self.device)
        self.target_network = QNetwork(obs_dim, action_dim, hidden_dim=config.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(capacity=config.buffer_size, seed=seed)
        self.train_steps = 0

    def select_action(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.action_dim))

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q_network(obs_t)
            return int(torch.argmax(q_vals, dim=1).item())

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def update(self) -> Dict[str, float] | None:
        if len(self.replay_buffer) < max(self.config.batch_size, self.config.min_buffer_size):
            return None

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.config.batch_size)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_network(obs_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_obs_t).max(dim=1).values
            targets = rewards_t + (1.0 - dones_t) * self.config.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.config.gradient_clip_norm)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        td_error = torch.mean(torch.abs(targets - q_values)).item()
        return {
            "loss": float(loss.item()),
            "td_error": float(td_error),
            "q_mean": float(q_values.mean().item()),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(payload["q_network"])
        self.target_network.load_state_dict(payload["target_network"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.train_steps = int(payload.get("train_steps", 0))
