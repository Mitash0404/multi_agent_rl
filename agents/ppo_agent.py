"""PPO agent placeholder for future deep MARL expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.backbone(obs)
        return {
            "logits": self.policy_head(x),
            "value": self.value_head(x).squeeze(-1),
        }


class PPOAgent:
    """Inference-ready PPO actor-critic scaffold.

    Training hooks are intentionally left for future work once DQN baselines are complete.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = ActorCritic(obs_dim, action_dim, hidden_dim=config.hidden_dim).to(self.device)

    def select_action(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.model(obs_t)["logits"]
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1)
            return int(action.item())
