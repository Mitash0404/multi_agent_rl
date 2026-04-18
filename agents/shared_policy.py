"""Parameter-sharing wrapper for shared-policy DQN experiments."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .dqn_agent import DQNAgent, DQNConfig


class SharedPolicyDQN:
    """Single DQN policy reused across all agents."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        config: DQNConfig,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.num_agents = num_agents
        self.base = DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config,
            seed=seed,
            device=device,
        )

    def select_actions(self, observations: Dict[int, np.ndarray], epsilon: float) -> Dict[int, int]:
        return {
            aid: self.base.select_action(obs, epsilon=epsilon)
            for aid, obs in observations.items()
        }

    def store_transition(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.base.store_transition(obs, action, reward, next_obs, done)

    def update(self):
        return self.base.update()

    def save(self, path: str) -> None:
        self.base.save(path)

    def load(self, path: str) -> None:
        self.base.load(path)
