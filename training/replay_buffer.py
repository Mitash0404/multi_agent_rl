"""Simple replay buffer for off-policy deep RL."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.buffer: Deque[Transition] = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            Transition(
                obs=np.asarray(obs, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                done=float(done),
            )
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample {batch_size} transitions from size {len(self.buffer)}")

        idxs = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch: List[Transition] = [self.buffer[int(i)] for i in idxs]

        obs = np.stack([t.obs for t in batch], axis=0)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_obs = np.stack([t.next_obs for t in batch], axis=0)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return obs, actions, rewards, next_obs, dones
