"""Utilities for optional agent communication channels."""

from __future__ import annotations

from typing import Dict

import numpy as np


def normalize_communication_tokens(
    comm_actions: Dict[int, int] | None,
    num_agents: int,
    vocab_size: int,
) -> np.ndarray:
    """Convert raw communication actions into a dense token array.

    Missing or out-of-range tokens are mapped to zero.
    """
    tokens = np.zeros(num_agents, dtype=np.int64)
    if not comm_actions:
        return tokens

    for agent_id, token in comm_actions.items():
        if agent_id < 0 or agent_id >= num_agents:
            continue
        if token < 0 or token >= vocab_size:
            continue
        tokens[agent_id] = token
    return tokens


def communication_feature_for_agent(
    tokens: np.ndarray,
    agent_id: int,
    vocab_size: int,
) -> np.ndarray:
    """Build normalized token histogram for all *other* agents."""
    if vocab_size <= 0:
        return np.empty((0,), dtype=np.float32)

    mask = np.ones(tokens.shape[0], dtype=bool)
    if 0 <= agent_id < tokens.shape[0]:
        mask[agent_id] = False
    others = tokens[mask]

    if others.size == 0:
        return np.zeros(vocab_size, dtype=np.float32)

    hist = np.bincount(others, minlength=vocab_size).astype(np.float32)
    hist /= max(float(others.size), 1.0)
    return hist
