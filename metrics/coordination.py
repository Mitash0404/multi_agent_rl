"""Coordination and emergent behavior metrics."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def compute_coordination_score(
    num_steps: int,
    collisions: int,
    resources_collected: int,
    success: bool,
    num_agents: int,
    collectors: Iterable[int],
) -> float:
    """Return a [0, 1] coordination score from episode-level stats."""
    steps = max(int(num_steps), 1)
    interference = 1.0 - (float(collisions) / float(steps * max(num_agents, 1)))
    interference = float(np.clip(interference, 0.0, 1.0))

    efficiency = float(resources_collected) / float(steps)
    efficiency = float(np.clip(efficiency * 2.0, 0.0, 1.0))

    collector_set = set(int(c) for c in collectors)
    role_spread = float(len(collector_set)) / float(max(num_agents, 1))

    success_bonus = 1.0 if success else 0.0
    score = 0.45 * interference + 0.25 * efficiency + 0.2 * role_spread + 0.1 * success_bonus
    return float(np.clip(score, 0.0, 1.0))


def compute_emergent_behavior_stats(collision_agents: List[int], collectors: List[int], num_agents: int) -> Dict[str, float]:
    """Lightweight heuristics for emergent behavior analysis."""
    blocking_intensity = float(len(collision_agents)) / float(max(num_agents, 1))

    counts = np.zeros(num_agents, dtype=np.float32)
    for c in collectors:
        if 0 <= c < num_agents:
            counts[c] += 1.0

    monopolization = float(counts.max() / max(float(counts.sum()), 1.0))
    specialization = float(np.std(counts) / max(float(np.mean(counts)), 1.0))

    return {
        "blocking_intensity": float(np.clip(blocking_intensity, 0.0, 1.0)),
        "monopolization": float(np.clip(monopolization, 0.0, 1.0)),
        "specialization": float(max(specialization, 0.0)),
    }
