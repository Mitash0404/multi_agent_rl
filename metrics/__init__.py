"""Metrics package for MARL analysis."""

from .coordination import compute_coordination_score, compute_emergent_behavior_stats
from .stability import convergence_episode, rolling_variance

__all__ = [
    "compute_coordination_score",
    "compute_emergent_behavior_stats",
    "rolling_variance",
    "convergence_episode",
]
