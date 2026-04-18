"""Reward function implementations for MARL settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RewardConfig:
    mode: str = "cooperative"
    step_penalty: float = -0.01
    collision_penalty: float = -0.1
    resource_reward: float = 1.0
    goal_reward: float = 5.0
    team_reward_weight: float = 1.0
    individual_reward_weight: float = 1.0
    competitive_steal_penalty: float = 0.25


def compute_multi_agent_rewards(
    num_agents: int,
    config: RewardConfig,
    collected_by: List[int],
    collisions: List[int],
    episode_success: bool,
) -> Dict[int, float]:
    """Compute per-agent rewards for one environment step."""
    rewards = {agent_id: float(config.step_penalty) for agent_id in range(num_agents)}
    for agent_id in collisions:
        rewards[agent_id] += config.collision_penalty

    resource_count = len(collected_by)
    mode = config.mode.lower()

    if mode == "cooperative":
        team_gain = config.resource_reward * resource_count * config.team_reward_weight
        for agent_id in rewards:
            rewards[agent_id] += team_gain
        if episode_success:
            for agent_id in rewards:
                rewards[agent_id] += config.goal_reward

    elif mode == "competitive":
        for collector in collected_by:
            rewards[collector] += config.resource_reward
            for agent_id in rewards:
                if agent_id != collector:
                    rewards[agent_id] -= config.resource_reward * config.competitive_steal_penalty
        if episode_success and collected_by:
            for collector in set(collected_by):
                rewards[collector] += config.goal_reward * 0.5

    elif mode == "mixed":
        team_gain = config.resource_reward * resource_count * config.team_reward_weight
        for agent_id in rewards:
            rewards[agent_id] += team_gain
        for collector in collected_by:
            rewards[collector] += config.resource_reward * config.individual_reward_weight
        if episode_success:
            for agent_id in rewards:
                rewards[agent_id] += config.goal_reward * config.team_reward_weight

    else:
        raise ValueError(f"Unsupported reward mode: {config.mode}")

    return rewards
