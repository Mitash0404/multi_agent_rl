"""Evaluation utilities for trained multi-agent policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from env.gridworld_env import MultiAgentGridWorldEnv
from metrics.coordination import compute_coordination_score


@dataclass
class EvalStats:
    mean_reward: float
    success_rate: float
    mean_coordination: float
    mean_collisions: float
    mean_steps: float


def evaluate_policies(
    env: MultiAgentGridWorldEnv,
    agents: Dict[int, object],
    num_episodes: int = 20,
    shared_policy: object | None = None,
    seed_offset: int = 10_000,
) -> EvalStats:
    episode_rewards: List[float] = []
    successes: List[float] = []
    coordination_scores: List[float] = []
    collision_counts: List[float] = []
    step_counts: List[float] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False

        total_rewards = {aid: 0.0 for aid in range(env.config.num_agents)}
        total_collisions = 0
        total_steps = 0
        total_resources = 0
        all_collectors: List[int] = []
        success = False

        while not done:
            if shared_policy is not None:
                actions = shared_policy.select_actions(obs, epsilon=0.0)
            else:
                actions = {aid: agents[aid].select_action(obs[aid], epsilon=0.0) for aid in agents}

            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            for aid, rew in rewards.items():
                total_rewards[aid] += rew

            total_collisions += int(info["collisions"])
            total_resources += int(info["resources_collected"])
            total_steps += 1
            all_collectors.extend(info["collectors"])
            success = bool(info["success"])
            obs = next_obs

        coordination = compute_coordination_score(
            num_steps=total_steps,
            collisions=total_collisions,
            resources_collected=total_resources,
            success=success,
            num_agents=env.config.num_agents,
            collectors=all_collectors,
        )

        episode_rewards.append(float(np.mean(list(total_rewards.values()))))
        successes.append(float(success))
        coordination_scores.append(coordination)
        collision_counts.append(float(total_collisions))
        step_counts.append(float(total_steps))

    return EvalStats(
        mean_reward=float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        success_rate=float(np.mean(successes)) if successes else 0.0,
        mean_coordination=float(np.mean(coordination_scores)) if coordination_scores else 0.0,
        mean_collisions=float(np.mean(collision_counts)) if collision_counts else 0.0,
        mean_steps=float(np.mean(step_counts)) if step_counts else 0.0,
    )
