"""Grid-world multi-agent environment for cooperation/competition studies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .communication import communication_feature_for_agent, normalize_communication_tokens
from .observation import build_global_planes, extract_partial_view
from .rewards import RewardConfig, compute_multi_agent_rewards

# Action mapping: stay, up, down, left, right
ACTION_DELTAS = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


@dataclass
class EnvConfig:
    height: int = 8
    width: int = 8
    num_agents: int = 2
    num_obstacles: int = 6
    num_resources: int = 5
    max_steps: int = 100
    observation_mode: str = "full"  # full | partial
    view_radius: int = 2
    communication_vocab_size: int = 0
    seed: int = 0
    reward: RewardConfig = field(default_factory=RewardConfig)


class MultiAgentGridWorldEnv:
    """A compact, research-friendly multi-agent environment.

    The environment is intentionally discrete and interpretable to make emergent
    behavior analysis easier.
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.agent_positions = np.zeros((config.num_agents, 2), dtype=np.int64)
        self.obstacles = np.empty((0, 2), dtype=np.int64)
        self.resources = np.empty((0, 2), dtype=np.int64)
        self.current_step = 0
        self.last_tokens = np.zeros(config.num_agents, dtype=np.int64)

    @property
    def action_dim(self) -> int:
        return len(ACTION_DELTAS)

    @property
    def observation_dim(self) -> int:
        h, w = self.config.height, self.config.width
        if self.config.observation_mode == "partial":
            patch = 2 * self.config.view_radius + 1
            spatial = 4 * patch * patch
        else:
            spatial = 4 * h * w
        return spatial + self.config.communication_vocab_size

    def reset(self, seed: int | None = None) -> Tuple[Dict[int, np.ndarray], Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.last_tokens = np.zeros(self.config.num_agents, dtype=np.int64)

        total_requested = self.config.num_agents + self.config.num_obstacles + self.config.num_resources
        capacity = self.config.height * self.config.width
        if total_requested > capacity:
            raise ValueError(
                "Requested entities exceed grid capacity: "
                f"{total_requested} > {capacity}. Adjust env config."
            )

        occupied = set()
        self.agent_positions = self._sample_unique_cells(self.config.num_agents, occupied)
        self.obstacles = self._sample_unique_cells(self.config.num_obstacles, occupied)
        self.resources = self._sample_unique_cells(self.config.num_resources, occupied)

        observations = {i: self._build_observation(i) for i in range(self.config.num_agents)}
        info = {
            "resources_remaining": int(self.resources.shape[0]),
            "collisions": 0,
            "success": False,
        }
        return observations, info

    def step(
        self,
        actions: Dict[int, int],
        comm_actions: Dict[int, int] | None = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, bool, Dict]:
        self.current_step += 1

        if self.config.communication_vocab_size > 0:
            self.last_tokens = normalize_communication_tokens(
                comm_actions,
                self.config.num_agents,
                self.config.communication_vocab_size,
            )

        old_positions = self.agent_positions.copy()
        proposed = self.agent_positions.copy()

        obstacle_set = {tuple(x) for x in self.obstacles.tolist()}
        for agent_id in range(self.config.num_agents):
            action = int(actions.get(agent_id, 0))
            dx, dy = ACTION_DELTAS.get(action, (0, 0))
            nx = int(self.agent_positions[agent_id, 0] + dx)
            ny = int(self.agent_positions[agent_id, 1] + dy)
            if 0 <= nx < self.config.height and 0 <= ny < self.config.width and (nx, ny) not in obstacle_set:
                proposed[agent_id] = [nx, ny]

        # Resolve multi-agent collisions (same target cell).
        collisions = set()
        for idx in range(self.config.num_agents):
            same = np.all(proposed == proposed[idx], axis=1)
            if np.count_nonzero(same) > 1:
                collisions.update(np.where(same)[0].tolist())

        # Resolve direct swaps (A->B and B->A in same step).
        for i in range(self.config.num_agents):
            for j in range(i + 1, self.config.num_agents):
                if np.array_equal(proposed[i], old_positions[j]) and np.array_equal(proposed[j], old_positions[i]):
                    collisions.add(i)
                    collisions.add(j)

        if collisions:
            for cidx in collisions:
                proposed[cidx] = old_positions[cidx]

        self.agent_positions = proposed

        collected_by: List[int] = []
        if self.resources.size:
            remaining = []
            for resource in self.resources:
                collectors = np.where(np.all(self.agent_positions == resource, axis=1))[0]
                if collectors.size:
                    collector = int(self.rng.choice(collectors))
                    collected_by.append(collector)
                else:
                    remaining.append(resource)
            if remaining:
                self.resources = np.vstack(remaining)
            else:
                self.resources = np.empty((0, 2), dtype=np.int64)

        success = self.resources.shape[0] == 0
        terminated = success
        truncated = self.current_step >= self.config.max_steps and not terminated

        rewards = compute_multi_agent_rewards(
            num_agents=self.config.num_agents,
            config=self.config.reward,
            collected_by=collected_by,
            collisions=sorted(collisions),
            episode_success=success,
        )

        observations = {i: self._build_observation(i) for i in range(self.config.num_agents)}
        info = {
            "resources_remaining": int(self.resources.shape[0]),
            "resources_collected": int(len(collected_by)),
            "collectors": collected_by,
            "collisions": int(len(collisions)),
            "collision_agents": sorted(collisions),
            "success": bool(success),
            "step": int(self.current_step),
        }

        return observations, rewards, terminated, truncated, info

    def sample_random_actions(self) -> Dict[int, int]:
        return {
            agent_id: int(self.rng.integers(0, self.action_dim))
            for agent_id in range(self.config.num_agents)
        }

    def render_ascii(self) -> str:
        grid = np.full((self.config.height, self.config.width), ".", dtype="<U2")
        for ox, oy in self.obstacles:
            grid[ox, oy] = "#"
        for rx, ry in self.resources:
            grid[rx, ry] = "R"
        for idx, (ax, ay) in enumerate(self.agent_positions):
            grid[ax, ay] = f"A{idx}"
        return "\n".join(" ".join(row.tolist()) for row in grid)

    def _sample_unique_cell(self, occupied: set) -> np.ndarray:
        while True:
            cell = (
                int(self.rng.integers(0, self.config.height)),
                int(self.rng.integers(0, self.config.width)),
            )
            if cell not in occupied:
                occupied.add(cell)
                return np.array(cell, dtype=np.int64)

    def _sample_unique_cells(self, count: int, occupied: set) -> np.ndarray:
        if count <= 0:
            return np.empty((0, 2), dtype=np.int64)
        return np.vstack([self._sample_unique_cell(occupied) for _ in range(count)])

    def _build_observation(self, agent_id: int) -> np.ndarray:
        planes = build_global_planes(
            grid_size=(self.config.height, self.config.width),
            agent_positions=self.agent_positions,
            agent_id=agent_id,
            obstacles=self.obstacles,
            resources=self.resources,
        )

        if self.config.observation_mode == "partial":
            spatial = extract_partial_view(
                planes,
                center=self.agent_positions[agent_id],
                radius=self.config.view_radius,
            )
        elif self.config.observation_mode == "full":
            spatial = planes
        else:
            raise ValueError(f"Unknown observation mode: {self.config.observation_mode}")

        spatial_flat = spatial.reshape(-1).astype(np.float32)
        comm_feature = communication_feature_for_agent(
            tokens=self.last_tokens,
            agent_id=agent_id,
            vocab_size=self.config.communication_vocab_size,
        )

        return np.concatenate([spatial_flat, comm_feature], dtype=np.float32)
