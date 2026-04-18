"""Observation builders for full and partial observability."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def build_global_planes(
    grid_size: Tuple[int, int],
    agent_positions: np.ndarray,
    agent_id: int,
    obstacles: np.ndarray,
    resources: np.ndarray,
) -> np.ndarray:
    """Create dense multi-channel representation of the environment state.

    Channels:
    0: obstacles
    1: resources
    2: self position
    3: other agents
    """
    h, w = grid_size
    planes = np.zeros((4, h, w), dtype=np.float32)

    if obstacles.size:
        planes[0, obstacles[:, 0], obstacles[:, 1]] = 1.0
    if resources.size:
        planes[1, resources[:, 0], resources[:, 1]] = 1.0

    self_pos = agent_positions[agent_id]
    planes[2, self_pos[0], self_pos[1]] = 1.0

    for idx, pos in enumerate(agent_positions):
        if idx == agent_id:
            continue
        planes[3, pos[0], pos[1]] = 1.0

    return planes


def extract_partial_view(planes: np.ndarray, center: np.ndarray, radius: int) -> np.ndarray:
    """Extract a square local patch around center with zero padding."""
    channels, h, w = planes.shape
    window = 2 * radius + 1

    padded = np.pad(
        planes,
        ((0, 0), (radius, radius), (radius, radius)),
        mode="constant",
        constant_values=0.0,
    )
    cx, cy = int(center[0]), int(center[1])
    x0, y0 = cx, cy
    view = padded[:, x0 : x0 + window, y0 : y0 + window]
    if view.shape != (channels, window, window):
        safe = np.zeros((channels, window, window), dtype=np.float32)
        safe[:, : view.shape[1], : view.shape[2]] = view
        return safe
    return view
