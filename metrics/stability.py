"""Policy and reward stability metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def rolling_variance(values: Iterable[float], window: int = 25) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float32)
    if values.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if window <= 1:
        return np.zeros_like(values)

    out = np.zeros_like(values)
    for i in range(values.size):
        lo = max(0, i - window + 1)
        out[i] = np.var(values[lo : i + 1])
    return out


def convergence_episode(values: Iterable[float], window: int = 50, variance_threshold: float = 0.05) -> int | None:
    """Return earliest episode index where reward variance remains low."""
    vals = np.asarray(list(values), dtype=np.float32)
    if vals.size < window:
        return None

    rolling_var = rolling_variance(vals, window=window)
    stable = np.where(rolling_var < variance_threshold)[0]
    if stable.size == 0:
        return None
    return int(stable[0])
