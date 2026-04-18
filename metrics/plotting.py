"""Plotting helpers for training and experiment analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np


def save_training_curves(metrics: Dict[str, Iterable[float]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(1, len(list(metrics["mean_reward"])) + 1)
    mean_reward = np.asarray(list(metrics["mean_reward"]), dtype=np.float32)
    success_rate = np.asarray(list(metrics["success"]), dtype=np.float32)
    coordination = np.asarray(list(metrics["coordination"]), dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(episodes, mean_reward, label="Mean Reward", color="#1f77b4")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].plot(episodes, success_rate, label="Success", color="#2ca02c")
    axes[1].set_ylabel("Success")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    axes[2].plot(episodes, coordination, label="Coordination", color="#ff7f0e")
    axes[2].set_ylabel("Coordination")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def save_reward_mode_comparison(series: Dict[str, Iterable[float]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, values in series.items():
        y = np.asarray(list(values), dtype=np.float32)
        x = np.arange(1, y.size + 1)
        ax.plot(x, y, label=name)

    ax.set_title("Reward Structure Comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
