"""Generate occupancy heatmaps from random rollouts or trained policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from agents.dqn_agent import DQNAgent, DQNConfig
from env.gridworld_env import MultiAgentGridWorldEnv
from training.train import build_env, load_config


def _load_agents(env: MultiAgentGridWorldEnv, checkpoint_dir: Path) -> Dict[int, DQNAgent]:
    agents: Dict[int, DQNAgent] = {}
    for aid in range(env.config.num_agents):
        agent = DQNAgent(
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            config=DQNConfig(),
            seed=aid,
            device="cpu",
        )
        ckpts = sorted(checkpoint_dir.glob(f"agent_{aid}_ep*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found for agent {aid} in {checkpoint_dir}")
        agent.load(ckpts[-1])
        agents[aid] = agent
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent occupancy heatmaps")
    parser.add_argument("--config", type=str, default="configs/cooperative.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/plots/occupancy_heatmaps.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = build_env(cfg)

    agents = None
    if args.checkpoint_dir:
        agents = _load_agents(env, Path(args.checkpoint_dir))

    occupancy = np.zeros((env.config.num_agents, env.config.height, env.config.width), dtype=np.float32)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done = False

        while not done:
            for aid, (x, y) in enumerate(env.agent_positions):
                occupancy[aid, x, y] += 1.0

            if agents is None:
                actions = env.sample_random_actions()
            else:
                actions = {aid: agents[aid].select_action(obs[aid], epsilon=0.0) for aid in agents}

            obs, _, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated

    fig, axes = plt.subplots(1, env.config.num_agents, figsize=(5 * env.config.num_agents, 4), squeeze=False)
    for aid in range(env.config.num_agents):
        ax = axes[0, aid]
        hm = occupancy[aid] / max(float(occupancy[aid].sum()), 1.0)
        im = ax.imshow(hm, cmap="magma")
        ax.set_title(f"Agent {aid} Occupancy")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

    print(f"Saved heatmaps: {output}")


if __name__ == "__main__":
    main()
