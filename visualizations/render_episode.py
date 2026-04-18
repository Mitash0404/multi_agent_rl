"""Render one episode as an animated GIF from random or trained policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap

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


def _frame_from_env(env: MultiAgentGridWorldEnv) -> np.ndarray:
    frame = np.zeros((env.config.height, env.config.width), dtype=np.int32)

    for x, y in env.obstacles:
        frame[x, y] = 1
    for x, y in env.resources:
        frame[x, y] = 2
    for aid, (x, y) in enumerate(env.agent_positions):
        frame[x, y] = 3 + aid

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one episode as GIF")
    parser.add_argument("--config", type=str, default="configs/cooperative.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/plots/episode.gif")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = build_env(cfg)

    agents = None
    if args.checkpoint_dir:
        agents = _load_agents(env, Path(args.checkpoint_dir))

    obs, _ = env.reset(seed=args.seed)
    done = False
    frames: List[np.ndarray] = [_frame_from_env(env)]

    while not done:
        if agents is None:
            actions = env.sample_random_actions()
        else:
            actions = {aid: agents[aid].select_action(obs[aid], epsilon=0.0) for aid in agents}

        obs, _, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        frames.append(_frame_from_env(env))

    max_class = 3 + env.config.num_agents
    colors = ["#f7f7f7", "#3b3b3b", "#f0b429", "#277da1", "#577590", "#43aa8b", "#f94144", "#9d4edd"]
    cmap = ListedColormap(colors[: max_class + 1])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Multi-Agent Episode")
    ax.set_xticks([])
    ax.set_yticks([])

    img = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=max_class)

    def update(frame_idx: int):
        img.set_data(frames[frame_idx])
        ax.set_xlabel(f"Step {frame_idx}")
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=350, blit=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ani.save(output, writer=animation.PillowWriter(fps=3))
    plt.close(fig)

    print(f"Saved episode render: {output}")


if __name__ == "__main__":
    main()
