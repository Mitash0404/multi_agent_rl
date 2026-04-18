"""Config-driven training loop for multi-agent DQN baselines."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from agents.dqn_agent import DQNAgent, DQNConfig
from agents.shared_policy import SharedPolicyDQN
from env.gridworld_env import EnvConfig, MultiAgentGridWorldEnv
from env.rewards import RewardConfig
from metrics.coordination import compute_coordination_score, compute_emergent_behavior_stats
from metrics.plotting import save_training_curves
from metrics.stability import convergence_episode, rolling_variance
from training.evaluate import evaluate_policies


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(config: Dict) -> MultiAgentGridWorldEnv:
    env_cfg = config.get("env", {})
    reward_cfg = config.get("reward", {})

    reward = RewardConfig(**reward_cfg)
    env = MultiAgentGridWorldEnv(
        EnvConfig(
            height=int(env_cfg.get("height", 8)),
            width=int(env_cfg.get("width", 8)),
            num_agents=int(env_cfg.get("num_agents", 2)),
            num_obstacles=int(env_cfg.get("num_obstacles", 6)),
            num_resources=int(env_cfg.get("num_resources", 5)),
            max_steps=int(env_cfg.get("max_steps", 100)),
            observation_mode=str(env_cfg.get("observation_mode", "full")),
            view_radius=int(env_cfg.get("view_radius", 2)),
            communication_vocab_size=int(env_cfg.get("communication_vocab_size", 0)),
            seed=int(config.get("experiment", {}).get("seed", 0)),
            reward=reward,
        )
    )
    return env


def epsilon_for_episode(episode: int, start: float, end: float, decay_episodes: int) -> float:
    if decay_episodes <= 0:
        return end
    slope = (end - start) / float(decay_episodes)
    eps = start + slope * episode
    return float(max(end, eps))


def init_agents(config: Dict, env: MultiAgentGridWorldEnv) -> Tuple[Dict[int, DQNAgent], SharedPolicyDQN | None]:
    dqn_cfg = DQNConfig(**config.get("dqn", {}))
    train_cfg = config.get("training", {})
    device = str(train_cfg.get("device", "cpu"))
    seed = int(config.get("experiment", {}).get("seed", 0))
    policy_mode = str(train_cfg.get("policy_mode", "independent")).lower()

    if policy_mode == "shared":
        shared = SharedPolicyDQN(
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            num_agents=env.config.num_agents,
            config=dqn_cfg,
            seed=seed,
            device=device,
        )
        return {}, shared

    agents = {
        aid: DQNAgent(
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            config=dqn_cfg,
            seed=seed + aid,
            device=device,
        )
        for aid in range(env.config.num_agents)
    }
    return agents, None


def run_training(config: Dict, output_dir: str | Path) -> Dict:
    exp_cfg = config.get("experiment", {})
    train_cfg = config.get("training", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(exp_cfg.get("seed", 0))
    set_global_seed(seed)

    env = build_env(config)
    agents, shared_policy = init_agents(config, env)

    num_episodes = int(train_cfg.get("num_episodes", 500))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 100))
    eval_interval = int(train_cfg.get("eval_interval", 50))
    eval_episodes = int(train_cfg.get("eval_episodes", 20))
    log_interval = int(train_cfg.get("log_interval", 10))
    epsilon_start = float(train_cfg.get("epsilon_start", 1.0))
    epsilon_end = float(train_cfg.get("epsilon_end", 0.05))
    epsilon_decay_episodes = int(train_cfg.get("epsilon_decay_episodes", 300))

    metrics_csv = output_dir / "episode_metrics.csv"
    summary_json = output_dir / "summary.json"
    plot_path = output_dir / "training_curves.png"

    metric_rows: List[Dict] = []

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "epsilon",
                "mean_reward",
                "success",
                "coordination",
                "collisions",
                "resources_collected",
                "steps",
                "loss",
                "td_error",
                "blocking_intensity",
                "monopolization",
                "specialization",
                "eval_mean_reward",
                "eval_success_rate",
                "eval_coordination",
            ],
        )
        writer.writeheader()

        for episode in range(1, num_episodes + 1):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            epsilon = epsilon_for_episode(episode - 1, epsilon_start, epsilon_end, epsilon_decay_episodes)

            total_rewards = {aid: 0.0 for aid in range(env.config.num_agents)}
            total_collisions = 0
            total_resources = 0
            total_steps = 0
            all_collectors: List[int] = []
            success = False
            losses = []
            td_errors = []

            while not done:
                if shared_policy is not None:
                    actions = shared_policy.select_actions(obs, epsilon=epsilon)
                else:
                    actions = {
                        aid: agents[aid].select_action(obs[aid], epsilon=epsilon)
                        for aid in range(env.config.num_agents)
                    }

                comm_actions = None
                if env.config.communication_vocab_size > 0:
                    comm_actions = {
                        aid: int(np.random.randint(0, env.config.communication_vocab_size))
                        for aid in range(env.config.num_agents)
                    }

                next_obs, rewards, terminated, truncated, info = env.step(actions, comm_actions=comm_actions)
                done = terminated or truncated

                for aid in range(env.config.num_agents):
                    reward = rewards[aid]
                    total_rewards[aid] += reward

                    if shared_policy is not None:
                        shared_policy.store_transition(obs[aid], actions[aid], reward, next_obs[aid], done)
                        update_out = shared_policy.update()
                    else:
                        agents[aid].store_transition(obs[aid], actions[aid], reward, next_obs[aid], done)
                        update_out = agents[aid].update()

                    if update_out is not None:
                        losses.append(update_out["loss"])
                        td_errors.append(update_out["td_error"])

                total_collisions += int(info["collisions"])
                total_resources += int(info["resources_collected"])
                total_steps += 1
                all_collectors.extend(info["collectors"])
                success = bool(info["success"])

                obs = next_obs

            mean_reward = float(np.mean(list(total_rewards.values())))
            coordination = compute_coordination_score(
                num_steps=total_steps,
                collisions=total_collisions,
                resources_collected=total_resources,
                success=success,
                num_agents=env.config.num_agents,
                collectors=all_collectors,
            )
            emergent = compute_emergent_behavior_stats(
                collision_agents=info.get("collision_agents", []),
                collectors=all_collectors,
                num_agents=env.config.num_agents,
            )

            eval_mean_reward = np.nan
            eval_success_rate = np.nan
            eval_coordination = np.nan

            if episode % eval_interval == 0:
                eval_stats = evaluate_policies(
                    env=env,
                    agents=agents,
                    num_episodes=eval_episodes,
                    shared_policy=shared_policy,
                    seed_offset=100_000 + episode,
                )
                eval_mean_reward = eval_stats.mean_reward
                eval_success_rate = eval_stats.success_rate
                eval_coordination = eval_stats.mean_coordination

            row = {
                "episode": episode,
                "epsilon": epsilon,
                "mean_reward": mean_reward,
                "success": float(success),
                "coordination": coordination,
                "collisions": total_collisions,
                "resources_collected": total_resources,
                "steps": total_steps,
                "loss": float(np.mean(losses)) if losses else np.nan,
                "td_error": float(np.mean(td_errors)) if td_errors else np.nan,
                "blocking_intensity": emergent["blocking_intensity"],
                "monopolization": emergent["monopolization"],
                "specialization": emergent["specialization"],
                "eval_mean_reward": eval_mean_reward,
                "eval_success_rate": eval_success_rate,
                "eval_coordination": eval_coordination,
            }
            writer.writerow(row)
            metric_rows.append(row)

            if episode % log_interval == 0 or episode == 1:
                print(
                    f"[Episode {episode:4d}] reward={mean_reward:7.3f} success={float(success):.2f} "
                    f"coord={coordination:.3f} eps={epsilon:.3f}"
                )

            if episode % checkpoint_interval == 0:
                ckpt_dir = output_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                if shared_policy is not None:
                    shared_policy.save(ckpt_dir / f"shared_dqn_ep{episode}.pt")
                else:
                    for aid, agent in agents.items():
                        agent.save(ckpt_dir / f"agent_{aid}_ep{episode}.pt")

    series = {
        "mean_reward": [row["mean_reward"] for row in metric_rows],
        "success": [row["success"] for row in metric_rows],
        "coordination": [row["coordination"] for row in metric_rows],
    }
    save_training_curves(series, str(plot_path))

    rewards = [row["mean_reward"] for row in metric_rows]
    summary = {
        "run_name": str(exp_cfg.get("name", "run")),
        "seed": seed,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_episodes": num_episodes,
        "final_mean_reward": float(rewards[-1]) if rewards else 0.0,
        "best_mean_reward": float(np.max(rewards)) if rewards else 0.0,
        "mean_success_rate": float(np.mean([row["success"] for row in metric_rows])) if metric_rows else 0.0,
        "mean_coordination": float(np.mean([row["coordination"] for row in metric_rows])) if metric_rows else 0.0,
        "convergence_episode": convergence_episode(rewards, window=min(50, max(5, num_episodes // 10))),
        "rolling_reward_variance_tail": float(
            np.mean(rolling_variance(rewards, window=min(25, max(5, num_episodes // 20)))[-10:])
        )
        if rewards
        else 0.0,
        "env_config": asdict(env.config),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "output_dir": str(output_dir),
        "metrics_csv": str(metrics_csv),
        "summary_json": str(summary_json),
        "plot": str(plot_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train independent/shared DQN policies in multi-agent gridworld.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to results/logs/<config_name>_<timestamp>",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.output_dir is None:
        stem = Path(args.config).stem
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / "logs" / f"{stem}_{run_stamp}"
    else:
        output_dir = Path(args.output_dir)

    outputs = run_training(config, output_dir)
    print("Training complete. Artifacts:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
