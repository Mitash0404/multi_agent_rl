"""Run cooperative/competitive/mixed reward ablations."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from metrics.plotting import save_reward_mode_comparison
from training.train import load_config, run_training


def _read_mean_reward_series(metrics_csv: str) -> List[float]:
    values: List[float] = []
    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row["mean_reward"]))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward structure study")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/cooperative.yaml", "configs/competitive.yaml", "configs/mixed.yaml"],
        help="Config files to compare",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for experiment artifacts",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root) if args.output_root else Path("results") / "logs" / f"reward_study_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    reward_series: Dict[str, List[float]] = {}
    summary_rows: List[Dict] = []

    for cfg_path in args.configs:
        cfg = load_config(cfg_path)
        run_name = cfg.get("experiment", {}).get("name", Path(cfg_path).stem)
        run_dir = out_root / run_name

        outputs = run_training(deepcopy(cfg), run_dir)
        mean_rewards = _read_mean_reward_series(outputs["metrics_csv"])
        reward_series[run_name] = mean_rewards

        summary_rows.append(
            {
                "run_name": run_name,
                "config": cfg_path,
                "final_mean_reward": mean_rewards[-1] if mean_rewards else 0.0,
                "best_mean_reward": max(mean_rewards) if mean_rewards else 0.0,
            }
        )

    save_reward_mode_comparison(reward_series, str(out_root / "reward_comparison.png"))

    summary_csv = out_root / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "config", "final_mean_reward", "best_mean_reward"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Reward structure study complete. Outputs: {out_root}")


if __name__ == "__main__":
    main()
