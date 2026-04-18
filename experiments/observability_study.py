"""Compare full vs partial observability in the same environment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from training.train import load_config, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Observability ablation study")
    parser.add_argument("--base-config", type=str, default="configs/cooperative.yaml")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--view-radius", type=int, default=2)
    args = parser.parse_args()

    base = load_config(args.base_config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root) if args.output_root else Path("results") / "logs" / f"observability_study_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    full_cfg = deepcopy(base)
    full_cfg["experiment"]["name"] = f"{base['experiment']['name']}_full"
    full_cfg["env"]["observation_mode"] = "full"

    partial_cfg = deepcopy(base)
    partial_cfg["experiment"]["name"] = f"{base['experiment']['name']}_partial"
    partial_cfg["env"]["observation_mode"] = "partial"
    partial_cfg["env"]["view_radius"] = int(args.view_radius)

    run_training(full_cfg, out_root / "full")
    run_training(partial_cfg, out_root / "partial")

    print(f"Observability study complete. Outputs: {out_root}")


if __name__ == "__main__":
    main()
