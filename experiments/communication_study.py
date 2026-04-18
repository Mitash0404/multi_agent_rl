"""Evaluate performance with and without communication tokens."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from training.train import load_config, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Communication ablation study")
    parser.add_argument("--base-config", type=str, default="configs/cooperative.yaml")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=4)
    args = parser.parse_args()

    base = load_config(args.base_config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root) if args.output_root else Path("results") / "logs" / f"communication_study_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    no_comm = deepcopy(base)
    no_comm["experiment"]["name"] = f"{base['experiment']['name']}_no_comm"
    no_comm["env"]["observation_mode"] = "partial"
    no_comm["env"]["communication_vocab_size"] = 0

    with_comm = deepcopy(base)
    with_comm["experiment"]["name"] = f"{base['experiment']['name']}_with_comm"
    with_comm["env"]["observation_mode"] = "partial"
    with_comm["env"]["communication_vocab_size"] = int(args.vocab_size)

    run_training(no_comm, out_root / "no_communication")
    run_training(with_comm, out_root / "with_communication")

    print(f"Communication study complete. Outputs: {out_root}")


if __name__ == "__main__":
    main()
