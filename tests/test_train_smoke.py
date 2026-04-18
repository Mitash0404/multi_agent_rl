import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _torch_available() -> bool:
    proc = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode == 0


@pytest.mark.slow
def test_train_smoke(tmp_path: Path):
    if not _torch_available():
        pytest.skip("PyTorch is unavailable or unstable in this runtime")

    config = {
        "experiment": {"name": "smoke", "seed": 7},
        "env": {
            "height": 6,
            "width": 6,
            "num_agents": 2,
            "num_obstacles": 2,
            "num_resources": 2,
            "max_steps": 20,
            "observation_mode": "full",
            "view_radius": 2,
            "communication_vocab_size": 0,
        },
        "reward": {
            "mode": "cooperative",
            "step_penalty": -0.01,
            "collision_penalty": -0.1,
            "resource_reward": 1.0,
            "goal_reward": 2.0,
            "team_reward_weight": 1.0,
            "individual_reward_weight": 0.0,
            "competitive_steal_penalty": 0.0,
        },
        "training": {
            "policy_mode": "independent",
            "num_episodes": 6,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay_episodes": 4,
            "checkpoint_interval": 3,
            "eval_interval": 3,
            "eval_episodes": 2,
            "log_interval": 2,
            "device": "cpu",
        },
        "dqn": {
            "gamma": 0.95,
            "lr": 0.001,
            "batch_size": 4,
            "buffer_size": 200,
            "min_buffer_size": 4,
            "target_update_interval": 10,
            "gradient_clip_norm": 5.0,
            "hidden_dim": 64,
        },
    }

    cfg_path = tmp_path / "smoke.yaml"
    out_dir = tmp_path / "smoke_run"

    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    cmd = [
        sys.executable,
        "-m",
        "training.train",
        "--config",
        str(cfg_path),
        "--output-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "episode_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "training_curves.png").exists()
