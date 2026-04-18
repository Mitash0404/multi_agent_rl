# Multi-Agent Reinforcement Learning: Emergent Cooperation and Competition

A research-oriented multi-agent RL framework for studying how cooperation, competition, and mixed incentives shape learned behavior.

## What This Repo Includes

- Configurable grid-world environment with multiple agents, obstacles, resources, and collisions
- Reward modes: cooperative, competitive, and mixed
- Observation modes: full and partial observability
- Optional communication token channel
- Independent DQN baseline and shared-policy DQN variant
- Config-driven training and evaluation pipeline
- Experiment scripts for reward, observability, and communication ablations
- Coordination and stability metrics, plus plotting utilities
- Episode rendering and occupancy heatmaps

## Project Layout

```text
multi-agent-rl/
├── README.md
├── requirements.txt
├── configs/
├── env/
├── agents/
├── training/
├── experiments/
├── metrics/
├── visualizations/
├── tests/
└── results/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train a Baseline

```bash
python -m training.train --config configs/cooperative.yaml
```

Artifacts are written to `results/logs/<run_name_timestamp>/`:

- `episode_metrics.csv`
- `summary.json`
- `training_curves.png`
- `checkpoints/`

## Evaluate and Analyze

Reward structure study:

```bash
python -m experiments.reward_structure_study
```

Observability study:

```bash
python -m experiments.observability_study --base-config configs/cooperative.yaml
```

Communication study:

```bash
python -m experiments.communication_study --base-config configs/cooperative.yaml --vocab-size 4
```

## Visualize Behavior

Render an episode (random policy):

```bash
python -m visualizations.render_episode --config configs/cooperative.yaml --output results/plots/random_episode.gif
```

Render with trained checkpoints:

```bash
python -m visualizations.render_episode \
  --config configs/cooperative.yaml \
  --checkpoint-dir results/logs/<run>/checkpoints \
  --output results/plots/trained_episode.gif
```

Generate occupancy heatmaps:

```bash
python -m visualizations.heatmaps \
  --config configs/cooperative.yaml \
  --checkpoint-dir results/logs/<run>/checkpoints \
  --output results/plots/occupancy_heatmaps.png
```

## Metrics Collected

Per-episode logging includes:

- mean reward
- success indicator
- collision count
- resources collected
- coordination score
- training loss and TD error
- emergent behavior heuristics (blocking, monopolization, specialization)
- periodic evaluation reward/success/coordination

## Running Tests

```bash
pytest -q
```

## Notes on Extensions

The framework is structured to support:

- centralized training with decentralized execution
- PPO/A2C style extensions
- richer communication protocols
- larger scale agent-count studies
