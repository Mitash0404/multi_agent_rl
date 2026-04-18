# Multi-Agent Reinforcement Learning for Emergent Cooperation and Competition

## Overview

This project explores how multiple reinforcement learning agents behave when placed in the same environment under different incentive structures. Rather than optimizing a single agent for a fixed task, the goal is to study how cooperation, competition, coordination, and selfish behavior emerge when several agents must share resources, avoid collisions, and optimize long-term reward.

The system is designed as a research-style multi-agent reinforcement learning framework that supports both cooperative and competitive reward settings. Agents are trained in a shared environment and evaluated not only on reward, but also on coordination quality, convergence stability, policy behavior, and emergent strategy formation.

This project is intended to go beyond a typical RL implementation. It is structured to answer questions such as:

- How does reward design influence emergent behavior?
- Under what conditions do agents learn to cooperate?
- When do selfish or adversarial behaviors appear?
- How stable is training across different multi-agent setups?
- Does communication improve coordination?

## Motivation

Most introductory reinforcement learning projects focus on a single agent solving a well-defined task. Real-world systems are often more complex:

- autonomous vehicles interact with other agents
- trading agents compete in markets
- warehouse robots coordinate over shared space
- bidding systems react to competitors
- resource allocation problems involve multiple actors with conflicting goals

In such settings, performance depends not only on environment dynamics, but also on the behavior of other learning agents. Multi-agent reinforcement learning introduces additional challenges:

- non-stationary training dynamics
- unstable convergence
- credit assignment across agents
- coordination failure
- emergent exploitation strategies

This project studies those issues in a controlled, explainable environment.

## Objectives

The main goals of the project are:

- Build a configurable multi-agent RL environment with shared state and interacting agents
- Train agents under different reward structures:
  - cooperative
  - competitive
  - mixed / partially aligned
- Measure not just final reward, but also:
  - coordination success
  - stability of learned policies
  - convergence behavior
  - emergent interaction patterns
- Compare learning dynamics under different algorithmic choices
- Optionally study whether communication channels improve performance

## Problem Setting

Multiple agents are placed in a shared discrete environment. Each agent can:

- observe the environment or part of it
- take actions at each timestep
- receive rewards based on its behavior and the behavior of others

The environment contains goals such as:

- collecting resources
- reaching target zones
- avoiding obstacles
- preventing collisions
- competing for limited rewards

Different reward structures lead to different dynamics:

### Cooperative Setting

All agents share a common reward. Success requires coordination.

Examples:

- agents must collect all resources efficiently
- one agent blocks while another traverses
- agents avoid interfering with each other

### Competitive Setting

Agents are individually rewarded and may benefit from harming the performance of others.

Examples:

- two agents compete for a scarce resource
- one agent blocks another from reaching a goal
- reward is zero-sum or partially adversarial

### Mixed / Semi-Cooperative Setting

Agents share some objectives but have partially conflicting incentives.

Examples:

- team reward plus individual bonus
- common goal with resource competition
- partial collaboration followed by individual optimization

## Environment Design

The project uses a grid-based multi-agent environment for interpretability and extensibility.

### Core Environment Elements

- 2D grid world
- multiple mobile agents
- static obstacles
- collectible resources or goal tiles
- optional dynamic hazards
- finite horizon episodes

## State Representation

The state can include:

- agent positions
- obstacle positions
- resource locations
- occupancy map
- agent-specific inventory or status
- nearby agents
- optional communication signals

Two observation modes can be supported:

### Full Observability

Each agent sees the full environment state

### Partial Observability

Each agent sees only a local window around itself

Partial observability is especially useful for studying communication and coordination.

## Action Space

Typical actions:

- move up
- move down
- move left
- move right
- stay
- optional interact / collect / signal actions

## Reward Design

Reward design is the most important experimental axis in this project.

### Cooperative Rewards

Examples:

- +10 when a shared task is completed
- +1 when a resource is collected by any team member
- -1 for collisions
- -0.1 per step to encourage efficiency

### Competitive Rewards

Examples:

- +10 for collecting a resource before another agent
- -10 if competitor collects it
- +1 for occupying strategic positions
- penalties for inefficient paths

### Mixed Rewards

Examples:

- shared team reward + individual bonus
- reward for total resources collected + private score multiplier
- collision penalties for all, but individual goal bonuses

### Reward Shaping Experiments

The project should explicitly study how changing reward design affects:

- convergence speed
- cooperation emergence
- selfish behavior
- policy stability
- exploitation patterns

## Algorithms

The framework should support multiple RL approaches for comparison.

### Phase 1: Simpler Baselines

- Tabular Q-learning in tiny environments
- Independent DQN agents

### Phase 2: Deep RL

- DQN
- PPO
- A2C or Actor-Critic variants

### Multi-Agent Training Variants

- Independent learners
- Shared policy learners
- Centralized training with decentralized execution
- Optional parameter sharing

The first implementation can begin with independent DQN agents because they are simpler to debug in a discrete environment.

## Research Questions

This project should be framed around concrete research-style questions.

### 1. How does reward structure affect emergent behavior?

Do cooperative rewards consistently produce coordination, or do agents still learn selfish shortcuts?

### 2. How stable is training in multi-agent environments?

Do policies converge, oscillate, or collapse under non-stationarity?

### 3. What kinds of emergent strategies appear?

Examples:

- blocking behavior
- role specialization
- leader/follower behavior
- selfish resource hoarding
- implicit turn-taking

### 4. Does communication improve coordination?

If agents are allowed to transmit a limited signal, does it improve task completion or convergence speed?

### 5. How does scaling the number of agents affect learning?

Do strategies break down as the environment becomes more crowded?

## Metrics

This project should evaluate far more than average reward.

### Core Metrics

#### Episode Reward

Average reward per episode across training and evaluation

#### Success Rate

Fraction of episodes where the team or agent completes the target objective

#### Convergence Speed

How quickly learning stabilizes

#### Collision Rate

How often agents interfere with each other

#### Resource Efficiency

How efficiently resources are collected relative to an optimal baseline

#### Coordination Score

A custom metric measuring cooperation quality

Possible coordination score ideas:

- percentage of episodes with no interference
- average overlap reduction
- shared task completion efficiency
- successful sequential role execution

#### Policy Stability

Variance in reward or action distributions after apparent convergence

#### Emergent Behavior Frequency

Count or score of behaviors such as:

- blocking
- role specialization
- monopolization
- cooperative sequencing

## Experimental Plan

### Experiment 1: Cooperative vs Competitive Rewards

Train the same environment under different reward functions and compare:

- reward
- success rate
- coordination score
- emergent behaviors

### Experiment 2: Independent vs Shared Policies

Evaluate whether shared policy learning stabilizes behavior or limits specialization

### Experiment 3: Full vs Partial Observability

Measure coordination drop under limited observation

### Experiment 4: With vs Without Communication

Allow a discrete communication token and compare team performance

### Experiment 5: Scaling Number of Agents

Run with 2, 3, and 5 agents to study complexity and instability

## Expected Findings

Examples of the kinds of insights this project should aim to produce:

- Cooperative reward structures may improve shared success but also encourage free-riding if poorly designed
- Competitive rewards may produce faster specialization but higher instability
- Communication may improve convergence under partial observability
- Independent learners may exhibit more diverse strategies but worse stability
- Policy collapse or oscillation may increase as agent count rises

These findings are what make the project feel research-level rather than implementation-only.

## Project Structure

```text
multi-agent-rl/
│
├── README.md
├── requirements.txt
├── configs/
│   ├── cooperative.yaml
│   ├── competitive.yaml
│   ├── mixed.yaml
│
├── env/
│   ├── gridworld_env.py
│   ├── observation.py
│   ├── rewards.py
│   ├── communication.py
│
├── agents/
│   ├── dqn_agent.py
│   ├── ppo_agent.py
│   ├── shared_policy.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   ├── replay_buffer.py
│
├── experiments/
│   ├── reward_structure_study.py
│   ├── observability_study.py
│   ├── communication_study.py
│
├── metrics/
│   ├── coordination.py
│   ├── stability.py
│   ├── plotting.py
│
├── visualizations/
│   ├── render_episode.py
│   ├── heatmaps.py
│
└── results/
    ├── plots/
    ├── logs/
    ├── checkpoints/