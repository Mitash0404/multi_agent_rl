from env.rewards import RewardConfig, compute_multi_agent_rewards


def test_competitive_rewards_prioritize_collector():
    cfg = RewardConfig(mode="competitive", resource_reward=1.0, step_penalty=0.0, competitive_steal_penalty=0.25)
    rewards = compute_multi_agent_rewards(
        num_agents=3,
        config=cfg,
        collected_by=[1],
        collisions=[],
        episode_success=False,
    )
    assert rewards[1] > rewards[0]
    assert rewards[1] > rewards[2]


def test_cooperative_rewards_shared():
    cfg = RewardConfig(mode="cooperative", resource_reward=1.0, step_penalty=0.0)
    rewards = compute_multi_agent_rewards(
        num_agents=2,
        config=cfg,
        collected_by=[0],
        collisions=[],
        episode_success=False,
    )
    assert rewards[0] == rewards[1]
