from env.gridworld_env import EnvConfig, MultiAgentGridWorldEnv
from env.rewards import RewardConfig


def test_reset_and_step_contract():
    env = MultiAgentGridWorldEnv(
        EnvConfig(
            height=6,
            width=6,
            num_agents=2,
            num_obstacles=2,
            num_resources=2,
            max_steps=20,
            reward=RewardConfig(mode="cooperative"),
        )
    )

    obs, info = env.reset(seed=123)
    assert len(obs) == 2
    assert "resources_remaining" in info

    actions = {0: 0, 1: 4}
    next_obs, rewards, terminated, truncated, step_info = env.step(actions)

    assert len(next_obs) == 2
    assert len(rewards) == 2
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "collisions" in step_info


def test_partial_observation_dim():
    env = MultiAgentGridWorldEnv(
        EnvConfig(
            height=8,
            width=8,
            num_agents=3,
            num_obstacles=4,
            num_resources=3,
            observation_mode="partial",
            view_radius=2,
            communication_vocab_size=4,
            reward=RewardConfig(mode="mixed"),
        )
    )
    obs, _ = env.reset(seed=11)
    expected_dim = 4 * (2 * 2 + 1) * (2 * 2 + 1) + 4
    assert env.observation_dim == expected_dim
    assert obs[0].shape[0] == expected_dim
