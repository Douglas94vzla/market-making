import pytest
import numpy as np
import gymnasium as gym
from src.env.market_making_env import MarketMakingEnv

@pytest.fixture
def env():
    return MarketMakingEnv(
        max_inventory=1000.0,
        max_qty=10.0,
        sigma_window_60s=10.0
    )

def test_observation_space(env):
    obs, info = env.reset()
    assert obs.shape == (47,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)

def test_action_space(env):
    action = env.action_space.sample()
    assert action.shape == (3,)
    # bounded between 0 and 5/1 explicitly
    assert np.all(action >= 0.0)
    assert action[0] <= 5.0
    assert action[1] <= 5.0
    assert action[2] <= 1.0

def test_step_execution(env):
    env.reset()
    # (delta_bid, delta_ask, qty_pct)
    action = np.array([1.0, 1.0, 0.5], dtype=np.float32) 
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs.shape == (47,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

def test_reward_function_baseline():
    reward = MarketMakingEnv.compute_reward(
        inventory=0.0,      # flat
        realized_pnl=10.0,
        taker_cost=0.0,
        fill_occurred=True,
        sigma_30s=5.0,
        lambda_base=0.001,
        kappa=0.0007,
        rho=0.0001
    )
    expected_raw = 10.0 - 0.0 - 0.0 + 0.0001
    np.testing.assert_almost_equal(reward, expected_raw / (5.0 + 1e-8), decimal=4)

def test_reward_function_quartic_penalty():
    reward = MarketMakingEnv.compute_reward(
        inventory=0.9,      # > 0.8 triggering quartic
        realized_pnl=10.0,
        taker_cost=0.0,
        fill_occurred=True,
        sigma_30s=5.0,
        lambda_base=0.001,
        kappa=0.0007,
        rho=0.0001
    )
    
    # penalty formula logic: lambda * inv^2 * (inv/0.8)^2 for long
    inv_penalty = 0.001 * (0.9 ** 2) * ((0.9 / 0.8) ** 2)
    expected_raw = 10.0 - inv_penalty - 0.0 + 0.0001
    np.testing.assert_almost_equal(reward, expected_raw / (5.0 + 1e-8), decimal=4)

def test_random_policy_pnl(env):
    """
    Random policy should generate PnL roughly around 0 because the random walk
    is symmetric and no alpha/microstructure is yet present.
    """
    env.reset(seed=42)
    
    cumulative_pnl = 0.0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # Assuming PnL is tracked inside the environment mock logic
        # We can extract it via the tracking property. The spec asks PnL to be neutral.
    
    assert abs(env.realized_pnl) < 5000.0, f"PnL {env.realized_pnl} is too extreme for a random policy over 100 steps."
