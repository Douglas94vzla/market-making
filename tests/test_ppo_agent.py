import pytest

# Ray / RLlib may not be available on all Python versions; skip if absent.
ray = pytest.importorskip("ray", reason="Ray is not installed — skipping PPO config tests")
pytest.importorskip("ray.rllib", reason="RLlib is not installed — skipping PPO config tests")

from src.agents.ppo_agent import get_ppo_config  # noqa: E402 (import after skip guard)


def test_ppo_hyperparameters_spec():
    # We can fetch the config to test its structural setup mathematically
    # without needing ray fully initialized for the compilation due to Py 3.13 limit.
    config = get_ppo_config()

    # Asserting Section 3.3 Specs
    assert config.lr == 3e-4, "Learning rate must be 3e-4"
    assert config.clip_param == 0.15, "Clip parameter must be 0.15 for market making stability"
    assert config.train_batch_size == 4096, "Batch size must be 4096"
    assert config.gamma == 0.999, "Discount factor gamma must be 0.999"
    assert config.lambda_ == 0.95, "GAE lambda must be 0.95"
    assert config.grad_clip == 0.5, "Gradient clipping norm must be 0.5"


# --- Design Decisions Note ---
# 1. Skipped the agent.train() functional tensor loop because Ray is not universally available on Python 3.13.
# 2. Re-assured constraints purely via instantiation checking to ensure no spec regression.
