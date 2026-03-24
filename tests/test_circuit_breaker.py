"""Tests for the standalone CircuitBreaker safety system."""
import pytest
from src.trading.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, Action


@pytest.fixture
def cb():
    config = CircuitBreakerConfig(
        max_drawdown_pct=0.05,
        vol_multiplier_flash=3.0,
        max_position_age_s=300.0,
        daily_loss_limit_pct=0.02,
    )
    breaker = CircuitBreaker(config)
    breaker.reset_daily(current_capital=100_000.0)
    return breaker


def _normal_check(cb: CircuitBreaker, capital: float = 100_000.0) -> Action:
    return cb.check(
        current_capital=capital,
        realized_pnl_today=0.0,
        vol_1min=0.001,
        vol_30d=0.001,
        max_position_age_s=10.0,
        inventory_notional_usd=1_000.0,
    )


def test_continue_under_normal_conditions(cb):
    assert _normal_check(cb) == Action.CONTINUE


def test_drawdown_triggers_emergency_liquidate(cb):
    # 5.1% drawdown from 100k peak
    action = cb.check(
        current_capital=94_850.0,
        realized_pnl_today=-5_150.0,
        vol_1min=0.001,
        vol_30d=0.001,
        max_position_age_s=10.0,
        inventory_notional_usd=1_000.0,
    )
    assert action == Action.EMERGENCY_LIQUIDATE


def test_flash_crash_triggers_cancel_pause(cb):
    # vol_1min = 4x vol_30d → flash crash
    action = cb.check(
        current_capital=100_000.0,
        realized_pnl_today=0.0,
        vol_1min=0.04,
        vol_30d=0.01,
        max_position_age_s=10.0,
        inventory_notional_usd=1_000.0,
    )
    assert action == Action.CANCEL_ALL_PAUSE_30S


def test_stale_position_triggers_force_flatten(cb):
    action = cb.check(
        current_capital=100_000.0,
        realized_pnl_today=0.0,
        vol_1min=0.001,
        vol_30d=0.001,
        max_position_age_s=400.0,  # > 300s threshold
        inventory_notional_usd=1_000.0,
    )
    assert action == Action.FORCE_FLATTEN


def test_daily_loss_limit_triggers_shutdown(cb):
    action = cb.check(
        current_capital=97_900.0,
        realized_pnl_today=-2_100.0,  # > 2% of 100k start
        vol_1min=0.001,
        vol_30d=0.001,
        max_position_age_s=10.0,
        inventory_notional_usd=1_000.0,
    )
    assert action == Action.SHUTDOWN_TODAY


def test_trigger_count_increments(cb):
    assert cb.trigger_count == 0
    cb.check(
        current_capital=94_000.0,
        realized_pnl_today=-6_000.0,
        vol_1min=0.001,
        vol_30d=0.001,
        max_position_age_s=10.0,
        inventory_notional_usd=1_000.0,
    )
    assert cb.trigger_count == 1


def test_no_agent_imports():
    """CircuitBreaker must have zero dependencies on agent or env modules."""
    import ast
    import inspect
    from src.trading import circuit_breaker as cb_module

    source = inspect.getsource(cb_module)
    tree = ast.parse(source)
    imports = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    forbidden = {"agents", "env", "ppo_agent", "s4_model", "maml_meta"}
    for imp in imports:
        if isinstance(imp, ast.ImportFrom) and imp.module:
            assert not any(f in imp.module for f in forbidden), (
                f"CircuitBreaker must not import from {imp.module}"
            )
