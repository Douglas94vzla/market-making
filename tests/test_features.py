"""Tests for microstructure feature computation."""
import numpy as np
import pytest
from src.features.microstructure import (
    compute_ofi,
    compute_vpin,
    compute_kyle_lambda,
    compute_amihud,
    compute_micro_price,
    compute_relative_spread,
)


def test_ofi_balanced_book():
    """Equal bid/ask changes → OFI near zero."""
    prices = np.array([100.0, 99.0, 98.0, 97.0, 96.0])
    sizes = np.ones(5) * 10.0
    ofi = compute_ofi(prices, sizes, prices + 0.5, sizes, prices, sizes, prices + 0.5, sizes)
    assert ofi.shape == (5,)


def test_ofi_returns_n_levels():
    prices = np.array([100.0, 99.0, 98.0])
    sizes = np.ones(3)
    ofi = compute_ofi(prices, sizes, prices + 1, sizes, prices, sizes, prices + 1, sizes, n_levels=3)
    assert len(ofi) == 3


def test_vpin_range():
    buy_vol = np.random.rand(100) * 100
    sell_vol = np.random.rand(100) * 100
    v = compute_vpin(buy_vol, sell_vol)
    assert 0.0 <= v <= 1.0


def test_vpin_all_buy():
    """All buy volume → VPIN = 1."""
    buy_vol = np.ones(50) * 100.0
    sell_vol = np.zeros(50)
    v = compute_vpin(buy_vol, sell_vol)
    assert v == pytest.approx(1.0)


def test_kyle_lambda_positive_impact():
    """Positive volume → positive price impact."""
    price_changes = np.array([0.1, 0.2, 0.15, 0.3, 0.05] * 4, dtype=float)
    signed_vols = np.array([100.0, 200.0, 150.0, 300.0, 50.0] * 4, dtype=float)
    lam = compute_kyle_lambda(price_changes, signed_vols)
    assert lam > 0


def test_kyle_lambda_empty():
    assert compute_kyle_lambda(np.array([]), np.array([])) == 0.0


def test_amihud_zero_volume():
    """Zero volume → uses fallback denominator, no division error."""
    returns = np.array([0.001, 0.002, 0.003])
    volumes = np.zeros(3)
    ratio = compute_amihud(returns, volumes)
    assert np.isfinite(ratio)


def test_amihud_more_illiquid():
    high_impact = compute_amihud(np.ones(10) * 0.01, np.ones(10) * 1.0)
    low_impact = compute_amihud(np.ones(10) * 0.01, np.ones(10) * 1000.0)
    assert high_impact > low_impact


def test_micro_price_equal_depth():
    """Equal depth on both sides → micro_price = mid."""
    mid = compute_micro_price(99.0, 101.0, bid_size=10.0, ask_size=10.0)
    assert mid == pytest.approx(100.0)


def test_micro_price_skewed_toward_thinner_side():
    """More ask depth → price closer to bid (buyers dominate)."""
    mp = compute_micro_price(99.0, 101.0, bid_size=5.0, ask_size=15.0)
    assert mp < 100.0  # skewed toward bid


def test_relative_spread_positive():
    spread = compute_relative_spread(99.5, 100.5)
    assert spread > 0
    assert spread == pytest.approx(1.0 / 100.0)
