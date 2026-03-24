"""
microstructure.py — Market microstructure feature computation.

Implements the core microstructure signals used in the 47D observation space:
- OFI (Order Flow Imbalance) — Cont, Kukanov & Stoikov (2014)
- VPIN (Volume-synchronized Probability of Informed Trading) — Easley et al. (2012)
- Kyle's Lambda (price impact per unit volume) — Kyle (1985)
- Amihud illiquidity ratio — Amihud (2002)
- Micro-price (volume-weighted mid) — Stoikov (2018)

All functions are stateless and operate on numpy arrays for speed.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence


def compute_ofi(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
    prev_bid_prices: np.ndarray,
    prev_bid_sizes: np.ndarray,
    prev_ask_prices: np.ndarray,
    prev_ask_sizes: np.ndarray,
    n_levels: int = 5,
) -> np.ndarray:
    """
    Order Flow Imbalance across n_levels of the LOB.

    OFI_k = ΔBid_size_k * I(bid_price_k >= prev_bid_k) - ΔAsk_size_k * I(ask_price_k <= prev_ask_k)

    Returns shape (n_levels,) — one OFI value per level.
    Reference: Cont, Kukanov & Stoikov (2014).
    """
    ofi = np.zeros(n_levels)
    for k in range(min(n_levels, len(bid_prices), len(ask_prices))):
        delta_bid = bid_sizes[k] if bid_prices[k] >= prev_bid_prices[k] else -bid_sizes[k]
        delta_ask = ask_sizes[k] if ask_prices[k] <= prev_ask_prices[k] else -ask_sizes[k]
        ofi[k] = delta_bid - delta_ask
    return ofi


def compute_vpin(
    buy_volume: np.ndarray,
    sell_volume: np.ndarray,
    bucket_size: float = 50.0,
    n_buckets: int = 50,
) -> float:
    """
    Volume-synchronized Probability of Informed Trading (VPIN).

    VPIN = (1/n) * Σ |V_buy_i - V_sell_i| / V_bucket

    Returns a float in [0, 1] — higher values indicate more informed flow.
    Reference: Easley, López de Prado & O'Hara (2012).
    """
    if len(buy_volume) == 0 or len(sell_volume) == 0:
        return 0.5
    total = buy_volume + sell_volume
    imbalance = np.abs(buy_volume - sell_volume)
    denom = np.where(total > 0, total, 1.0)
    return float(np.mean(imbalance / denom))


def compute_kyle_lambda(
    price_changes: np.ndarray,
    signed_volumes: np.ndarray,
) -> float:
    """
    Kyle's Lambda: price impact coefficient λ from OLS regression.

    Δp_t = λ * Q_t + ε_t

    Returns λ (price impact per unit signed volume).
    Reference: Kyle (1985).
    """
    if len(price_changes) < 10:
        return 0.0
    X = signed_volumes
    y = price_changes
    # OLS: λ = (X'X)^{-1} X'y  (scalar form for single regressor)
    XtX = float(np.dot(X, X))
    if abs(XtX) < 1e-12:
        return 0.0
    Xty = float(np.dot(X, y))
    return Xty / XtX


def compute_amihud(
    abs_returns: np.ndarray,
    volumes: np.ndarray,
) -> float:
    """
    Amihud illiquidity ratio: ILLIQ = (1/T) * Σ |r_t| / vol_t

    Higher values indicate less liquid markets (larger price impact per dollar traded).
    Reference: Amihud (2002).
    """
    if len(abs_returns) == 0:
        return 0.0
    denom = np.where(volumes > 0, volumes, 1.0)
    return float(np.mean(abs_returns / denom))


def compute_micro_price(
    best_bid: float,
    best_ask: float,
    bid_size: float,
    ask_size: float,
) -> float:
    """
    Volume-weighted mid price (micro-price).

    micro_price = bid * (ask_size / total) + ask * (bid_size / total)

    Weights the mid toward the side with less depth — the direction of likely next trade.
    Reference: Stoikov (2018).
    """
    total = bid_size + ask_size
    if total <= 0:
        return (best_bid + best_ask) / 2.0
    return best_bid * (ask_size / total) + best_ask * (bid_size / total)


def compute_relative_spread(best_bid: float, best_ask: float) -> float:
    """(ask - bid) / mid — normalized spread."""
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return 0.0
    return (best_ask - best_bid) / mid
