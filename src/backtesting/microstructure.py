import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class OrderExecutionParameters:
    qty: float
    price: float
    side: str # 'buy' or 'sell'
    is_maker: bool = True

class LatencyModel:
    """
    Simulates physical and matching engine latencies.
    Network latency ~ log-normal (mu=2ms, sigma=0.5ms).
    Matching engine latency ~ exponential (scale=0.5ms).
    Total delay returned in milliseconds.
    """
    def __init__(self, network_mu_ms: float = 2.0, network_sigma_ms: float = 0.5, 
                 matching_scale_ms: float = 0.5):
        # Convert ms to log-normal parameters
        # For log-normal, if X = exp(Y) and Y ~ N(m, s^2)
        # E[X] = exp(m + s^2/2), Var[X] = (exp(s^2) - 1)*exp(2m + s^2)
        # We approximate for small variance:
        self.m = np.log(network_mu_ms) - 0.5 * np.log(1 + (network_sigma_ms / network_mu_ms) ** 2)
        self.s = np.sqrt(np.log(1 + (network_sigma_ms / network_mu_ms) ** 2))
        self.matching_scale = matching_scale_ms

    def get_latency_ms(self) -> float:
        net_lat = np.random.lognormal(mean=self.m, sigma=self.s)
        match_lat = np.random.exponential(scale=self.matching_scale)
        return net_lat + match_lat

class AdverseSelectionModel:
    """
    If a limit order is taken under 50ms of placement, simulate adverse price movement.
    # Fixed: original used a fixed 40% probability regardless of market regime.
    # In high-volatility regimes, adverse selection is significantly higher (up to ~54%).
    # Probability now varies with the HMM regime probability vector.
    Penalty: 0.5 bps.
    """
    def __init__(self, threshold_ms: float = 50.0, base_prob: float = 0.4, penalty_bps: float = 0.5, prob: float = None):
        self.threshold = threshold_ms
        self.base_prob = prob if prob is not None else base_prob  # `prob` kept for backward compatibility
        self.penalty = penalty_bps / 10000.0  # bps to decimal

    def _adverse_selection_prob(self, regime_probs: Optional[np.ndarray] = None, base: Optional[float] = None) -> float:
        """
        Regime-aware adverse selection probability.
        regime_probs: shape (4,) — [trend+, trend-, mean-rev, high-vol]
        High-vol regime (index 3) has highest adverse selection multiplier.
        # Fixed: fixed 40% overestimates PnL in calm regimes and underestimates risk in volatile ones
        """
        if regime_probs is None:
            return base if base is not None else self.base_prob
        b = base if base is not None else self.base_prob
        regime_weights = np.array([0.85, 0.90, 0.70, 1.35])  # multipliers per regime
        effective_multiplier = float(np.dot(regime_probs, regime_weights))
        return min(b * effective_multiplier, 0.85)  # cap at 85%

    def apply(
        self,
        exec_params: OrderExecutionParameters,
        time_in_book_ms: float,
        regime_probs: Optional[np.ndarray] = None,
    ) -> OrderExecutionParameters:
        prob = self._adverse_selection_prob(regime_probs)
        if time_in_book_ms < self.threshold and np.random.rand() < prob:
            # Price moves adversely (if buy, we paid higher; if sell, we sold lower)
            if exec_params.side == 'buy':
                exec_params.price *= (1 + self.penalty)
            else:
                exec_params.price *= (1 - self.penalty)
        return exec_params

class PriceImpactModel:
    """
    Almgren-Chriss Slippage: DeltaP = eta * sign(Q) * (Q/V_avg)^delta * sigma + eps
    For large market orders crossing the spread.
    """
    def __init__(self, eta: float = 0.1, delta: float = 0.6):
        self.eta = eta
        self.delta = delta

    def compute_slippage(self, order_qty: float, level_avg_vol: float, current_sigma: float) -> float:
        # Avoid division by zero
        v_avg = max(level_avg_vol, 1e-5)
        # Random noise term epsilon
        eps = np.random.normal(0, current_sigma * 0.1)
        
        impact = self.eta * ((order_qty / v_avg) ** self.delta) * current_sigma + eps
        return impact

class SnipingSimulator:
    """
    Simulated probability that a limit order is canceled by competitor
    before reaching the matching engine during high volatility.
    """
    def __init__(self, base_prob: float = 0.05, high_vol_prob: float = 0.15):
        self.base_prob = base_prob
        self.high_vol_prob = high_vol_prob

    def is_sniped(self, is_high_volatility: bool) -> bool:
        prob = self.high_vol_prob if is_high_volatility else self.base_prob
        return np.random.rand() < prob

# --- Design Decisions Note ---
# 1. Using mathematical mapping from mean/std space to log-normal parameters space for physical latency accuracy.
# 2. Extracting parameter classes to allow them to be mocked individually in isolated tests.
