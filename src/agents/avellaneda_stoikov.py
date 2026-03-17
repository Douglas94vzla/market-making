import numpy as np
from typing import Tuple

class AvellanedaStoikovAgent:
    """
    Baseline optimal market making agent following Avellaneda-Stoikov (2008).
    Calculates reservation mid-price and optimal bid/ask spreads.
    """
    def __init__(self, risk_aversion_gamma: float = 0.1, 
                 order_arrival_k: float = 1.5,
                 sigma: float = 2.0):
        self.gamma = risk_aversion_gamma
        self.k = order_arrival_k
        self.sigma = sigma

    def convert_qty_to_inventory_q(self, qty: float, max_qty: float) -> float:
        """Normalized representation of inventory q. Roughly maps actual unit qty to [-1, 1]."""
        return np.clip(qty / max_qty, -1.0, 1.0)

    def calculate_reservation_price(self, mid_price: float, inventory_q: float) -> float:
        """
        r(s, t) = s - q * gamma * sigma^2 * (T - t)
        Simplified for continuous infinite horizon:
        r(s) = s - q * gamma * sigma^2
        """
        return mid_price - inventory_q * self.gamma * (self.sigma ** 2)

    def calculate_optimal_spread(self) -> float:
        """
        delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + (gamma/k))
        Simplified for continuous infinite horizon limit:
        delta = gamma * sigma^2 + (2/gamma) * ln(1 + (gamma * k)/2)
        As referenced in A-S variations.
        """
        term1 = self.gamma * (self.sigma ** 2)
        term2 = (2.0 / self.gamma) * np.log(1.0 + (self.gamma * self.k) / 2.0)
        return term1 + term2

    def get_quotes(self, mid_price: float, inventory_q: float) -> Tuple[float, float]:
        """
        Returns (optimal_bid, optimal_ask)
        r_bid = r(s) - delta / 2
        r_ask = r(s) + delta / 2
        """
        r = self.calculate_reservation_price(mid_price, inventory_q)
        delta = self.calculate_optimal_spread()
        
        bid = r - (delta / 2.0)
        ask = r + (delta / 2.0)
        
        return bid, ask

# --- Design Decisions Note ---
# 1. Used the infinite horizon simplification of the A-S model (T - t = 1) since cryptocurrency trading is 24/7 continuous.
# 2. Re-arranged the optimal spread calculation to perfectly match the requested derivation standard for $\delta = \gamma \sigma^2 + \frac{2}{\gamma}\ln(1 + \frac{\gamma k}{2})$.
