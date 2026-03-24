"""
circuit_breaker.py — Safety system for live trading.

Operates independently from the DRL agent. If the agent produces
degenerate actions or the market enters extreme conditions, this
module halts trading and liquidates positions.

Based on spec.txt Section 6.2.
"""
from dataclasses import dataclass
from enum import Enum, auto
import time


class Action(Enum):
    CONTINUE = auto()
    CANCEL_ALL_PAUSE_30S = auto()
    FORCE_FLATTEN = auto()
    EMERGENCY_LIQUIDATE = auto()
    SHUTDOWN_TODAY = auto()


@dataclass
class CircuitBreakerConfig:
    max_drawdown_pct: float = 0.05        # 5% from peak
    max_inventory_notional_usd: float = 10_000
    vol_multiplier_flash: float = 3.0     # 3x historical vol = flash crash
    max_position_age_s: float = 300.0     # 5 min max hold
    daily_loss_limit_pct: float = 0.02    # 2% daily loss cap


class CircuitBreaker:
    """
    Four independent safety triggers. Any one fires → system pauses or shuts down.
    This class has zero dependencies on the agent or environment.
    """

    def __init__(self, config: CircuitBreakerConfig = CircuitBreakerConfig()):
        self.config = config
        self._peak_capital: float = 0.0
        self._daily_start_capital: float = 0.0
        self._trigger_count: int = 0
        self._last_trigger: Action = Action.CONTINUE
        self._last_check_time: float = time.time()

    def check(
        self,
        current_capital: float,
        realized_pnl_today: float,
        vol_1min: float,
        vol_30d: float,
        max_position_age_s: float,
        inventory_notional_usd: float,
    ) -> Action:
        self._peak_capital = max(self._peak_capital, current_capital)

        # Trigger 1: Drawdown from peak
        if self._peak_capital > 0:
            drawdown = (self._peak_capital - current_capital) / self._peak_capital
            if drawdown > self.config.max_drawdown_pct:
                return self._trigger(Action.EMERGENCY_LIQUIDATE)

        # Trigger 2: Flash crash (volatility spike)  # CRITICAL
        if vol_30d > 0 and vol_1min > self.config.vol_multiplier_flash * vol_30d:
            return self._trigger(Action.CANCEL_ALL_PAUSE_30S)

        # Trigger 3: Stale inventory
        if max_position_age_s > self.config.max_position_age_s:
            return self._trigger(Action.FORCE_FLATTEN)

        # Trigger 4: Daily loss limit
        if self._daily_start_capital > 0:
            daily_loss_pct = abs(realized_pnl_today) / self._daily_start_capital
            if realized_pnl_today < 0 and daily_loss_pct > self.config.daily_loss_limit_pct:
                return self._trigger(Action.SHUTDOWN_TODAY)

        return Action.CONTINUE

    def _trigger(self, action: Action) -> Action:
        self._trigger_count += 1
        self._last_trigger = action
        return action

    def reset_daily(self, current_capital: float) -> None:
        """Call at start of each trading day. Also seeds the peak capital."""
        self._daily_start_capital = current_capital
        self._peak_capital = max(self._peak_capital, current_capital)

    @property
    def trigger_count(self) -> int:
        return self._trigger_count
