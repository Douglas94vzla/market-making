import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from sortedcontainers import SortedDict

from .ws_client import LOBSnapshot

logger = logging.getLogger(__name__)

@dataclass
class LimitOrderBook:
    symbol: str
    # Bids: Descending order (highest first) -> Negative keys hack for SortedDict
    _bids: SortedDict = field(default_factory=lambda: SortedDict(lambda x: -x))
    # Asks: Ascending order (lowest first) -> Normal keys
    _asks: SortedDict = field(default_factory=SortedDict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_update_id: int = 0

    def __post_init__(self):
        self._check_invariants()

    def _check_invariants(self):
        """Crossed book invariant check."""
        if __debug__:
            if self._bids and self._asks:
                best_bid = self._bids.peekitem(0)[0]
                best_ask = self._asks.peekitem(0)[0]
                assert best_bid < best_ask, f"Book is crossed! Symbol {self.symbol}, Best Bid: {best_bid}, Best Ask: {best_ask}"

    @property
    def best_bid(self) -> Optional[float]:
        try:
            return self._bids.peekitem(0)[0]
        except IndexError:
            return None

    @property
    def best_ask(self) -> Optional[float]:
        try:
            return self._asks.peekitem(0)[0]
        except IndexError:
            return None

    @property
    def mid_price(self) -> Optional[float]:
        b = self.best_bid
        a = self.best_ask
        if b is not None and a is not None:
            return (b + a) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        b = self.best_bid
        a = self.best_ask
        if b is not None and a is not None:
            return a - b
        return None

    async def apply_snapshot(self, snapshot: LOBSnapshot):
        """Replaces the entire LOB state."""
        async with self.lock:
            self._bids.clear()
            self._asks.clear()
            
            for price, qty in snapshot.bids:
                if qty > 0:
                    self._bids[price] = qty
            
            for price, qty in snapshot.asks:
                if qty > 0:
                    self._asks[price] = qty
            
            self._check_invariants()

    async def apply_delta(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        """Applies depth updates (deltas) to the existing LOB."""
        async with self.lock:
            for price, qty in bids:
                if qty == 0:
                    self._bids.pop(price, None)
                else:
                    self._bids[price] = qty
            
            for price, qty in asks:
                if qty == 0:
                    self._asks.pop(price, None)
                else:
                    self._asks[price] = qty
                    
            self._check_invariants()

    async def get_top_n(self, n: int) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Returns the top N levels for bids and asks."""
        async with self.lock:
            bids = list(self._bids.items())[:n]
            asks = list(self._asks.items())[:n]
            return bids, asks

# --- Design Decisions Note ---
# 1. Used negative values as keys for bids lambda in SortedDict to ensure descending order while iterating: `SortedDict(lambda x: -x)`.
# 2. Used `__debug__` for invariant assertion to ensure negligible performance overhead in production.
# 3. Using `asyncio.Lock` since this operates inside an asyncio context.
