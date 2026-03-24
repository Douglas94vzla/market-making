import asyncio
import time
from collections import deque
from typing import Optional
import structlog

from binance import AsyncClient

logger = structlog.get_logger()


class OrderManager:
    """
    Manages the lifecycle of limit orders on Binance (or Testnet).
    Keeps track of active bid/ask orders and handles cancellations.
    """

    def __init__(self, client: AsyncClient, symbol: str, testnet: bool = True):
        self.client = client
        self.symbol = symbol.upper()
        self.testnet = testnet

        # Active order IDs
        self.active_bid_id: Optional[int] = None
        self.active_ask_id: Optional[int] = None

        # PnL tracking
        self.realized_pnl: float = 0.0
        self.inventory: float = 0.0  # net BTC position
        self.avg_entry_price: float = 0.0
        self.total_fills: int = 0

        # For Sharpe estimation
        self._pnl_snapshots: deque = deque(maxlen=200)
        self._last_snapshot_time: float = time.time()

    async def cancel_all_active(self):
        """Cancel both active bid and ask if they exist."""
        for oid in [self.active_bid_id, self.active_ask_id]:
            if oid is not None:
                try:
                    await self.client.cancel_order(symbol=self.symbol, orderId=oid)
                    logger.info("Order cancelled", order_id=oid)
                except Exception as e:
                    # Order may already be filled or cancelled — that is OK
                    logger.debug("Cancel order skipped", order_id=oid, reason=str(e))
        self.active_bid_id = None
        self.active_ask_id = None

    async def place_limit_order(self, side: str, price: float, quantity: float) -> Optional[int]:
        """Place a limit order. Returns order ID or None on failure."""
        try:
            price_str = f"{price:.2f}"
            qty_str = f"{quantity:.5f}"
            order = await self.client.create_order(
                symbol=self.symbol,
                side=side,
                type="LIMIT",
                timeInForce="GTC",  # Good Till Cancelled - works on Testnet and Production
                quantity=qty_str,
                price=price_str,
            )
            order_id = order["orderId"]
            logger.info("Order placed", side=side, price=price_str, qty=qty_str, order_id=order_id)
            return order_id
        except Exception as e:
            logger.error("Failed to place order", side=side, price=price, error=str(e))
            return None

    async def check_fills(self):
        """Check if active orders have been filled and update inventory/PnL."""
        for side, oid_attr in [("BUY", "active_bid_id"), ("SELL", "active_ask_id")]:
            oid = getattr(self, oid_attr)
            if oid is None:
                continue
            try:
                order = await self.client.get_order(symbol=self.symbol, orderId=oid)
                status = order["status"]

                if status == "FILLED":
                    filled_qty = float(order["executedQty"])
                    filled_price = float(order["cummulativeQuoteQty"]) / filled_qty

                    if side == "BUY":
                        # Update inventory and avg entry price
                        total_cost = self.avg_entry_price * self.inventory + filled_price * filled_qty
                        self.inventory += filled_qty
                        self.avg_entry_price = total_cost / self.inventory if self.inventory > 0 else 0.0
                    else:
                        # Realize PnL on sell
                        pnl = (filled_price - self.avg_entry_price) * filled_qty
                        self.realized_pnl += pnl
                        self.inventory -= filled_qty

                    self.total_fills += 1
                    logger.info(
                        "Order FILLED",
                        side=side,
                        price=f"{filled_price:.2f}",
                        qty=f"{filled_qty:.5f}",
                        realized_pnl=f"{self.realized_pnl:.4f}",
                        inventory=f"{self.inventory:.5f}",
                    )
                    setattr(self, oid_attr, None)

                elif status in ("CANCELED", "EXPIRED", "REJECTED"):
                    setattr(self, oid_attr, None)

            except Exception as e:
                logger.error("Error checking order fill", order_id=oid, error=str(e))

    def record_pnl_snapshot(self, mid_price: float):
        """Record a PnL snapshot for rolling Sharpe estimation."""
        now = time.time()
        unrealized = (mid_price - self.avg_entry_price) * self.inventory if self.inventory > 0 else 0.0
        total_pnl = self.realized_pnl + unrealized
        self._pnl_snapshots.append(total_pnl)
        self._last_snapshot_time = now

    def estimate_sharpe(self) -> float:
        """
        Rough annualized Sharpe from rolling PnL snapshots.
        Each snapshot is taken every ~5 minutes; assumes 105,120 snapshots/year.
        """
        if len(self._pnl_snapshots) < 10:
            return 0.0
        import numpy as np
        returns = np.diff(list(self._pnl_snapshots))
        if returns.std() < 1e-8:
            return 0.0
        # Snapshots every ~5min → 105,120 per year
        annualization = (105_120 / len(returns)) ** 0.5
        return float((returns.mean() / returns.std()) * annualization)
