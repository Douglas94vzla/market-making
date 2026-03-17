import asyncio
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from src.data.ws_client import LOBSnapshot, TradeEvent
from src.data.lob_state import LimitOrderBook

@dataclass
class OrderExecutionParameters:
    qty: float
    price: float
    side: str # 'buy' or 'sell'
    is_maker: bool = True

class BacktestEngine:
    """
    Simulated Event-driven Backtest Engine integrating Order Management
    and Tick data feed.
    """
    def __init__(self, lob: LimitOrderBook):
        self.lob = lob
        
        # Order Management System State
        self.active_orders: Dict[str, dict] = {}
        
        # Metrics Tracking
        self.pnl_history: List[float] = []
        self.trades_history: List[dict] = []
        
        self.realized_pnl = 0.0
        self.current_inventory = 0.0
        self.avg_entry_price = 0.0

    def calculate_metrics(self) -> dict:
        if not self.pnl_history:
            return {"sharpe": 0.0, "max_drawdown": 0.0, "pnl": 0.0}
            
        pnl_arr = np.array(self.pnl_history)
        returns = np.diff(pnl_arr, prepend=0)
        
        sharpe = 0.0
        std_ret = np.std(returns)
        if std_ret > 0:
            sharpe = np.mean(returns) / std_ret * np.sqrt(31536000 * 10)  # rough annualization depending on tick frequency
            
        peak = np.maximum.accumulate(pnl_arr)
        drawdown = (peak - pnl_arr) / (peak + 1e-8)
        max_dd = np.max(drawdown)
        
        return {
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "total_pnl": self.realized_pnl
        }

    async def ingest_market_event(self, event: LOBSnapshot | TradeEvent):
        """
        Processes tick data and triggers limit order evaluation against 
        the market.
        """
        if isinstance(event, LOBSnapshot):
            await self.lob.apply_snapshot(event)
            await self._match_orders(timestamp_ms=event.timestamp)
        elif isinstance(event, TradeEvent):
            await self._match_orders(timestamp_ms=event.timestamp, trade=event)

    async def place_limit_order(self, order_id: str, side: str, price: float, qty: float, timestamp_ms: float):
        """Places a limit order in the OMS to be tracked against market ticks."""
        self.active_orders[order_id] = {
            "side": side,
            "price": price,
            "qty": qty,
            "placement_time": timestamp_ms,
            "filled": 0.0
        }

    async def cancel_order(self, order_id: str):
        if order_id in self.active_orders:
            del self.active_orders[order_id]

    async def _match_orders(self, timestamp_ms: float, trade: Optional[TradeEvent] = None):
        """
        Core logic of evaluating if maker orders are filled in the limit order book.
        A naive implementation for backtesting checks if market crossed order limit.
        """
        mid = self.lob.mid_price
        if mid is None:
            return
            
        # Collect finished orders
        finished = []
        for oid, order in self.active_orders.items():
            fill_qty = 0.0
            fill_price = order['price']
            
            # Simple assumption: market drops below buy limit -> fill buy
            if order['side'] == 'buy' and mid <= order['price']:
                fill_qty = order['qty'] - order['filled']
            # Simple assumption: market rises above sell limit -> fill sell
            elif order['side'] == 'sell' and mid >= order['price']:
                fill_qty = order['qty'] - order['filled']
                
            if fill_qty > 0:
                self._record_trade(
                    side=order['side'], 
                    qty=fill_qty, 
                    price=fill_price
                )
                order['filled'] += fill_qty
                if order['filled'] >= order['qty']:
                    finished.append(oid)
                    
        for oid in finished:
            del self.active_orders[oid]

        # Record PnL tick
        if len(self.pnl_history) == 0 or self.pnl_history[-1] != self.realized_pnl:
            self.pnl_history.append(self.realized_pnl)

    def _record_trade(self, side: str, qty: float, price: float):
        self.trades_history.append({"side": side, "qty": qty, "price": price})
        
        if side == 'buy':
            if self.current_inventory < 0: # Closing short
                self.realized_pnl += (self.avg_entry_price - price) * min(qty, abs(self.current_inventory))
            self.current_inventory += qty
            if self.current_inventory > 0 and self.avg_entry_price == 0:
                self.avg_entry_price = price
                
        elif side == 'sell':
            if self.current_inventory > 0: # Closing long
                self.realized_pnl += (price - self.avg_entry_price) * min(qty, self.current_inventory)
            self.current_inventory -= qty
            if self.current_inventory < 0 and self.avg_entry_price == 0:
                self.avg_entry_price = price

# --- Design Decisions Note ---
# 1. Implementing the core loop of evaluating limits against Market Ticks.
# 2. Simplification on the matching for unit testing (evaluating limits against rolling mid price). Full queue volume tracking from Phase 3 will inject logic here.
# 3. Generating robust metric aggregations correctly formatted for testing.
