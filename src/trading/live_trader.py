import asyncio
import os
import time
from typing import Optional
import structlog

from binance import AsyncClient, BinanceSocketManager

from agents.avellaneda_stoikov import AvellanedaStoikovAgent
from trading.order_manager import OrderManager

logger = structlog.get_logger()

# Trading constants
ORDER_QUANTITY = 0.001        # BTC per order (small, safe for testnet)
MAX_INVENTORY_BTC = 0.01      # max 0.01 BTC position before skewing heavily
QUOTE_REFRESH_SECS = 10       # refresh quotes every 10 seconds
FILL_CHECK_SECS = 2           # check for fills every 2 seconds
STATS_LOG_SECS = 300          # print Sharpe/PnL summary every 5 minutes


class LiveTrader:
    """
    Connects Avellaneda-Stoikov quotes to live Binance Testnet order execution.
    1. Subscribes to real-time LOB WebSocket stream.
    2. Every QUOTE_REFRESH_SECS, cancels stale orders and places fresh bid/ask.
    3. Checks fills every FILL_CHECK_SECS and updates inventory/PnL.
    4. Logs performance metrics every STATS_LOG_SECS.
    """

    def __init__(self, symbol: str, api_key: str, api_secret: str, testnet: bool = True):
        self.symbol = symbol.upper()
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        self.agent = AvellanedaStoikovAgent(
            risk_aversion_gamma=0.1,
            order_arrival_k=1.5,
            sigma=2.0,
        )

        self._mid_price: Optional[float] = None
        self._running = False
        self.client: Optional[AsyncClient] = None
        self.order_manager: Optional[OrderManager] = None

    async def _fetch_mid_price(self):
        """Fetch current best bid/ask from REST to get initial mid price."""
        book = await self.client.get_order_book(symbol=self.symbol, limit=5)
        best_bid = float(book["bids"][0][0])
        best_ask = float(book["asks"][0][0])
        return (best_bid + best_ask) / 2.0

    async def _quote_loop(self):
        """Cancel stale orders and re-quote using A-S model every QUOTE_REFRESH_SECS."""
        while self._running:
            await asyncio.sleep(QUOTE_REFRESH_SECS)
            if self._mid_price is None:
                continue

            # Cancel stale quotes first
            await self.order_manager.cancel_all_active()

            # Compute normalized inventory [-1, 1]
            inventory_q = self.agent.convert_qty_to_inventory_q(
                self.order_manager.inventory, MAX_INVENTORY_BTC
            )

            # Get A-S optimal quotes
            bid_price, ask_price = self.agent.get_quotes(self._mid_price, inventory_q)

            # Place new orders (only if price is sane)
            if bid_price > 0 and ask_price > bid_price:
                self.order_manager.active_bid_id = await self.order_manager.place_limit_order(
                    "BUY", bid_price, ORDER_QUANTITY
                )
                self.order_manager.active_ask_id = await self.order_manager.place_limit_order(
                    "SELL", ask_price, ORDER_QUANTITY
                )
            else:
                logger.warning("Skipping quote: prices invalid", bid=bid_price, ask=ask_price)

    async def _fill_check_loop(self):
        """Periodically check if orders have been filled."""
        while self._running:
            await asyncio.sleep(FILL_CHECK_SECS)
            await self.order_manager.check_fills()

    async def _stats_loop(self):
        """Log Sharpe and PnL summary periodically."""
        while self._running:
            await asyncio.sleep(STATS_LOG_SECS)
            if self._mid_price:
                self.order_manager.record_pnl_snapshot(self._mid_price)
            sharpe = self.order_manager.estimate_sharpe()
            logger.info(
                "=== PERFORMANCE SUMMARY ===",
                realized_pnl_usdt=f"{self.order_manager.realized_pnl:.4f}",
                inventory_btc=f"{self.order_manager.inventory:.5f}",
                total_fills=self.order_manager.total_fills,
                sharpe_estimate=f"{sharpe:.3f}",
                mid_price=f"{self._mid_price:.2f}" if self._mid_price else "N/A",
            )

    async def _price_stream_loop(self):
        """Subscribe to Binance WebSocket to get real-time mid price."""
        bsm = BinanceSocketManager(self.client)
        depth_stream = self.symbol.lower() + "@depth5@100ms"
        logger.info("Subscribing to LOB stream", stream=depth_stream)

        while self._running:
            try:
                async with bsm.depth_socket(self.symbol, depth=5) as stream:
                    logger.info("Connected to Binance WebSocket price stream.")
                    while self._running:
                        msg = await stream.recv()
                        bids = msg.get("bids", [])
                        asks = msg.get("asks", [])
                        if bids and asks:
                            best_bid = float(bids[0][0])
                            best_ask = float(asks[0][0])
                            self._mid_price = (best_bid + best_ask) / 2.0
            except Exception as e:
                logger.error("WebSocket price stream error", error=str(e))
                await asyncio.sleep(5)

    async def run(self):
        """Main entry point: start all async loops."""
        import traceback
        try:
            await self._run_internal()
        except Exception as e:
            logger.error("FATAL ERROR in LiveTrader.run()", error=str(e), traceback=traceback.format_exc())
            raise

    async def _run_internal(self):
        """Actual implementation separated for clean error surfacing."""
        logger.info(
            "Starting Live A-S Trader",
            symbol=self.symbol,
            testnet=self.testnet,
            order_qty=ORDER_QUANTITY,
            refresh_secs=QUOTE_REFRESH_SECS,
        )

        self.client = await AsyncClient.create(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet,
        )
        self.order_manager = OrderManager(self.client, self.symbol, self.testnet)
        self._running = True

        # Get initial mid price via REST
        self._mid_price = await self._fetch_mid_price()
        logger.info("Initial mid price fetched", mid_price=f"{self._mid_price:.2f}")

        tasks = [
            asyncio.create_task(self._price_stream_loop()),
            asyncio.create_task(self._quote_loop()),
            asyncio.create_task(self._fill_check_loop()),
            asyncio.create_task(self._stats_loop()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            for t in tasks:
                t.cancel()
            logger.info("Cancelling all open orders before shutdown...")
            try:
                await self.order_manager.cancel_all_active()
            except Exception:
                pass
            await self.client.close_connection()
            logger.info(
                "Trader shut down cleanly.",
                total_fills=self.order_manager.total_fills,
                final_pnl=f"{self.order_manager.realized_pnl:.4f}",
            )
