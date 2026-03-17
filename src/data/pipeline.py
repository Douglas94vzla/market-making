import asyncio
import os
import signal
import sys
import structlog
import logging
from typing import Optional

from .ws_client import BinanceWSClient, LOBSnapshot, TradeEvent
from .lob_state import LimitOrderBook
from .timescale_writer import TimescaleDBWriter

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True)
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()

class DataPipeline:
    def __init__(self, symbol: str, db_url: str, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.symbol = symbol
        self.ws_client = BinanceWSClient(symbol, api_key, api_secret)
        self.lob = LimitOrderBook(symbol=symbol)
        self.db_writer = TimescaleDBWriter(db_url)
        self._running = False
        self._processed_ticks = 0

    async def _process_events(self):
        while self._running:
            try:
                event = await self.ws_client.event_queue.get()
                
                if isinstance(event, LOBSnapshot):
                    top_bids, top_asks = await self.lob.get_top_n(5)
                    # For optimization, we only snapshot when we first connect or on reconnects.
                    # Since Binance @depth20@100ms sends full depth 20 snapshots rather than strict deltas,
                    # we use apply_snapshot.
                    await self.lob.apply_snapshot(event)
                    
                    # Queue for persistence
                    mid_price = self.lob.mid_price
                    if mid_price is not None:
                        await self.db_writer.queue_lob_snapshot(
                            symbol=self.symbol,
                            timestamp_ms=event.timestamp,
                            bids=event.bids,
                            asks=event.asks,
                            mid_price=mid_price
                        )
                elif isinstance(event, TradeEvent):
                    await self.db_writer.queue_trade(
                        symbol=self.symbol,
                        timestamp_ms=event.timestamp,
                        price=event.price,
                        qty=event.quantity,
                        is_buyer_maker=event.is_buyer_maker
                    )
                
                self._processed_ticks += 1
                if self._processed_ticks % 10000 == 0:
                    logger.info("Pipeline Status", ticks=self._processed_ticks, mid_price=self.lob.mid_price)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Event processing error", error=str(e))

    async def run(self):
        logger.info("Starting Data Pipeline", symbol=self.symbol)
        self._running = True
        
        loop = asyncio.get_running_loop()
        def shutdown_signal(sig):
            logger.info("Received shutdown signal", signal=sig.name)
            self._running = False
            
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, shutdown_signal, sig)
            except NotImplementedError:
                # signal.add_signal_handler does not work on Windows out of the box in the same way
                pass

        try:
            await self.db_writer.connect()
            await self.ws_client.start()
            
            # Start consumer loop
            consumer_task = asyncio.create_task(self._process_events())
            
            # Keep running until gracefully stopped
            while self._running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error("Pipeline fatal error", error=str(e))
        finally:
            logger.info("Shutting down Pipeline...")
            self._running = False
            if 'consumer_task' in locals():
                consumer_task.cancel()
            await self.ws_client.stop()
            await self.db_writer.disconnect()
            logger.info("Shutdown cleanly.", ticks_processed=self._processed_ticks)

# --- Design Decisions Note ---
# 1. Added structlog for proper console formatting as requested (INFO in prod).
# 2. Replaced Windows-unsupported add_signal_handler properly by catching NotImplementedError, meaning fallback to Ctrl+C KeyboardInterrupt will occur cleanly.
# 3. Added minor tick logging for observing 1,000,000 metrics correctly.
