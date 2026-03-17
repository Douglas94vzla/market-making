import asyncio
import json
import logging
from typing import List, Tuple, Optional
import asyncpg
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS lob_snapshots (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    bids JSONB NOT NULL,
    asks JSONB NOT NULL,
    mid_price FLOAT8
);

CREATE TABLE IF NOT EXISTS trades (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price FLOAT8 NOT NULL,
    qty FLOAT8 NOT NULL,
    side CHAR(1) NOT NULL
);

SELECT create_hypertable('lob_snapshots', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
SELECT create_hypertable('trades', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
"""

class TimescaleDBWriter:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool: Optional[asyncpg.Pool] = None
        self._running = False
        self._batch_task: Optional[asyncio.Task] = None
        
        self.lob_queue: asyncio.Queue = asyncio.Queue()
        self.trade_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self):
        """Creates connection pool and ensures tables exist."""
        self.pool = await asyncpg.create_pool(
            dsn=self.db_url,
            min_size=2,
            max_size=10
        )
        async with self.pool.acquire() as conn:
            await conn.execute(CREATE_TABLES)
        logger.info("Connected to TimescaleDB and verified hypertable schema.")
        self._running = True
        self._batch_task = asyncio.create_task(self._batch_loop())

    async def disconnect(self):
        """Closes the connection pool and stops the batch loop."""
        self._running = False
        if self._batch_task:
            self._batch_task.cancel()
        if self.pool:
            await self.pool.close()

    async def queue_lob_snapshot(self, symbol: str, timestamp_ms: int, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], mid_price: float):
        """Add a LOB snapshot to the batch queue."""
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        await self.lob_queue.put((
            dt, symbol, json.dumps(bids), json.dumps(asks), mid_price
        ))

    async def queue_trade(self, symbol: str, timestamp_ms: int, price: float, qty: float, is_buyer_maker: bool):
        """Add a trade to the batch queue."""
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        side = 'S' if is_buyer_maker else 'B' # Logic: if buyer is maker, then aggressor (taker) was a sell
        await self.trade_queue.put((
            dt, symbol, price, qty, side
        ))

    async def _batch_loop(self):
        """Batch write every 500ms using asyncio.gather for both tables."""
        flush_interval = 0.5 # 500ms
        
        while self._running:
            await asyncio.sleep(flush_interval)
            
            lob_batch = []
            while not self.lob_queue.empty():
                lob_batch.append(self.lob_queue.get_nowait())
                
            trade_batch = []
            while not self.trade_queue.empty():
                trade_batch.append(self.trade_queue.get_nowait())

            if not lob_batch and not trade_batch:
                continue

            tasks = []
            if lob_batch:
                tasks.append(self._flush_lobs(lob_batch))
            if trade_batch:
                tasks.append(self._flush_trades(trade_batch))

            if tasks:
                try:
                    await asyncio.gather(*tasks)
                    # logger.debug(f"Flushed {len(lob_batch)} LOB snapshots and {len(trade_batch)} trades.")
                except Exception as e:
                    logger.error(f"Error during batch write to TimescaleDB: {e}")

    async def _flush_lobs(self, records: List[Tuple]):
        async with self.pool.acquire() as conn:
            await conn.copy_records_to_table(
                'lob_snapshots',
                records=records,
                columns=['time', 'symbol', 'bids', 'asks', 'mid_price']
            )

    async def _flush_trades(self, records: List[Tuple]):
        async with self.pool.acquire() as conn:
            await conn.copy_records_to_table(
                'trades',
                records=records,
                columns=['time', 'symbol', 'price', 'qty', 'side']
            )

# --- Design Decisions Note ---
# 1. Used connection pooling configured between 2 and 10 connections to balance latency and load.
# 2. Replaced `executemany` with `copy_records_to_table` since it's orders of magnitude faster in asyncpg for large inserts typical in market making.
# 3. Mapped Side logic (is_buyer_maker -> 'S' if True else 'B') since maker being a buyer means the taker executed a market sell.
