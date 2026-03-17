import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time

from binance import AsyncClient, BinanceSocketManager

logger = logging.getLogger(__name__)

@dataclass
class LOBSnapshot:
    symbol: str
    timestamp: int
    bids: List[tuple[float, float]] # (price, quantity)
    asks: List[tuple[float, float]] # (price, quantity)

@dataclass
class TradeEvent:
    symbol: str
    timestamp: int
    price: float
    quantity: float
    is_buyer_maker: bool # Indicates side: if true, trade was a sell taker, buy maker

class BinanceWSClient:
    def __init__(self, symbol: str, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.symbol = symbol.upper()
        self.api_key = api_key
        self.api_secret = api_secret
        self.client: Optional[AsyncClient] = None
        self.bsm: Optional[BinanceSocketManager] = None
        self.event_queue: asyncio.Queue[LOBSnapshot | TradeEvent] = asyncio.Queue(maxsize=10_000)
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self):
        """Starts the WebSocket client with exponential backoff reconnection."""
        self._running = True
        self.client = await AsyncClient.create(self.api_key, self.api_secret)
        self.bsm = BinanceSocketManager(self.client)
        
        # Start connection loop
        self._tasks.append(asyncio.create_task(self._connection_loop()))

    async def stop(self):
        """Gracefully shutdown the client."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self.client:
            await self.client.close_connection()

    async def _connection_loop(self):
        base_delay = 1.0
        max_delay = 60.0
        attempts = 0

        while self._running:
            try:
                # We need multiple streams: @depth20@100ms and @aggTrade
                depth_stream = self.symbol.lower() + '@depth20@100ms'
                trade_stream = self.symbol.lower() + '@aggTrade'
                
                # Setup multiplex socket since BinanceSocketManager supports multiplex
                # In python-binance 1.0.19 it is usually bsm.multiplex_socket(['{}@depth20@100ms'.format(symbol), ...])
                # However, python-binance multiplex requires stream array
                streams = [depth_stream, trade_stream]
                
                logger.info(f"Connecting to Binance UX streams: {streams}")
                ts = self.bsm.multiplex_socket(streams)
                
                async with ts as socket:
                    logger.info("Successfully connected to Binance WebSockets.")
                    attempts = 0 # reset attempts on successful connection
                    while self._running:
                        msg = await socket.recv()
                        await self._process_message(msg)

            except Exception as e:
                logger.error(f"WebSocket connection error: {e}. Reconnecting...")
                
            if self._running:
                delay = min(max_delay, base_delay * (2 ** attempts))
                jitter = random.uniform(0, 0.1 * delay) # 10% jitter
                sleep_time = delay + jitter
                logger.info(f"Waiting {sleep_time:.2f}s before reconnecting (Attempt {attempts + 1})")
                await asyncio.sleep(sleep_time)
                attempts += 1

    async def _process_message(self, msg: Dict[str, Any]):
        try:
            # multiplex_socket wraps msgs in {'stream': ..., 'data': ...}
            if 'data' not in msg:
                return
            data = msg['data']
            stream = msg['stream']

            if '@depth20' in stream:
                snapshot = LOBSnapshot(
                    symbol=self.symbol,
                    timestamp=int(time.time() * 1000), # depth snapshot might not have standard event time, or we can use local
                    bids=[(float(p), float(q)) for p, q in data['bids']],
                    asks=[(float(p), float(q)) for p, q in data['asks']]
                )
                await self._push_to_queue(snapshot)
            elif '@aggTrade' in stream:
                trade = TradeEvent(
                    symbol=self.symbol,
                    timestamp=data['T'],
                    price=float(data['p']),
                    quantity=float(data['q']),
                    is_buyer_maker=data['m']
                )
                await self._push_to_queue(trade)
                
        except Exception as e:
            logger.error(f"Error processing message {msg}: {e}")

    async def _push_to_queue(self, event: LOBSnapshot | TradeEvent):
        try:
            self.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full. Dropping oldest event.")
            try:
                self.event_queue.get_nowait()
                self.event_queue.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass # Highly unlikely, but handles race conditions

# --- Design Decisions Note ---
# 1. Used python-binance 1.0.19 multiplex_socket to combine streams natively.
# 2. Used time.time() for LOBSnapshot timestamp because standard snapshot responses in Binance WS might lack high-precision timestamps compared to updates.
# 3. Implemented QueueFull handling by dropping the oldest event to prevent memory leaks and ensure we process the freshest data (LIFO-like drop).
