import pytest
import asyncio
from src.data.lob_state import LimitOrderBook
from src.data.ws_client import LOBSnapshot

@pytest.fixture
def lob():
    return LimitOrderBook("BTCUSDT")

@pytest.mark.asyncio
async def test_initial_snapshot(lob):
    snapshot = LOBSnapshot(
        symbol="BTCUSDT",
        timestamp=1000,
        bids=[(100.0, 1.5), (99.0, 2.0)],
        asks=[(101.0, 1.0), (102.0, 3.0)]
    )
    
    await lob.apply_snapshot(snapshot)
    
    assert lob.best_bid == 100.0
    assert lob.best_ask == 101.0
    assert lob.mid_price == 100.5
    assert lob.spread == 1.0

@pytest.mark.asyncio
async def test_apply_delta_updates(lob):
    # Setup initial
    snapshot = LOBSnapshot(
        symbol="BTCUSDT",
        timestamp=1000,
        bids=[(100.0, 1.5)],
        asks=[(101.0, 1.0)]
    )
    await lob.apply_snapshot(snapshot)
    
    # Delta adding a better bid and worse ask
    await lob.apply_delta(bids=[(100.5, 0.5)], asks=[(102.0, 1.0)])
    
    assert lob.best_bid == 100.5
    assert lob.best_ask == 101.0
    assert lob.mid_price == 100.75
    
    top_bids, _ = await lob.get_top_n(2)
    assert top_bids == [(100.5, 0.5), (100.0, 1.5)]

@pytest.mark.asyncio
async def test_apply_delta_deletions(lob):
    # Setup initial
    snapshot = LOBSnapshot(
        symbol="BTCUSDT",
        timestamp=1000,
        bids=[(100.0, 1.5), (99.0, 2.0)],
        asks=[(101.0, 1.0), (102.0, 3.0)]
    )
    await lob.apply_snapshot(snapshot)
    
    # Delta deleting best bid and best ask (qty = 0)
    await lob.apply_delta(bids=[(100.0, 0.0)], asks=[(101.0, 0.0)])
    
    assert lob.best_bid == 99.0
    assert lob.best_ask == 102.0
    assert lob.mid_price == 100.5
    assert lob.spread == 3.0

@pytest.mark.asyncio
async def test_crossed_book_invariant():
    lob = LimitOrderBook("BTCUSDT")
    snapshot = LOBSnapshot(
        symbol="BTCUSDT",
        timestamp=1000,
        # Ask is lower than bid -> book crossed
        bids=[(100.0, 1.0)],
        asks=[(99.0, 1.0)]
    )
    
    with pytest.raises(AssertionError, match="Book is crossed!"):
        await lob.apply_snapshot(snapshot)

# --- Design Decisions Note ---
# 1. Used standard pytest-asyncio to handle coroutines.
# 2. Verified invariant check inside DEBUG mode specifically catching `AssertionError`.
