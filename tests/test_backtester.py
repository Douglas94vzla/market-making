import pytest
import numpy as np
import asyncio
from src.backtesting.microstructure import LatencyModel, AdverseSelectionModel, PriceImpactModel, SnipingSimulator, OrderExecutionParameters
from src.backtesting.engine import BacktestEngine
from src.agents.avellaneda_stoikov import AvellanedaStoikovAgent
from src.data.lob_state import LimitOrderBook
from src.data.ws_client import LOBSnapshot, TradeEvent

def test_latency_model_distribution():
    # Test generation of logs
    model = LatencyModel(network_mu_ms=2.0, network_sigma_ms=0.5, matching_scale_ms=0.5)
    
    samples = [model.get_latency_ms() for _ in range(1000)]
    mean_lat = np.mean(samples)
    
    # Expected: mu_net + mu_match ~ 2.0 + 0.5 = 2.5
    # Variance adds up, so mean should be in the ballpark of [2.0, 3.0] due to statistical noise
    assert 2.0 < mean_lat < 3.0, f"Mean latency {mean_lat} significantly out of bounds."

def test_adverse_selection_pennying():
    np.random.seed(42) # making pseudo-determenistic
    model = AdverseSelectionModel(threshold_ms=50.0, prob=1.0, penalty_bps=0.5) # 100% prob for test
    params = OrderExecutionParameters(qty=1.0, price=10000.0, side='buy')
    
    # Time in book < 50ms (triggers Adverse)
    res = model.apply(params, time_in_book_ms=30.0)
    
    # We paid worse (higher)
    assert res.price > 10000.0
    assert np.isclose(res.price, 10000.0 * (1.0 + (0.5/10000.0)))
    
def test_price_impact_almgren_chriss():
    np.random.seed(42)
    model = PriceImpactModel(eta=0.1, delta=0.6)
    
    # Large order vs V_avg
    large_impact = model.compute_slippage(order_qty=10.0, level_avg_vol=1.0, current_sigma=5.0)
    
    # Small order vs V_avg
    small_impact = model.compute_slippage(order_qty=0.01, level_avg_vol=1.0, current_sigma=5.0)
    
    assert large_impact > small_impact

@pytest.mark.asyncio
async def test_avellaneda_stoikov_in_engine():
    """
    Test that the AS logic places orders through the engine optimally.
    """
    lob = LimitOrderBook("BTCUSDT")
    engine = BacktestEngine(lob)
    agent = AvellanedaStoikovAgent(risk_aversion_gamma=0.1, order_arrival_k=1.5, sigma=2.0)
    
    # Initial state
    snapshot = LOBSnapshot("BTCUSDT", 1000, [(99.0, 1.0)], [(101.0, 1.0)])
    await engine.ingest_market_event(snapshot)
    
    mid = lob.mid_price
    assert mid == 100.0
    
    # Flatten inventory calculate spreads
    q = agent.convert_qty_to_inventory_q(0.0, max_qty=10.0)
    optimal_bid, optimal_ask = agent.get_quotes(mid_price=mid, inventory_q=q)
    
    # Agent should quote symmetrically around mid when q=0
    assert np.isclose(mid - optimal_bid, optimal_ask - mid)
    
    # Quote via Engine
    await engine.place_limit_order("o1", side="buy", price=optimal_bid, qty=1.0, timestamp_ms=1010)
    await engine.place_limit_order("o2", side="sell", price=optimal_ask, qty=1.0, timestamp_ms=1010)
    
    assert len(engine.active_orders) == 2
    
    # Move market down strictly below optimal bid to force fill
    market_drop = LOBSnapshot("BTCUSDT", 1020, [(optimal_bid - 1.0, 1.0)], [(optimal_bid - 0.5, 1.0)])
    await engine.ingest_market_event(market_drop)
    
    assert engine.current_inventory == 1.0
    
    # Move market strictly up to force ask fill
    market_pump = LOBSnapshot("BTCUSDT", 1030, [(optimal_ask + 0.5, 1.0)], [(optimal_ask + 1.0, 1.0)])
    await engine.ingest_market_event(market_pump)
    
    assert engine.current_inventory == 0.0
    
    # Metrics check - Since we sold higher and bought lower on a cycle, PnL > 0
    metrics = engine.calculate_metrics()
    assert metrics['total_pnl'] > 0.0
    
# --- Design Decisions Note ---
# 1. Used deterministic `np.random.seed` in microscopic distribution checks to avoid flaky testing failures.
# 2. Used an event-sequence playback simulation to verify the full End-to-End integration between the A-S agent, Engine Limits, and Simulated LOB.
