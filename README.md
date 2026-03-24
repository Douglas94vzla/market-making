# DRL Market Making System

> Adaptive market making agent for Binance using Deep Reinforcement Learning,
> classical microstructure theory, and production-grade safety systems.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/Douglas94vzla/market-making/actions/workflows/ci.yml/badge.svg)](https://github.com/Douglas94vzla/market-making/actions)

---

## What this is

A quantitative trading system that learns to provide liquidity in cryptocurrency
markets. It combines two approaches:

- **Avellaneda-Stoikov (2008)** — the analytical optimal market making solution,
  used as a theoretical prior and performance baseline.
- **PPO with Stable-Baselines3** — a DRL agent that learns deviations from the
  theoretical optimum in real market conditions where A-S assumptions break down.

The system is designed to run on a single VPS (no GPU required) and connects to
Binance via WebSocket for real-time order book data.

---

## Architecture

```
Binance WebSocket
      │
      ▼
┌─────────────────┐     ┌──────────────────────┐
│  LOB (in-memory)│────▶│  Feature Engine       │
│  SortedDict     │     │  OFI · VPIN · Kyle-λ  │
│  O(log N) upd.  │     │  Amihud · Momentum    │
└─────────────────┘     └──────────┬───────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │   HMM Regime Detector         │
                    │   4 states: trend± / mean-rev │
                    │   / high-vol                  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │   Gymnasium Environment       │
                    │   31D observation space       │
                    │   3D continuous action space  │
                    │   Sharpe-normalized reward    │
                    └──────────────┬───────────────┘
                                   │
               ┌───────────────────┼──────────────────┐
               ▼                   ▼                  ▼
    ┌──────────────────┐  ┌──────────────┐  ┌────────────────┐
    │ Avellaneda-Stoikov│  │  PPO Agent   │  │ Circuit Breaker│
    │ Baseline         │  │  SB3, CPU    │  │ 4 safety trips │
    │ Theoretical opt. │  │  8 envs par. │  │ Independent    │
    └──────────────────┘  └──────────────┘  └────────────────┘
               │                   │
               └─────────┬─────────┘
                         ▼
              ┌─────────────────────┐
              │  Event-Driven       │
              │  Backtester         │
              │  Adverse sel. model │
              │  Almgren-Chriss     │
              │  Sniping simulator  │
              └─────────────────────┘
```

---

## Key design decisions

### Reward function

The reward is normalized by realized volatility (Sharpe-per-step), with a clipped
denominator to prevent collapse during flash crashes:

```python
R(t) = (PnL_realized − λ·q² − κ·cost_taker + ρ·fill_bonus) / min(σ_30s, σ_p95)
```

Inventory penalty is asymmetric: short positions are penalized 50% more than long
(`λ_short = 1.5 × λ_long`) because crypto perpetuals carry positive funding rates.
At `|inventory| > 0.8`, the penalty escalates from quadratic to quartic.

### Regime-aware backtesting

Adverse selection probability varies with the HMM market regime — it's not fixed.
In high-volatility regimes (regime 3), adverse selection rises from 40% to ~54%.
This prevents the common backtesting mistake of overestimating PnL in volatile periods.

### Safety-first architecture

The `CircuitBreaker` class is completely independent from the agent. It monitors
four conditions and acts autonomously:

| Trigger | Condition | Action |
|---------|-----------|--------|
| Drawdown | > 5% from peak | Emergency liquidate |
| Flash crash | Vol > 3× historical | Cancel all, pause 30s |
| Stale position | Held > 5 min | Force flatten |
| Daily loss | > 2% capital | Shutdown today |

---

## Observation space (31 dimensions)

| Features | Dims | Description |
|----------|------|-------------|
| OFI multi-level | 5 | Order Flow Imbalance, levels 1–5 (Cont et al. 2014) |
| VPIN | 1 | Vol-synchronized Prob. Informed Trading (Easley et al. 2012) |
| Kyle lambda | 1 | Price impact per unit volume (Kyle 1985) |
| Relative spread | 1 | (ask − bid) / mid |
| Volatility | 3 | Log-return σ at 10s, 60s, 300s windows |
| Inventory | 1 | Normalized position [−1, 1] |
| Unrealized PnL | 1 | Marked to market, normalized |
| HMM regime | 4 | Probability vector over 4 market states |
| Micro-price | 1 | Volume-weighted mid (Stoikov 2018) |
| Momentum | 5 | Returns at 1s, 5s, 30s, 60s, 300s, ATR-normalized |
| Amihud ratio | 1 | \|Δp\| / volume (Amihud 2002) |
| Time encoding | 2 | sin/cos of time-of-day |
| Funding rate | 1 | Current perpetual funding rate |
| VWAP deviation | 3 | (mid − VWAP_k) / σ for k = 30s, 120s, 300s |

---

## Backtester realism

The event-driven backtester models four sources of PnL degradation that naive
backtests ignore:

- **Network latency**: log-normal(μ=2ms, σ=0.5ms) + exponential matching engine delay
- **Queue position**: orders fill after simulated prior queue volume is consumed
- **Adverse selection**: regime-aware probability (40–65%) of post-fill adverse move
- **Sniping**: 5–15% of limit orders canceled before reaching the matching engine
- **Price impact**: Almgren-Chriss model for orders > 0.1% of level-1 volume

In testing, naive backtests overestimate PnL by 2–5× versus this model.

---

## Quickstart

```bash
# 1. Start infrastructure
make up          # TimescaleDB + Grafana on Docker

# 2. Initialize database
make init-db     # Creates hypertables, indexes, compression policy

# 3. Collect market data (run for at least 24h before training)
make run         # Starts WebSocket pipeline → TimescaleDB

# 4. Train the agent
make train       # PPO, 2M timesteps, ~6h on CPU VPS

# 5. Run backtest
make backtest    # Compare A-S baseline vs trained PPO
```

---

## Performance targets

| Metric | Minimum | Ambitious |
|--------|---------|-----------|
| Sharpe ratio (annualized) | > 2.0 | > 3.5 |
| Max drawdown | < 15% | < 8% |
| Fill rate | > 60% | > 75% |
| Feature compute latency | < 5ms (Python) | < 50µs (Rust) |
| Adverse selection rate | < 30% | < 20% |

---

## Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| DRL | Stable-Baselines3 PPO | CPU-efficient, production-stable |
| Environment | Gymnasium 0.29 | Standard RL interface |
| Features | Python → Rust/PyO3 | Python for dev, Rust for production latency |
| Regime | hmmlearn HMM | 4-state Gaussian HMM, weekly retraining |
| Storage | TimescaleDB | 90% compression vs CSV, fast time-window queries |
| Monitoring | Prometheus + Grafana | Real-time PnL, latency p99, regime state |
| Transport | asyncpg + websockets | Zero-overhead async I/O |

---

## References

- Avellaneda & Stoikov (2008). *High-frequency trading in a limit order book*. Quantitative Finance.
- Easley, López de Prado & O'Hara (2012). *Flow Toxicity and Liquidity*. Review of Financial Studies.
- Cont, Kukanov & Stoikov (2014). *The Price Impact of Order Book Events*. J. Financial Econometrics.
- Kyle (1985). *Continuous Auctions and Insider Trading*. Econometrica.
- Almgren & Chriss (2001). *Optimal execution of portfolio transactions*. Journal of Risk.
- Amihud (2002). *Illiquidity and stock returns*. Journal of Financial Markets.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.

---

## Status

| Component | Status |
|-----------|--------|
| WebSocket pipeline | Complete |
| TimescaleDB schema | Complete |
| Gymnasium environment | Complete |
| A-S baseline agent | Complete |
| Feature engine (Python) | Complete |
| HMM regime detector | Complete |
| PPO agent (SB3) | Complete |
| Event-driven backtester | Complete |
| Circuit breaker | Complete |
| Monitoring (Grafana) | In progress |
| Feature engine (Rust) | Planned |
| Live trading (Testnet) | Planned |
