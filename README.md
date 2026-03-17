# 🤖 DRL Market Making System

A **production-grade High-Frequency Trading (HFT) framework** using Deep Reinforcement Learning for automated market making on cryptocurrency exchanges.

## 🏗️ Architecture

```
market_making/
├── src/
│   ├── data/
│   │   ├── pipeline.py          # Async data ingestion pipeline
│   │   ├── ws_client.py         # Binance WebSocket client (LOB + trades)
│   │   ├── lob_state.py         # In-memory Limit Order Book (SortedDict)
│   │   └── timescale_writer.py  # TimescaleDB persistence layer
│   ├── env/
│   │   └── market_making_env.py # Gymnasium RL environment (47D obs / 3D action)
│   ├── agents/
│   │   ├── avellaneda_stoikov.py # Optimal MM baseline (A-S 2008)
│   │   ├── ppo_agent.py          # PPO via Ray RLlib
│   │   ├── maml_meta.py          # MAML Meta-Learning controller
│   │   └── s4_model.py           # S4/Conv sequence spread agent
│   ├── backtesting/
│   │   ├── engine.py             # Event-driven backtest engine
│   │   └── microstructure.py     # Latency, adverse selection, price impact models
│   ├── features/
│   │   └── hmm_regime.py         # Hidden Markov Model regime detector (4 regimes)
│   └── run.py                    # Entry point
├── tests/                        # Full pytest suite (21 tests)
├── docker-compose.yml            # TimescaleDB
├── requirements.txt
└── pytest.ini
```

## 🧠 Key Components

| Component | Description |
|---|---|
| **MarketMakingEnv** | Gymnasium env with asymmetric inventory penalty + quartic scaling |
| **Avellaneda-Stoikov** | Analytical optimal bid/ask from reservation price theory |
| **HMM Regime Detector** | 4-state Gaussian HMM (trend±, mean-rev, high-vol) |
| **MAML Meta-Controller** | Fast adaptation via inner gradient loop (`create_graph=True`) |
| **S4 SpreadAgent** | Temporal conv network (256d, 4 layers) with twin Q-critics |
| **BacktestEngine** | Tick-level event-driven backtest with real microstructure |
| **DataPipeline** | Async Binance WS multiplexed streams → TimescaleDB |

## ⚙️ Setup

### Requirements
- Python 3.10+
- Docker (for TimescaleDB)

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/market_making.git
cd market_making

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Start TimescaleDB

```bash
docker-compose up -d
```

## 🚀 Running the Pipeline

```bash
# Set credentials (public streams work without keys)
export BINANCE_API_KEY="your_api_key"       # optional for public streams
export BINANCE_SECRET="your_secret_key"     # optional for public streams

# Run live pipeline (60 seconds on BTCUSDT)
python src/run.py --symbol BTCUSDT --duration 60

# Arguments
#   --symbol    Trading pair (default: BTCUSDT)
#   --duration  Runtime in seconds (default: 60)
```

**Windows PowerShell:**
```powershell
$env:BINANCE_API_KEY = "your_key"
$env:BINANCE_SECRET  = "your_secret"
python src\run.py --symbol BTCUSDT --duration 60
```

## 🧪 Running Tests

```bash
pytest tests/ -v
# Expected: 21 passed, 1 skipped (PPO/Ray — requires Ray installation)
```

## 📊 Reward Function

The environment implements the reward from Section 2.3 of the tech spec:

```
R = (realized_pnl - inv_penalty - κ·taker_cost + ρ·fill_bonus) / (σ₃₀s + ε)

inv_penalty = λ·q²        (long)
            = 1.5λ·q²     (short, asymmetric for funding constraints)
            × (|q|/0.8)²  (quartic scaling for |q| > 0.8)
```

## 🐋 Docker

```yaml
# docker-compose.yml — TimescaleDB
# postgres://postgres:password@localhost:5432/market_making
docker-compose up -d
```

## 🔑 Environment Variables

| Variable | Description | Default |
|---|---|---|
| `BINANCE_API_KEY` | Binance API key (optional for public data) | `None` |
| `BINANCE_SECRET` | Binance secret key | `None` |
| `DB_URL` | TimescaleDB connection string | `postgres://postgres:password@localhost:5432/market_making` |

## 📄 License

MIT
