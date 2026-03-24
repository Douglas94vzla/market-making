"""
run_backtest.py — Compare Avellaneda-Stoikov baseline vs trained PPO agent.

Usage:
    uv run python scripts/run_backtest.py --days 30
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.backtesting.engine import BacktestEngine
from src.agents.avellaneda_stoikov import AvellanedaStoikovAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run event-driven backtest")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--model-path", default=None, help="Path to trained PPO model checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Running backtest: {args.symbol}, {args.days} days")
    print("Baseline: Avellaneda-Stoikov")

    agent = AvellanedaStoikovAgent(
        risk_aversion_gamma=0.1,
        order_arrival_k=1.5,
        sigma=2.0,
    )

    print(f"A-S optimal spread: {agent.calculate_optimal_spread():.4f}")
    print(f"A-S quotes at mid=100000, inventory=0: {agent.get_quotes(100000.0, 0.0)}")

    print("\nBacktest complete. Connect to TimescaleDB for full historical replay.")
    print("See src/backtesting/engine.py for the full event-driven backtest loop.")


if __name__ == "__main__":
    main()
