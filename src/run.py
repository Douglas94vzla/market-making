import asyncio
import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


async def main():
    parser = argparse.ArgumentParser(description="DRL Market Making System")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to trade/track")
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds (0=unlimited)")
    parser.add_argument(
        "--mode",
        type=str,
        default="pipeline",
        choices=["pipeline", "trade"],
        help="pipeline: data collection | trade: live A-S trading on Testnet",
    )
    args = parser.parse_args()

    db_url = os.getenv("DB_URL", "postgres://postgres:password@timescaledb:5432/market_making")
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET")

    if args.mode == "trade":
        # --- Live A-S Paper Trading on Testnet ---
        from trading.live_trader import LiveTrader

        testnet_key = os.getenv("TESTNET_API_KEY", api_key)
        testnet_secret = os.getenv("TESTNET_API_SECRET", api_secret)
        use_testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

        print(f"[TRADE MODE] Starting A-S Market Maker on {'Testnet' if use_testnet else 'LIVE'} for {args.symbol}")

        trader = LiveTrader(
            symbol=args.symbol,
            api_key=testnet_key,
            api_secret=testnet_secret,
            testnet=use_testnet,
        )

        # Run the trader directly — it blocks until stopped
        if args.duration > 0:
            try:
                await asyncio.wait_for(trader.run(), timeout=args.duration)
            except asyncio.TimeoutError:
                trader._running = False
        else:
            await trader.run()

    else:
        # --- Data Pipeline Mode (default) ---
        from data.pipeline import DataPipeline

        print(f"[PIPELINE MODE] Collecting data for {args.symbol}...")
        if args.duration > 0:
            print(f"Will stop after {args.duration} seconds.")
        else:
            print("Running continuously. Send SIGTERM or Ctrl+C to stop.")

        pipeline = DataPipeline(
            symbol=args.symbol,
            db_url=db_url,
            api_key=api_key,
            api_secret=api_secret,
        )

        # Run the pipeline directly — it blocks until stopped
        if args.duration > 0:
            try:
                await asyncio.wait_for(pipeline.run(), timeout=args.duration)
            except asyncio.TimeoutError:
                pipeline._running = False
        else:
            await pipeline.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
