import asyncio
import os
import argparse
from data.pipeline import DataPipeline

async def main():
    parser = argparse.ArgumentParser(description="Run High-Frequency Trading Data Pipeline")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to track")
    parser.add_argument("--duration", type=int, default=60, help="Duration to run in seconds")
    args = parser.parse_args()

    db_url = os.getenv("DB_URL", "postgres://postgres:password@localhost:5432/market_making")
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET")

    pipeline = DataPipeline(
        symbol=args.symbol,
        db_url=db_url,
        api_key=api_key,
        api_secret=api_secret
    )
    
    # Start the pipeline as a background task
    pipeline_task = asyncio.create_task(pipeline.run())
    
    print(f"Running pipeline for {args.symbol} for {args.duration} seconds...")
    try:
        await asyncio.sleep(args.duration)
    except asyncio.CancelledError:
        pass
    finally:
        # Graceful shutdown
        pipeline._running = False
        await pipeline_task

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
