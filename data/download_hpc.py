import argparse
import os
from data.ingestion import DataIngestor

def main():
    parser = argparse.ArgumentParser(description="Download crypto data for HPC training.")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="1h", help="e.g., 1m, 1h, 6h, 1d")
    parser.add_argument("--since", type=str, default="2020-01-01T00:00:00Z")
    parser.add_argument("--output", type=str, default="data/raw/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    ingestor = DataIngestor()
    df = ingestor.fetch_ccxt_ohlcv(
        exchange_id=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        since=args.since
    )
    
    file_name = f"{args.exchange}_{args.symbol.replace('/', '_')}_{args.timeframe}.parquet"
    save_path = os.path.join(args.output, file_name)
    df.to_parquet(save_path)
    print(f"Downloaded {len(df)} rows to {save_path}")

if __name__ == "__main__":
    main()
