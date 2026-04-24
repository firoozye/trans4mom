import os
import yaml
import pandas as pd
from datetime import datetime
from data.ingestion import DataIngestor
from data.processor import FeatureProcessor

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_env(env_path=".env"):
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
        print(f"Loaded environment variables from {env_path}")

def main():
    # 1. Load Config
    load_env()
    config = load_config("config.yaml")
    data_cfg = config['data']
    feat_cfg = config['features']

    # 2. Setup Paths
    raw_dir = data_cfg['raw_path']
    processed_path = data_cfg['processed_path']
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    # 3. Initialize Tools
    ingestor = DataIngestor()
    processor = FeatureProcessor(
        window_sizes=feat_cfg['window_sizes'],
        outlier_threshold=feat_cfg['outlier_threshold'],
        impute_nans=feat_cfg['impute_nans']
    )

    all_assets = []

    # 4. Download and Process
    if data_cfg['exchange'] == 'databento':
        print(f"--- Fetching Macro Basket from DataBento ({data_cfg['dataset']}) ---")
        from datetime import timedelta
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
        
        for symbol in data_cfg['symbols']:
            raw_file = os.path.join(raw_dir, f"{symbol.replace('.', '_')}.parquet")
            if os.path.exists(raw_file):
                print(f"  Skipping {symbol} (already exists).")
                df_raw = pd.read_parquet(raw_file)
                df_processed = processor.process_features(df_raw)
                df_processed['symbol'] = symbol
                all_assets.append(df_processed)
                continue

            print(f"  Downloading {symbol}...")
            try:
                df_raw = ingestor.fetch_databento(
                    symbols=[symbol],
                    start=data_cfg['since'],
                    end=end_date,
                    dataset=data_cfg['dataset'],
                    schema='ohlcv-1d',
                    stype_in='continuous'
                )
                if df_raw.empty: 
                    print(f"    Warning: No data for {symbol}")
                    continue
                
                # Save Raw
                raw_file = os.path.join(raw_dir, f"{symbol.replace('.', '_')}.parquet")
                df_raw.to_parquet(raw_file)
                
                # Process
                df_processed = processor.process_features(df_raw)
                df_processed['symbol'] = symbol
                all_assets.append(df_processed)
                print(f"    Success: {len(df_raw)} rows.")
            except Exception as e:
                print(f"    Error fetching {symbol}: {e}")
    else:
        for symbol in data_cfg['symbols']:
            # ... existing CCXT logic ...
            df_raw = ingestor.fetch_ccxt_ohlcv(
                exchange_id=data_cfg['exchange'],
                symbol=symbol,
                timeframe=data_cfg['timeframe'],
                since=data_cfg['since']
            )
            # ... process and save ...
            raw_file = os.path.join(raw_dir, f"{symbol.replace('/', '_')}.parquet")
            df_raw.to_parquet(raw_file)
            df_processed = processor.process_features(df_raw)
            df_processed['symbol'] = symbol
            all_assets.append(df_processed)

    # 5. Combine and Save Final Parquet
    final_df = pd.concat(all_assets)
    final_df.to_parquet(processed_path)
    print(f"\nSUCCESS: Combined processed data saved to {processed_path}")
    print(f"Total rows: {len(final_df)}")
    print(f"Symbols: {final_df['symbol'].unique()}")

if __name__ == "__main__":
    main()
