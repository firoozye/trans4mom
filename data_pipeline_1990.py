import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.ingestion import DataIngestor
from data.processor import FeatureProcessor

# Use the specific config for the 1990-Present run
CONFIG_PATH = "config_1990.yaml"

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
    config = load_config(CONFIG_PATH)
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
        return_windows=feat_cfg['return_windows'],
        macd_pairs=[tuple(p) for p in feat_cfg['macd_pairs']],
        vol_span=feat_cfg['vol_span']
    )

    all_assets = []

    # 4. Processing Loop
    symbols = data_cfg['symbols']
    print(f"--- Starting Pipeline for {len(symbols)} symbols (yfinance 1990+) ---")
    
    # Dates
    start_date = data_cfg['since']
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')

    for symbol in symbols.keys():
        safe_sym = symbol.replace('=', '_')
        raw_file = os.path.join(raw_dir, f"{safe_sym}.parquet")
        
        # Step A: Ensure Raw Data Exists
        if not os.path.exists(raw_file):
            print(f"  Downloading {symbol} from yfinance...")
            try:
                df_raw = ingestor.fetch_yfinance(
                    symbol=symbol,
                    start=start_date,
                    end=end_date
                )
                
                if df_raw.empty:
                    print(f"    Warning: No data found for {symbol}")
                    continue
                
                df_raw.to_parquet(raw_file)
                print(f"    Raw data saved ({len(df_raw)} rows, starts {df_raw.index.min().date()}).")
            except Exception as e:
                print(f"    Error fetching {symbol}: {e}")
                continue
        else:
            print(f"  Using existing raw data for {symbol} in {raw_dir}.")
            df_raw = pd.read_parquet(raw_file)

        # Step B: Process Features
        print(f"    Processing features...")
        # Note: df_raw already has 'symbol' column from ingestor.fetch_yfinance
        df_processed = processor.process_features(df_raw, config=config)
        all_assets.append(df_processed)

    # 5. Combine and Save Final Parquet
    if all_assets:
        final_df = pd.concat(all_assets)
        final_df.to_parquet(processed_path)
        print(f"\nSUCCESS: Combined processed data saved to {processed_path}")
        print(f"Total rows: {len(final_df)}")
        print(f"Symbols: {final_df['symbol'].unique()}")
    else:
        print("\nERROR: No assets processed.")

if __name__ == "__main__":
    main()
