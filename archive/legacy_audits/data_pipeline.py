import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
        return_windows=feat_cfg['return_windows'],
        macd_pairs=[tuple(p) for p in feat_cfg['macd_pairs']],
        vol_span=feat_cfg['vol_span']
    )

    all_assets = []

    # 4. Processing Loop
    symbols_map = data_cfg['symbols']
    print(f"--- Starting Pipeline for {len(symbols_map)} symbols (Hybrid Mode) ---")
    
    # Dates
    start_date = data_cfg['since']
    db_start = data_cfg['databento_start']
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')

    for db_symbol, yf_symbol in symbols_map.items():
        safe_sym = db_symbol.replace('.', '_')
        raw_file = os.path.join(raw_dir, f"{safe_sym}.parquet")
        
        # Determine if we need to fetch data
        needs_fetch = False
        df_raw = pd.DataFrame()

        if not os.path.exists(raw_file):
            needs_fetch = True
            print(f"  No existing data for {db_symbol}. Starting full download...")
        else:
            df_raw = pd.read_parquet(raw_file)
            last_date = df_raw.index.max()
            # Ensure both are timezone-aware (UTC) for comparison
            now_utc = datetime.now(last_date.tz) if last_date.tz else datetime.now()
            # If data is older than 2 days, update it
            if last_date < (now_utc - timedelta(days=2)):
                needs_fetch = True
                print(f"  Existing data for {db_symbol} ends at {last_date}. Updating...")
            else:
                print(f"  Data for {db_symbol} is up to date.")

        if needs_fetch:
            try:
                if df_raw.empty:
                    # Full download (Hybrid)
                    print(f"    Fetching yfinance ({start_date} to {db_start})...")
                    df_early = ingestor.fetch_yfinance(yf_symbol, start=start_date, end=db_start)
                    
                    print(f"    Fetching Databento ({db_start} to {end_date})...")
                    df_recent = ingestor.fetch_databento(
                        symbols=[db_symbol],
                        start=db_start,
                        end=end_date,
                        dataset=data_cfg['dataset'],
                        schema='ohlcv-1d',
                        stype_in='continuous'
                    )
                    df_recent.columns = [c.lower() for c in df_recent.columns]
                    df_raw = pd.concat([df_early, df_recent])
                else:
                    # Incremental update (Databento only for recent)
                    update_start = (df_raw.index.max() + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
                    if update_start < end_date:
                        print(f"    Fetching Databento update ({update_start} to {end_date})...")
                        df_new = ingestor.fetch_databento(
                            symbols=[db_symbol],
                            start=update_start,
                            end=end_date,
                            dataset=data_cfg['dataset'],
                            schema='ohlcv-1d',
                            stype_in='continuous'
                        )
                        df_new.columns = [c.lower() for c in df_new.columns]
                        df_raw = pd.concat([df_raw, df_new])
                
                df_raw = df_raw[~df_raw.index.duplicated(keep='last')].sort_index()
                df_raw.to_parquet(raw_file)
                print(f"    Raw data saved/updated ({len(df_raw)} rows).")
            except Exception as e:
                print(f"    Error updating {db_symbol}: {e}")
                if df_raw.empty: continue

        # Step B: Process Features
        print(f"    Processing features...")
        df_raw['symbol'] = db_symbol # Use DataBento symbol as canonical
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
