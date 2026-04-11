import os
import yaml
import pandas as pd
from data.ingestion import DataIngestor
from data.processor import FeatureProcessor

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Load Config
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

    # 4. Download and Process each symbol
    for symbol in data_cfg['symbols']:
        print(f"--- Processing {symbol} ---")
        
        # Download
        df_raw = ingestor.fetch_ccxt_ohlcv(
            exchange_id=data_cfg['exchange'],
            symbol=symbol,
            timeframe=data_cfg['timeframe'],
            since=data_cfg['since']
        )
        
        # Save Raw
        raw_file = os.path.join(raw_dir, f"{symbol.replace('/', '_')}.parquet")
        df_raw.to_parquet(raw_file)
        print(f"  Saved raw data to {raw_file}")

        # Feature Engineering
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
