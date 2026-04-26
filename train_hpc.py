import os
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from engine.trainer import run_training_job
from data.processor import FeatureProcessor

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_annualization_factor(timeframe: str) -> float:
    """Calculate the number of periods in a year for crypto (24/7)."""
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    
    if unit == 'h':
        return (24 / value) * 365
    elif unit == 'd':
        return (1 / value) * 365
    return 252.0 # Default fallback

def prepare_hpc_data(df, feat_cols, seq_len=64):
    """
    Reshapes data for multi-asset training, similar to train_cross_sectional.py
    """
    symbols = df['symbol'].unique()
    num_assets = len(symbols)
    
    # Pivot to align assets by timestamp
    pivoted = df.pivot_table(index=df.index, columns='symbol', values=feat_cols + ['scaled_returns', 'spread'])
    pivoted = pivoted.sort_index().ffill().bfill()
    
    # features: (time, assets, num_features)
    x_list = [pivoted[f].values for f in feat_cols]
    x_vals = np.stack(x_list, axis=-1) 
    y_vals = pivoted['scaled_returns'].values 
    s_vals = pivoted['spread'].values 
    
    all_x, all_y, all_s = [], [], []
    for i in range(len(pivoted) - seq_len):
        all_x.append(x_vals[i:i+seq_len])
        all_y.append(y_vals[i:i+seq_len])
        all_s.append(s_vals[i:i+seq_len])
        
    return (torch.tensor(np.array(all_x), dtype=torch.float32), 
            torch.tensor(np.array(all_y), dtype=torch.float32),
            torch.tensor(np.array(all_s), dtype=torch.float32))

def main():
    parser = argparse.ArgumentParser(description="HPC Training Script for Momentum Transformer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    # 1. Load Configuration
    config = load_config(args.config)
    
    # 2. Extract Hyperparameters
    data_cfg = config['data']
    feat_cfg = config['features']
    model_cfg = config['model']
    train_cfg = config['training']
    hpc_cfg = config['hpc']

    # 3. Load Data
    data_path = data_cfg['processed_path']
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Error.")
        return
    else:
        print(f"Loading real data from {data_path}...")
        df = pd.read_parquet(data_path)
        
        # Filter for training period (Pre-2025)
        train_end = '2024-12-31'
        df = df[df.index <= train_end]
        print(f"Training on data up to {train_end} ({len(df)} rows)")
        
        seq_len = 64
        symbols = df['symbol'].unique()
        num_assets = len(symbols)
        
        # Extract features (macd_*), targets (scaled_returns), and dynamic costs (spread)
        features = [f'ret_{k}' for k in feat_cfg['return_windows']] + \
                   [f'macd_{S}_{L}' for S, L in feat_cfg['macd_pairs']]
        
        x_toy, y_toy, s_toy = prepare_hpc_data(df, features, seq_len=seq_len)
        print(f"Prepared {x_toy.shape[0]} sequences of length {seq_len} for {num_assets} assets")

    # 4. Setup Hyperparameters for Trainer
    ann_factor = get_annualization_factor(data_cfg['timeframe'])
    print(f"Using annualization factor: {ann_factor:.2f} for timeframe: {data_cfg['timeframe']}")

    hparams = {
        'input_dim': len(features),
        'num_vars': num_assets, # Multi-asset cross-sectional dimension
        'hidden_dim': model_cfg['hidden_dim'],
        'num_heads': model_cfg['num_heads'],
        'num_assets': num_assets,
        'lr': train_cfg['lr'],
        'trans_cost': train_cfg['trans_cost'],
        'annualization': ann_factor,
        'batch_size': train_cfg['batch_size'],
        'epochs': args.epochs if args.epochs is not None else train_cfg['epochs']
    }

    # 5. Launch Training
    print(f"Launching HPC Training from config: {args.config}")
    run_training_job(x_toy, y_toy, train_spreads=s_toy if 's_toy' in locals() else None, hparams=hparams)

if __name__ == "__main__":
    main()
