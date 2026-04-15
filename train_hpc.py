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
        print(f"Data not found at {data_path}. Generating toy data for script verification...")
        num_assets = len(data_cfg['symbols'])
        x_toy = torch.randn(100, 64, len(feat_cfg['window_sizes']), 1) 
        y_toy = torch.randn(100, 64, num_assets)
    else:
        print(f"Loading real data from {data_path}...")
        df = pd.read_parquet(data_path)
        
        # Simple Pivot/Reshape for Transformer
        # Required format: (batch, time, num_vars, input_dim)
        # For this demo, we'll slice into sequences of length 64
        seq_len = 64
        num_assets = len(df['symbol'].unique())
        
        # Extract features (macd_*) and targets (scaled_returns)
        features = [f'macd_{w}' for w in feat_cfg['window_sizes']]
        
        all_x, all_y = [], []
        for sym, group in df.groupby('symbol'):
            group = group.sort_index()
            x_vals = group[features].values
            y_vals = group[['scaled_returns']].values
            
            # Create sliding windows
            for i in range(len(group) - seq_len):
                all_x.append(x_vals[i:i+seq_len])
                all_y.append(y_vals[i:i+seq_len])
        
        x_toy = torch.tensor(np.array(all_x), dtype=torch.float32).unsqueeze(-1)
        y_toy = torch.tensor(np.array(all_y), dtype=torch.float32) # (N, 64, 1)
        num_assets = 1 # Each sample is a single asset sequence
        print(f"Prepared {x_toy.shape[0]} sequences of length {seq_len}")



    # 4. Setup Hyperparameters for Trainer
    ann_factor = get_annualization_factor(data_cfg['timeframe'])
    print(f"Using annualization factor: {ann_factor:.2f} for timeframe: {data_cfg['timeframe']}")

    hparams = {
        'input_dim': model_cfg['input_dim'],
        'num_vars': len(feat_cfg['window_sizes']),
        'hidden_dim': model_cfg['hidden_dim'],
        'num_heads': model_cfg['num_heads'],
        'num_assets': num_assets if 'num_assets' in locals() else len(data_cfg['symbols']),
        'lr': train_cfg['lr'],
        'trans_cost': train_cfg['trans_cost'],
        'annualization': ann_factor,
        'batch_size': train_cfg['batch_size'],
        'epochs': args.epochs if args.epochs is not None else train_cfg['epochs']
    }

    # 5. Launch Training
    print(f"Launching HPC Training from config: {args.config}")
    run_training_job(x_toy, y_toy, hparams=hparams)

if __name__ == "__main__":
    main()
