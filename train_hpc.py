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
        return 252.0
    return 252.0 # Default fallback

def prepare_hpc_data(df, feat_cols, config, seq_len=64):
    """
    Reshapes data for 'Universal/Pooled' training (Wood et al., 2022).
    Each asset's sequence is treated as an independent sample in the batch.
    """
    # 1. Asset ID Mapping (if not already in df)
    if 'asset_id' not in df.columns:
        asset_id_map = {}
        if config and 'asset_map' in config:
            for idx, (category, syms) in enumerate(config['asset_map'].items()):
                for s in syms:
                    asset_id_map[s] = idx
        df['asset_id'] = df['symbol'].map(lambda s: asset_id_map.get(s, 4))

    # 2. Spread Fallback (if not in df)
    if 'spread' not in df.columns:
        df['spread'] = 0.0005 # 5bps default fallback
        
    all_x, all_y, all_s, all_aid = [], [], [], []
    
    # Process each asset independently
    for sym, group in df.groupby('symbol'):
        group = group.sort_index()
        group = group.dropna(subset=feat_cols + ['scaled_returns', 'spread'])
        
        if len(group) <= seq_len:
            continue
            
        x_vals = group[feat_cols].values
        y_vals = group[['scaled_returns']].values
        s_vals = group[['spread']].values
        aid_vals = group['asset_id'].values
        
        # Create sliding windows for this specific asset
        for i in range(len(group) - seq_len):
            all_x.append(x_vals[i:i+seq_len])
            all_y.append(y_vals[i:i+seq_len])
            all_s.append(s_vals[i:i+seq_len])
            # Asset ID is static for the sequence, just take the first value
            all_aid.append(aid_vals[i])
            
    # Stack all independent sequences together
    # Shapes:
    # X: (Total_Samples, Seq_Len, Num_Features)
    # Y, S: (Total_Samples, Seq_Len, 1)
    # AID: (Total_Samples,)
    return (torch.tensor(np.array(all_x), dtype=torch.float32), 
            torch.tensor(np.array(all_y), dtype=torch.float32),
            torch.tensor(np.array(all_s), dtype=torch.float32),
            torch.tensor(np.array(all_aid), dtype=torch.long))

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
        
        x_toy, y_toy, s_toy, aid_toy = prepare_hpc_data(df, features, config, seq_len=seq_len)
        print(f"Prepared {x_toy.shape[0]} sequences of length {seq_len} for {num_assets} assets")

    # 4. Setup Hyperparameters for Trainer
    ann_factor = get_annualization_factor(data_cfg['timeframe'])
    print(f"Using annualization factor: {ann_factor:.2f} for timeframe: {data_cfg['timeframe']}")

    hparams = {
        'input_dim': 1, # Each variable (MACD, return) is a 1D scalar
        'num_vars': len(features), # 8 MACD/return signals per asset
        'hidden_dim': model_cfg['hidden_dim'],
        'num_heads': model_cfg['num_heads'],
        'num_assets': 1, # Model outputs a single position sizing value
        'lr': train_cfg['lr'],
        'trans_cost': train_cfg['trans_cost'],
        'annualization': ann_factor,
        'batch_size': train_cfg['batch_size'],
        'epochs': args.epochs if args.epochs is not None else train_cfg['epochs']
    }

    # 5. Launch Training
    print(f"Launching HPC Training from config: {args.config}")
    
    # CRITICAL: Ensure tensors are on CPU before passing to trainer
    # to avoid driver-level OOM during model initialization.
    x_toy = x_toy.cpu()
    y_toy = y_toy.cpu()
    s_toy = s_toy.cpu()
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    run_training_job(x_toy, y_toy, train_spreads=s_toy, hparams=hparams)

if __name__ == "__main__":
    main()
