import os
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from engine.trainer import Trainer, run_training_job
from data.processor import FeatureProcessor
from engine.loss import SharpeLoss

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_annualization_factor(timeframe: str):
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == 'h':
        return (24 / value) * 365
    elif unit == 'd':
        return 252.0
    return 252.0

def prepare_hpc_data_modulated(df, feat_cols, config, seq_len=64):
    """
    Pooled strategy with Multi-Prior Modulation.
    Extracts features + priors (Mom, Trend, Range, Mean).
    """
    all_x, all_y, all_s, all_aid, all_pm, all_pt = [], [], [], [], [], []
    
    for sym, group in df.groupby('symbol'):
        group = group.sort_index().ffill().bfill()
        # Features needed: feat_cols (MACD/Rets) + range_60 + mean_60
        # Priors needed: prior_mom, prior_trend
        cols = feat_cols + ['range_60', 'mean_60', 'prior_mom', 'prior_trend', 'scaled_returns', 'spread', 'asset_id']
        group = group.dropna(subset=cols)
        
        if len(group) <= seq_len:
            continue
            
        x_vals = group[feat_cols + ['range_60', 'mean_60']].values
        y_vals = group[['scaled_returns']].values
        s_vals = group[['spread']].values
        aid_vals = group['asset_id'].values
        pm_vals = group[['prior_mom']].values
        pt_vals = group[['prior_trend']].values
        
        for i in range(len(group) - seq_len):
            all_x.append(x_vals[i:i+seq_len])
            all_y.append(y_vals[i:i+seq_len])
            all_s.append(s_vals[i:i+seq_len])
            all_aid.append(aid_vals[i])
            all_pm.append(pm_vals[i:i+seq_len])
            all_pt.append(pt_vals[i:i+seq_len])
            
    return (torch.tensor(np.array(all_x), dtype=torch.float32), 
            torch.tensor(np.array(all_y), dtype=torch.float32),
            torch.tensor(np.array(all_s), dtype=torch.float32),
            torch.tensor(np.array(all_aid), dtype=torch.long),
            torch.tensor(np.array(all_pm), dtype=torch.float32),
            torch.tensor(np.array(all_pt), dtype=torch.float32))

def main():
    parser = argparse.ArgumentParser(description="HPC Training: Modulated Momentum Transformer")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config['data']
    feat_cfg = config['features']
    model_cfg = config['model']
    train_cfg = config['training']

    # 1. Process Features locally first (ensure new priors exist)
    print("Processing features with broader priors...")
    processor = FeatureProcessor(
        return_windows=feat_cfg['return_windows'],
        macd_pairs=[tuple(p) for p in feat_cfg['macd_pairs']]
    )
    
    # Load 1990 dataset
    df_raw = pd.read_parquet(data_cfg['processed_path'])
    df = processor.process_features(df_raw, config=config)
    
    # 2. Prepare Modulated Data
    feat_cols = [f'ret_{k}' for k in feat_cfg['return_windows']] + \
                [f'macd_{S}_{L}' for S, L in feat_cfg['macd_pairs']]
    
    seq_len = 64
    x, y, s, aid, pm, pt = prepare_hpc_data_modulated(df, feat_cols, config, seq_len=seq_len)
    print(f"Prepared {x.shape[0]} sequences. Inputs: {x.shape[2]} vars.")

    # 3. Setup Model and Trainer
    # We must update Trainer to handle the new forward(pm, pt) signature
    # For speed, I will use a custom loop here instead of the generic Trainer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from models.architecture import EnsembleMomentumTransformer
    
    model = EnsembleMomentumTransformer(
        input_dim=1,
        num_vars=x.shape[2],
        hidden_dim=model_cfg['hidden_dim'],
        num_heads=model_cfg['num_heads'],
        num_layers=2, # As per paper search grid
        dropout=0.3
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'])
    loss_fn = SharpeLoss(trans_cost=train_cfg['trans_cost'], annualization=252.0)
    
    train_loader = DataLoader(TensorDataset(x, y, s, aid, pm, pt), batch_size=train_cfg['batch_size'], shuffle=True)
    
    print(f"--- Starting Modulated Training (Hidden: {model_cfg['hidden_dim']}, Layers: 2) ---")
    for epoch in range(train_cfg['epochs']):
        model.train()
        total_loss = 0
        for bx, by, bs, baid, bpm, bpt in train_loader:
            bx, by, bs, baid, bpm, bpt = bx.to(device), by.to(device), bs.to(device), baid.to(device), bpm.to(device), bpt.to(device)
            
            optimizer.zero_grad()
            # Pass priors to the new architecture
            out = model(bx, asset_id=baid, prior_mom=bpm, prior_trend=bpt)
            loss = loss_fn(out, by, spreads=bs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}")
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"weights/modulated_baseline_epoch_{epoch}.pt")

    torch.save(model.state_dict(), "weights/modulated_baseline_final.pt")
    print("Training Complete.")

if __name__ == "__main__":
    main()
