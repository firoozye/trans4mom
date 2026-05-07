import os
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from engine.loss import SharpeLoss
from models.architecture import EnsembleMomentumTransformer

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, df, feat_cols, seq_len=252):
        self.seq_len = seq_len
        self.feat_cols = feat_cols
        
        # Mapping for Asset IDs (matching config_1990 symbols)
        asset_id_map = {
            "ES=F": 0, "NQ=F": 0, "RTY=F": 0, "YM=F": 0, "ER2=F": 0, "^N225": 0, "^FTSE": 0, "^GDAXI": 0, "^FCHI": 0, "^HSI": 0, "^AXJO": 0, "^SSMI": 0, "DX=F": 0,
            "ZT=F": 1, "ZF=F": 1, "ZN=F": 1, "ZB=F": 1, "FGBL=F": 1, "FGBM=F": 1, "FGBS=F": 1, "TWE=F": 1,
            "CL=F": 2, "NG=F": 2, "RB=F": 2, "HO=F": 2, "BZ=F": 2, "QM=F": 2, "GC=F": 2, "SI=F": 2, "HG=F": 2, "PL=F": 2, "PA=F": 2,
            "ZC=F": 3, "ZW=F": 3, "ZS=F": 3, "CC=F": 3, "KC=F": 3, "CT=F": 3, "SB=F": 3, "OJ=F": 3, "LE=F": 3, "GF=F": 3, "HE=F": 3,
            "6E=F": 4, "6J=F": 4, "6B=F": 4, "6A=F": 4, "6C=F": 4, "6S=F": 4, "6N=F": 4, "6M=F": 4, "6L=F": 4
        }
        
        self.data_groups = []
        self.indices = []
        
        for sym, group in df.groupby('symbol'):
            group = group.sort_index().ffill().bfill()
            if len(group) <= seq_len: continue
            
            # Pre-extract numpy arrays for fast access
            vals = {
                'x': group[feat_cols].values.astype(np.float32),
                'y': group['scaled_returns'].values.astype(np.float32),
                's': group['spread'].values.astype(np.float32),
                'pm': group['prior_mom'].values.astype(np.float32),
                'mon': group.index.month.values.astype(np.int64),
                'day': group.index.dayofweek.values.astype(np.int64),
                'aid': asset_id_map.get(sym, 0)
            }
            
            group_idx = len(self.data_groups)
            self.data_groups.append(vals)
            
            # Valid start indices for this group
            for i in range(len(group) - seq_len + 1):
                self.indices.append((group_idx, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        g_idx, start = self.indices[idx]
        g = self.data_groups[g_idx]
        end = start + self.seq_len
        
        return (
            torch.from_numpy(g['x'][start:end]),
            torch.from_numpy(g['y'][start:end]),
            torch.from_numpy(g['s'][start:end]),
            torch.tensor(g['aid'], dtype=torch.long),
            torch.from_numpy(g['pm'][start:end]).unsqueeze(-1),
            torch.from_numpy(g['mon'][start:end]),
            torch.from_numpy(g['day'][start:end])
        )

def main():
    parser = argparse.ArgumentParser(description="HPC Training: Strict Momentum Transformer Reproduction")
    parser.add_argument("--config", type=str, default="config_1990.yaml")
    parser.add_argument("--load_weights", type=str, default=None, help="Path to weights to resume from")
    parser.add_argument("--start_epoch", type=int, default=1, help="Epoch to start/resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config['data']
    feat_cfg = config['features']
    model_cfg = config['model']
    train_cfg = config['training']

    # 1. Load Data
    print(f"Loading data from {data_cfg['processed_path']}...")
    df = pd.read_parquet(data_cfg['processed_path'])
    # Train/Validation split: 1990-2024 for training
    train_df = df[df.index <= '2024-12-31']
    
    feat_cols = [f'ret_{k}' for k in feat_cfg['return_windows']] + \
                [f'macd_{S}_{L}' for S, L in feat_cfg['macd_pairs']]
    
    # 2. Prepare Dataset
    print("Initializing memory-efficient dataset...")
    dataset = TimeSeriesDataset(train_df, feat_cols, seq_len=model_cfg['seq_len'])
    loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=4)
    
    print(f"Starting Training: {len(dataset)} sequences | {model_cfg['seq_len']} days | {len(train_df['symbol'].unique())} assets")
    
    # 3. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_vars = len(feat_cols) # Exactly 8
    
    model = EnsembleMomentumTransformer(
        input_dim=1,
        num_vars=num_vars,
        hidden_dim=model_cfg['hidden_dim'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        dropout=model_cfg['dropout'],
        output_dropout=0.5, # Fixed at 0.5 as per reproduction plan
        num_static_vars=5
    ).to(device)
    
    if args.load_weights:
        print(f"Resuming from weights: {args.load_weights}")
        model.load_state_dict(torch.load(args.load_weights, map_location=device))
    
    # Optimizer with Weight Decay
    wd = float(train_cfg.get('weight_decay', 1e-5))
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'], weight_decay=wd)
    loss_fn = SharpeLoss(trans_cost=train_cfg['trans_cost'], annualization=train_cfg['annualization'])
    
    print(f"--- Starting Strict Reproduction (Hidden: {model_cfg['hidden_dim']}, Seq: {model_cfg['seq_len']}, Layers: {model_cfg['num_layers']}) ---")
    for epoch in range(args.start_epoch, train_cfg['epochs'] + 1):
        model.train()
        total_loss = 0
        for batch in loader:
            bx, by, bs, baid, bpm, bmon, bday = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            out = model(bx, asset_id=baid, month=bmon, day=bday, prior_mom=bpm)
            loss = loss_fn(out, by.unsqueeze(-1), spreads=bs.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {avg_loss:.4f}")
            torch.save(model.state_dict(), f"weights/strict_v2_epoch_{epoch}.pt")

    torch.save(model.state_dict(), "weights/strict_v2_final.pt")
    print("Training Complete.")

if __name__ == "__main__":
    main()
