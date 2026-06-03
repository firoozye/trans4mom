import os
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from engine.loss import SharpeLoss
from models.architecture import EnsembleMomentumTransformer

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class TimeSeriesDataset(Dataset):
    def __init__(self, df, feat_cols, seq_len=252):
        self.seq_len = seq_len
        self.feat_cols = feat_cols
        
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
    parser = argparse.ArgumentParser(description="HPC Training: Wood et al. 2022 Exact Replication")
    parser.add_argument("--config", type=str, default="config_1990.yaml")
    parser.add_argument("--load_weights", type=str, default=None, help="Path to weights to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg, feat_cfg = config['data'], config['features']
    model_cfg, train_cfg = config['model'], config['training']

    # 1. Load Data
    print(f"Loading data from {data_cfg['processed_path']}...")
    df = pd.read_parquet(data_cfg['processed_path'])
    
    # 2. Chronological Split (Wood et al. Strategy)
    train_df = df[df.index <= train_cfg['train_split']]
    val_df = df[(df.index > train_cfg['train_split']) & (df.index <= train_cfg['val_split'])]
    
    feat_cols = [f'ret_{k}' for k in feat_cfg['return_windows']] + \
                [f'macd_{S}_{L}' for S, L in feat_cfg['macd_pairs']] + \
                ['cpd_sev_21', 'cpd_sev_126']
    
    # 3. Datasets
    print(f"Initializing datasets: Train <= {train_cfg['train_split']} | Val <= {train_cfg['val_split']}")
    train_ds = TimeSeriesDataset(train_df, feat_cols, seq_len=model_cfg['seq_len'])
    val_ds = TimeSeriesDataset(val_df, feat_cols, seq_len=model_cfg['seq_len'])
    
    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=4)
    
    # 4. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleMomentumTransformer(
        input_dim=1, num_vars=len(feat_cols), hidden_dim=model_cfg['hidden_dim'],
        num_heads=model_cfg['num_heads'], num_layers=model_cfg['num_layers'],
        dropout=model_cfg['dropout'], output_dropout=model_cfg['dropout'], num_static_vars=5
    ).to(device)
    
    if args.load_weights:
        model.load_state_dict(torch.load(args.load_weights, map_location=device))
    
    # 5. Optimizer: AdamW + High WD (Exact Replication)
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    loss_fn = SharpeLoss(trans_cost=train_cfg['trans_cost'], annualization=train_cfg['annualization'])
    
    # 6. Training Loop with Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = train_cfg['patience']
    
    print(f"--- Starting Exact Replication (Layers: {model_cfg['num_layers']}, Hidden: {model_cfg['hidden_dim']}) ---")
    
    for epoch in range(1, train_cfg['epochs'] + 1):
        # TRAIN
        model.train()
        train_loss = 0
        for batch in train_loader:
            bx, by, bs, baid, bpm, bmon, bday = [b.to(device) for b in batch]
            optimizer.zero_grad()
            out = model(bx, asset_id=baid, month=bmon, day=bday, prior_mom=bpm)
            loss = loss_fn(out, by.unsqueeze(-1), spreads=bs.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # VALIDATE
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                bx, by, bs, baid, bpm, bmon, bday = [b.to(device) for b in batch]
                out = model(bx, asset_id=baid, month=bmon, day=bday, prior_mom=bpm)
                loss = loss_fn(out, by.unsqueeze(-1), spreads=bs.unsqueeze(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f}")
        
        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "weights/reproduction_v3_best.pt")
            print("  --> New Best Model Saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"weights/reproduction_v3_epoch_{epoch}.pt")

    print("Training Complete.")

if __name__ == "__main__":
    main()
