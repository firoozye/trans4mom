import os
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data.processor import FeatureProcessor
from engine.loss import SharpeLoss
from models.architecture import EnsembleMomentumTransformer

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_data_v2(df, feat_cols, seq_len=252):
    # Mapping for Asset IDs (Grouped as in final_audit)
    asset_id_map = {
        "ES=F": 0, "NQ=F": 0, "RTY=F": 0, "YM=F": 0, "ER2=F": 0, "^N225": 0, "^FTSE": 0, "^GDAXI": 0, "^FCHI": 0, "^HSI": 0, "^AXJO": 0, "^SSMI": 0, "DX=F": 0,
        "ZT=F": 1, "ZF=F": 1, "ZN=F": 1, "ZB=F": 1, "FGBL=F": 1, "FGBM=F": 1, "FGBS=F": 1, "TWE=F": 1,
        "CL=F": 2, "NG=F": 2, "RB=F": 2, "HO=F": 2, "BZ=F": 2, "QM=F": 2, "GC=F": 2, "SI=F": 2, "HG=F": 2, "PL=F": 2, "PA=F": 2,
        "ZC=F": 3, "ZW=F": 3, "ZS=F": 3, "CC=F": 3, "KC=F": 3, "CT=F": 3, "SB=F": 3, "OJ=F": 3, "LE=F": 3, "GF=F": 3, "HE=F": 3,
        "6E=F": 4, "6J=F": 4, "6B=F": 4, "6A=F": 4, "6C=F": 4, "6S=F": 4, "6N=F": 4, "6M=F": 4, "6L=F": 4
    }
    
    all_x, all_y, all_s, all_aid, all_pm, all_pt, all_mon, all_day = [], [], [], [], [], [], [], []
    
    from numpy.lib.stride_tricks import sliding_window_view
    
    for sym, group in df.groupby('symbol'):
        group = group.sort_index().ffill().bfill()
        if len(group) <= seq_len: continue
        
        # Features + Range/Mean
        x_vals = group[feat_cols + ['range_60', 'mean_60']].values
        y_vals = group[['scaled_returns']].values
        s_vals = group[['spread']].values
        pm_vals = group[['prior_mom']].values
        pt_vals = group[['prior_trend']].values
        
        # Temporal info (1-based for month)
        months = group.index.month.values
        days = group.index.dayofweek.values
        
        aid = asset_id_map.get(sym, 0)
        
        x_win = sliding_window_view(x_vals, (seq_len, x_vals.shape[1])).squeeze(1)
        y_win = sliding_window_view(y_vals, (seq_len, 1)).squeeze(1)
        s_win = sliding_window_view(s_vals, (seq_len, 1)).squeeze(1)
        pm_win = sliding_window_view(pm_vals, (seq_len, 1)).squeeze(1)
        pt_win = sliding_window_view(pt_vals, (seq_len, 1)).squeeze(1)
        mon_win = sliding_window_view(months, (seq_len,))
        day_win = sliding_window_view(days, (seq_len,))
        
        all_x.append(x_win)
        all_y.append(y_win)
        all_s.append(s_win)
        all_pm.append(pm_win)
        all_pt.append(pt_win)
        all_mon.append(mon_win)
        all_day.append(day_win)
        all_aid.append(np.full((len(x_win),), aid))

    return (torch.tensor(np.concatenate(all_x), dtype=torch.float32),
            torch.tensor(np.concatenate(all_y), dtype=torch.float32),
            torch.tensor(np.concatenate(all_s), dtype=torch.float32),
            torch.tensor(np.concatenate(all_aid), dtype=torch.long),
            torch.tensor(np.concatenate(all_pm), dtype=torch.float32),
            torch.tensor(np.concatenate(all_pt), dtype=torch.float32),
            torch.tensor(np.concatenate(all_mon), dtype=torch.long),
            torch.tensor(np.concatenate(all_day), dtype=torch.long))

def main():
    parser = argparse.ArgumentParser(description="HPC Training: Upgraded Momentum Transformer")
    parser.add_argument("--config", type=str, default="config_1990.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config['data']
    feat_cfg = config['features']
    model_cfg = config['model']
    train_cfg = config['training']

    # 1. Load Data
    print(f"Loading data from {data_cfg['processed_path']}...")
    df = pd.read_parquet(data_cfg['processed_path'])
    train_df = df[df.index <= '2024-12-31']
    
    feat_cols = [f'ret_{k}' for k in feat_cfg['return_windows']] + \
                [f'macd_{S}_{L}' for S, L in feat_cfg['macd_pairs']]
    
    # 2. Prepare Sequences
    print("Preparing sequences with temporal features...")
    x, y, s, aid, pm, pt, mon, day = prepare_data_v2(train_df, feat_cols, seq_len=model_cfg['seq_len'])
    print(f"Prepared {x.shape[0]} sequences. Inputs: {x.shape[2]} vars.")

    # 3. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleMomentumTransformer(
        input_dim=1,
        num_vars=x.shape[2],
        hidden_dim=model_cfg['hidden_dim'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        dropout=model_cfg['dropout'],
        num_static_vars=5
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'])
    loss_fn = SharpeLoss(trans_cost=train_cfg['trans_cost'], annualization=train_cfg['annualization'])
    
    train_loader = DataLoader(TensorDataset(x, y, s, aid, pm, pt, mon, day), batch_size=train_cfg['batch_size'], shuffle=True)
    
    print(f"--- Starting Upgraded Training (Hidden: {model_cfg['hidden_dim']}, Seq: {model_cfg['seq_len']}) ---")
    for epoch in range(1, train_cfg['epochs'] + 1):
        model.train()
        total_loss = 0
        for bx, by, bs, baid, bpm, bpt, bmon, bday in train_loader:
            bx, by, bs, baid, bpm, bpt, bmon, bday = [b.to(device) for b in [bx, by, bs, baid, bpm, bpt, bmon, bday]]
            
            optimizer.zero_grad()
            out = model(bx, asset_id=baid, month=bmon, day=bday, prior_mom=bpm, prior_trend=bpt)
            loss = loss_fn(out, by, spreads=bs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {avg_loss:.4f}")
            torch.save(model.state_dict(), f"weights/upgraded_v2_epoch_{epoch}.pt")

    torch.save(model.state_dict(), "weights/upgraded_v2_final.pt")
    print("Training Complete.")

if __name__ == "__main__":
    main()
