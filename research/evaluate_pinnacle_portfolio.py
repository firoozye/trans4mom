#!/usr/bin/env python3
"""
Evaluate the trained Pinnacle Momentum Transformer models
across the 55-asset multi-asset portfolio.
Calculates the true annualized Portfolio Sharpe Ratio, Annualized Return, and Volatility.
"""

import os
import glob
import torch
import pandas as pd
import numpy as np
from models.architecture import EnsembleMomentumTransformer

def evaluate_checkpoint(checkpoint_path: str, data_path: str, seq_len: int = 252):
    print(f"\n=======================================================")
    print(f"Evaluating: {os.path.basename(checkpoint_path)}")
    print(f"=======================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_parquet(data_path)
    feat_cols = [f'ret_{k}' for k in [1, 21, 63, 126, 252]] + \
                [f'macd_{S}_{L}' for S, L in [[8, 24], [16, 48], [32, 96]]] + \
                ['cpd_sev_21', 'cpd_sev_126']
                
    model = EnsembleMomentumTransformer(
        input_dim=1,
        num_vars=len(feat_cols),
        hidden_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        output_dropout=0.1,
        num_static_vars=5
    ).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    
    # Run backtest per asset
    asset_pnls = []
    prior_pnls = []
    
    with torch.no_grad():
        for sym, group in df.groupby('symbol'):
            group = group.sort_index().ffill().bfill()
            if len(group) <= seq_len:
                continue
                
            x_vals = group[feat_cols].values.astype(np.float32)
            rets = group['scaled_returns'].values.astype(np.float32)
            pm = group['prior_mom'].values.astype(np.float32)
            mon = group.index.month.values.astype(np.int64)
            day = group.index.dayofweek.values.astype(np.int64)
            aid = int(group['asset_id'].iloc[0])
            
            # Predict positions using sliding windows in batches
            num_steps = len(x_vals)
            windows_x, windows_pm, windows_mon, windows_day = [], [], [], []
            
            for i in range(num_steps):
                if i < seq_len:
                    wx = np.zeros((seq_len, len(feat_cols)), dtype=np.float32)
                    wx[-(i+1):] = x_vals[:i+1]
                    wpm = np.zeros((seq_len, 1), dtype=np.float32)
                    wpm[-(i+1):] = pm[:i+1].reshape(-1, 1)
                    wmon = np.zeros(seq_len, dtype=np.int64)
                    wmon[-(i+1):] = mon[:i+1]
                    wday = np.zeros(seq_len, dtype=np.int64)
                    wday[-(i+1):] = day[:i+1]
                else:
                    wx = x_vals[i-seq_len+1 : i+1]
                    wpm = pm[i-seq_len+1 : i+1].reshape(-1, 1)
                    wmon = mon[i-seq_len+1 : i+1]
                    wday = day[i-seq_len+1 : i+1]
                windows_x.append(wx)
                windows_pm.append(wpm)
                windows_mon.append(wmon)
                windows_day.append(wday)
                
            batch_size = 512
            model_pos_list = []
            for b in range(0, len(windows_x), batch_size):
                bx = torch.from_numpy(np.stack(windows_x[b:b+batch_size])).unsqueeze(-1).to(device)
                bpm = torch.from_numpy(np.stack(windows_pm[b:b+batch_size])).to(device)
                bmon = torch.from_numpy(np.stack(windows_mon[b:b+batch_size])).to(device)
                bday = torch.from_numpy(np.stack(windows_day[b:b+batch_size])).to(device)
                baid = torch.tensor([aid], dtype=torch.long, device=device).expand(bx.size(0))
                
                pos_out = model(bx, asset_id=baid, month=bmon, day=bday, prior_mom=bpm)
                # Take the position at the final time step
                final_pos = pos_out[:, -1, 0].cpu().numpy()
                model_pos_list.append(final_pos)
                
            model_pos = np.concatenate(model_pos_list)
            prior_pos = np.tanh(pm)
            
            # PnL: pos[t-1] * rets[t] - transaction costs
            costs_model = np.abs(np.diff(model_pos, prepend=0)) * 0.0001
            pnl_model = (model_pos[:-1] * rets[1:]) - costs_model[1:]
            
            costs_prior = np.abs(np.diff(prior_pos, prepend=0)) * 0.0001
            pnl_prior = (prior_pos[:-1] * rets[1:]) - costs_prior[1:]
            
            asset_pnls.append(pd.Series(pnl_model, index=group.index[1:], name=sym))
            prior_pnls.append(pd.Series(pnl_prior, index=group.index[1:], name=sym))
            
    # Combine into portfolio
    port_model = pd.concat(asset_pnls, axis=1).mean(axis=1)
    port_prior = pd.concat(prior_pnls, axis=1).mean(axis=1)
    
    def print_period_stats(model_s, prior_s, start_date, end_date, name):
        m = model_s.loc[start_date:end_date]
        p = prior_s.loc[start_date:end_date]
        
        sr_m = (m.mean() / (m.std() + 1e-9)) * np.sqrt(252)
        sr_p = (p.mean() / (p.std() + 1e-9)) * np.sqrt(252)
        ret_m = m.mean() * 252 * 100
        ret_p = p.mean() * 252 * 100
        vol_m = m.std() * np.sqrt(252) * 100
        vol_p = p.std() * np.sqrt(252) * 100
        
        print(f"{name:20s} | Model SR: {sr_m:5.2f} (Ret: {ret_m:5.1f}%, Vol: {vol_m:4.1f}%) | Prior SR: {sr_p:5.2f} (Ret: {ret_p:5.1f}%, Vol: {vol_p:4.1f}%)")

    print("\n--- Portfolio Performance Attribution ---")
    print_period_stats(port_model, port_prior, '1990-01-01', '2022-12-31', 'Train (1990-2022)')
    print_period_stats(port_model, port_prior, '2023-01-01', '2024-12-31', 'Val   (2023-2024)')
    print_period_stats(port_model, port_prior, '2025-01-01', '2026-11-20', 'Test  (2025-2026)')
    print_period_stats(port_model, port_prior, '1990-01-01', '2026-11-20', 'Full  (1990-2026)')

if __name__ == "__main__":
    DATA_PATH = "/cs/research/fca/home0/firoozye/dev/trans4mom/data/processed/pinnacle_futures_1990_2026.parquet"
    weights = sorted(glob.glob("/cs/research/fca/home0/firoozye/dev/trans4mom/weights/pinnacle_best_seed_*.pt"))
    for w in weights:
        evaluate_checkpoint(w, DATA_PATH)
