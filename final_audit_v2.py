import os
import matplotlib
matplotlib.use('Agg')
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.architecture import EnsembleMomentumTransformer
from data.processor import FeatureProcessor

def run_backtest(model, df, feat_cols, ann_factor, seq_len=64, base_cost=0.0001):
    symbols = df['symbol'].unique()
    asset_id_map = {
        "ES=F": 0, "NQ=F": 0, "RTY=F": 0, "YM=F": 0,
        "ZT=F": 1, "ZF=F": 1, "ZN=F": 1, "ZB=F": 1,
        "CL=F": 2, "NG=F": 2, "RB=F": 2, "HO=F": 2, "GC=F": 2, "SI=F": 2, "HG=F": 2, 
        "ZC=F": 2, "ZW=F": 2, "ZS=F": 2,
        "6E=F": 3, "6J=F": 3, "6B=F": 3, "6A=F": 3, "6C=F": 3, "6S=F": 3
    }
    
    portfolio_rets = []
    
    with torch.no_grad():
        for sym in symbols:
            print(f"Processing {sym}...")
            group = df[df['symbol'] == sym].sort_index()
            x_vals = group[feat_cols + ['range_60', 'mean_60']].values
            rets = group['scaled_returns'].values
            pm_vals = group['prior_mom'].values
            pt_vals = group['prior_trend'].values
            
            from numpy.lib.stride_tricks import sliding_window_view
            x_padded = np.pad(x_vals, ((seq_len-1, 0), (0, 0)), mode='constant')
            pm_padded = np.pad(pm_vals.reshape(-1, 1), ((seq_len-1, 0), (0, 0)), mode='constant')
            pt_padded = np.pad(pt_vals.reshape(-1, 1), ((seq_len-1, 0), (0, 0)), mode='constant')
            
            bx = torch.tensor(sliding_window_view(x_padded, (seq_len, x_vals.shape[1])).squeeze(1), dtype=torch.float32)
            bpm = torch.tensor(sliding_window_view(pm_padded, (seq_len, 1)).squeeze(1), dtype=torch.float32)
            bpt = torch.tensor(sliding_window_view(pt_padded, (seq_len, 1)).squeeze(1), dtype=torch.float32)
            baid = torch.full((bx.size(0),), asset_id_map.get(sym, 4), dtype=torch.long)
            
            pos_list = []
            for b in range(0, len(bx), 1024):
                out = model(bx[b:b+1024], asset_id=baid[b:b+1024], prior_mom=bpm[b:b+1024], prior_trend=bpt[b:b+1024])
                pos_list.append(out[:, -1, 0].cpu().numpy())
            
            pos = np.concatenate(pos_list)
            
            shifted_rets = np.roll(rets, -1)
            shifted_rets[-1] = 0
            pos_diff = np.abs(np.diff(pos, prepend=0))
            costs = pos_diff * base_cost
            net_rets = (pos * shifted_rets) - costs
            portfolio_rets.append(pd.Series(net_rets, index=group.index))

    combined = pd.DataFrame(portfolio_rets).T.ffill().fillna(0)
    port_rets = combined.mean(axis=1)
    sr = (port_rets.mean() / (port_rets.std() + 1e-9)) * np.sqrt(ann_factor)
    return port_rets, sr

# MAIN
print("STARTING AUDIT...")
df_raw = pd.read_parquet('data/processed/futures_1990_present.parquet')
processor = FeatureProcessor()
df = pd.concat([processor.process_features(g) for _, g in df_raw.groupby('symbol')])

model = EnsembleMomentumTransformer(input_dim=1, num_vars=10, hidden_dim=160, num_heads=4, num_layers=2)
model.load_state_dict(torch.load('weights/modulated_baseline_final.pt', map_location='cpu'))
model.eval()

rets, sr = run_backtest(model, df, ['ret_1', 'ret_21', 'ret_63', 'ret_126', 'ret_252', 'macd_8_24', 'macd_16_48', 'macd_32_96'], 252)
print(f"FINAL SHARPE: {sr:.2f}")

# OOS Audit
oos_rets = rets[rets.index >= '2025-01-01']
oos_sr = (oos_rets.mean() / (oos_rets.std() + 1e-9)) * np.sqrt(252)
print(f"OOS SHARPE: {oos_sr:.2f}")

# Yearly SR
yearly = rets.groupby(rets.index.year).apply(lambda x: (x.mean() / (x.std() + 1e-9)) * np.sqrt(252))
print(yearly)

# Plotting
plt.figure(figsize=(12, 8))
cum_pnl = rets.cumsum()
plt.plot(cum_pnl.index, cum_pnl.values, color='darkgreen', linewidth=2, label=f'Portfolio PnL (SR: {sr:.2f})')
plt.axvline(pd.to_datetime('2025-01-01'), color='red', linestyle='--', label='OOS Start')
plt.title(f"Residual Momentum Transformer: Cumulative Net Return\nOOS Sharpe: {oos_sr:.2f} | Full Sharpe: {sr:.2f}")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("research/final_audit_results.png")
print("\nCHART SAVED to research/final_audit_results.png")
