import os
import matplotlib
matplotlib.use('Agg')
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.architecture import EnsembleMomentumTransformer
from data.processor import FeatureProcessor

def run_modulated_backtest_verbose(model, df, feat_cols, ann_factor, seq_len=64, base_cost=0.0001):
    symbols = df['symbol'].unique()
    asset_id_map = {
        "ES=F": 0, "NQ=F": 0, "RTY=F": 0, "YM=F": 0,
        "ZT=F": 1, "ZF=F": 1, "ZN=F": 1, "ZB=F": 1,
        "CL=F": 2, "NG=F": 2, "RB=F": 2, "HO=F": 2, "GC=F": 2, "SI=F": 2, "HG=F": 2, 
        "ZC=F": 2, "ZW=F": 2, "ZS=F": 2,
        "6E=F": 3, "6J=F": 3, "6B=F": 3, "6A=F": 3, "6C=F": 3, "6S=F": 3
    }
    
    asset_rets = {}
    asset_srs = {}
    
    with torch.no_grad():
        for sym in symbols:
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
            for b in range(0, len(bx), 512):
                out = model(bx[b:b+512], asset_id=baid[b:b+512], prior_mom=bpm[b:b+512], prior_trend=bpt[b:b+512])
                pos_list.append(out[:, -1, 0].cpu().numpy())
            
            pos = np.concatenate(pos_list)
            
            # Returns logic
            shifted_rets = np.roll(rets, -1)
            shifted_rets[-1] = 0
            pos_diff = np.abs(np.diff(pos, prepend=0))
            costs = pos_diff * base_cost
            net_rets = (pos * shifted_rets) - costs
            
            asset_rets[sym] = pd.Series(net_rets, index=group.index)
            sr = (net_rets.mean() / (net_rets.std() + 1e-9)) * np.sqrt(ann_factor)
            asset_srs[sym] = sr

    combined_df = pd.DataFrame(asset_rets).ffill().fillna(0)
    port_rets = combined_df.mean(axis=1)
    port_sr = (port_rets.mean() / (port_rets.std() + 1e-9)) * np.sqrt(ann_factor)
    
    return port_rets, asset_srs, port_sr

# --- Execution ---
print("--- Detailed Audit Start ---")
df_raw = pd.read_parquet('data/processed/futures_1990_present.parquet')
processor = FeatureProcessor(
    return_windows=[1, 21, 63, 126, 252],
    macd_pairs=[(8, 24), (16, 48), (32, 96)]
)
processed_groups = [processor.process_features(g) for _, g in df_raw.groupby('symbol')]
df = pd.concat(processed_groups)

model = EnsembleMomentumTransformer(input_dim=1, num_vars=10, hidden_dim=160, num_heads=4, num_layers=2)
model.load_state_dict(torch.load('weights/modulated_baseline_final.pt', map_location='cpu'))
model.eval()

port_rets, asset_srs, port_sr = run_modulated_backtest_verbose(model, df, 
    ['ret_1', 'ret_21', 'ret_63', 'ret_126', 'ret_252', 'macd_8_24', 'macd_16_48', 'macd_32_96'], 
    ann_factor=252)

print(f"\nOverall Portfolio Sharpe: {port_sr:.2f}")

print("\nIndividual Asset Sharpes (Full Period):")
for sym, sr in sorted(asset_srs.items(), key=lambda x: x[1], reverse=True):
    print(f"  {sym:6}: {sr:6.2f}")

# Annual Portfolio Performance
annual_sr = port_rets.groupby(port_rets.index.year).apply(lambda x: (x.mean() / (x.std() + 1e-9)) * np.sqrt(252))
print("\nAnnual Portfolio Sharpe:")
print(annual_sr)

# Key Dates Cumulative
cum_pnl = port_rets.cumsum()
print("\nCumulative PnL Milestones:")
for date in ['2000-01-01', '2010-01-01', '2020-01-01', '2024-12-31', '2026-04-30']:
    if date in cum_pnl.index:
        print(f"  {date}: {cum_pnl.loc[date]:.4f}")
    else:
        # Find nearest
        nearest = cum_pnl.index[cum_pnl.index <= date][-1]
        print(f"  {date} (approx {nearest.date()}): {cum_pnl.loc[nearest]:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(cum_pnl.index, cum_pnl.values, color='green')
plt.title(f"Portfolio Cumulative Net Return (Sharpe: {port_sr:.2f})")
plt.grid(True, alpha=0.3)
plt.savefig("research/final_audit_results.png")
print("\nAudit Complete.")
