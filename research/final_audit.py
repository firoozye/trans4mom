import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.architecture import EnsembleMomentumTransformer
from data.processor import FeatureProcessor

def run_modulated_backtest(model, df, feat_cols, ann_factor, seq_len=64, base_cost=0.0001):
    """
    Run backtest for the Modulated Residual model across 24 assets.
    """
    symbols = df['symbol'].unique()
    portfolio_positions = {}
    
    # Paper asset categories
    asset_id_map = {
        "ES=F": 0, "NQ=F": 0, "RTY=F": 0, "YM=F": 0,
        "ZT=F": 1, "ZF=F": 1, "ZN=F": 1, "ZB=F": 1,
        "CL=F": 2, "NG=F": 2, "RB=F": 2, "HO=F": 2, "GC=F": 2, "SI=F": 2, "HG=F": 2, 
        "ZC=F": 2, "ZW=F": 2, "ZS=F": 2,
        "6E=F": 3, "6J=F": 3, "6B=F": 3, "6A=F": 3, "6C=F": 3, "6S=F": 3
    }

    with torch.no_grad():
        for sym in symbols:
            print(f"  Modulating {sym}...")
            group = df[df['symbol'] == sym].sort_index()
            group = group.ffill().bfill()
            
            # 1. Prepare Inputs
            x_vals = group[feat_cols + ['range_60', 'mean_60']].values
            rets = group['scaled_returns'].values
            pm_vals = group['prior_mom'].values
            pt_vals = group['prior_trend'].values
            
            num_steps = len(x_vals)
            windows, pm_list, pt_list = [], [], []
            for i in range(num_steps):
                if i < seq_len:
                    win = np.zeros((seq_len, len(feat_cols) + 2))
                    win[-(i+1):] = x_vals[:i+1]
                    pm_win = np.zeros((seq_len, 1))
                    pm_win[-(i+1):] = pm_vals[:i+1].reshape(-1, 1)
                    pt_win = np.zeros((seq_len, 1))
                    pt_win[-(i+1):] = pt_vals[:i+1].reshape(-1, 1)
                else:
                    win = x_vals[i-seq_len+1 : i+1]
                    pm_win = pm_vals[i-seq_len+1 : i+1].reshape(-1, 1)
                    pt_win = pt_vals[i-seq_len+1 : i+1].reshape(-1, 1)
                windows.append(win)
                pm_list.append(pm_win)
                pt_list.append(pt_win)
            
            x_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(next(model.parameters()).device)
            pm_tensor = torch.tensor(np.array(pm_list), dtype=torch.float32).to(x_tensor.device)
            pt_tensor = torch.tensor(np.array(pt_list), dtype=torch.float32).to(x_tensor.device)
            
            aid = asset_id_map.get(sym, 4)
            aid_tensor = torch.full((x_tensor.size(0),), aid, dtype=torch.long).to(x_tensor.device)
            
            # 2. Inference
            pos_list = []
            for b in range(0, len(x_tensor), 256):
                bx, baid, bpm, bpt = x_tensor[b:b+256], aid_tensor[b:b+256], pm_tensor[b:b+256], pt_tensor[b:b+256]
                out = model(bx, asset_id=baid, prior_mom=bpm, prior_trend=bpt)
                pos_list.append(out[:, -1, :].cpu().numpy())
            
            positions = np.concatenate(pos_list).flatten()
            
            # 3. Returns (t earns t+1)
            shifted_rets = np.roll(rets, -1)
            shifted_rets[-1] = 0
            pos_diff = np.abs(np.diff(positions, prepend=0))
            costs = pos_diff * base_cost
            
            net_rets = (positions * shifted_rets) - costs
            portfolio_positions[sym] = pd.Series(net_rets, index=group.index)

    # Aggregate
    combined_df = pd.DataFrame(portfolio_positions).ffill().fillna(0)
    portfolio_rets = combined_df.mean(axis=1)
    cum_pnl = portfolio_rets.cumsum()
    
    sr = (portfolio_rets.mean() / (portfolio_rets.std() + 1e-9)) * np.sqrt(ann_factor)
    
    oos_start = pd.to_datetime('2025-01-01')
    dates_no_tz = combined_df.index.tz_localize(None) if combined_df.index.tz is not None else combined_df.index
    oos_mask = dates_no_tz >= oos_start
    sr_oos = (portfolio_rets[oos_mask].mean() / (portfolio_rets[oos_mask].std() + 1e-9)) * np.sqrt(ann_factor) if oos_mask.any() else 0.0

    return {'dates': combined_df.index, 'cum_pnl': cum_pnl, 'sr': sr, 'sr_oos': sr_oos}

def main():
    # 1. Prepare Data with New Features
    data_path = 'data/processed/futures_1990_present.parquet'
    df_raw = pd.read_parquet(data_path)
    
    processor = FeatureProcessor(
        return_windows=[1, 21, 63, 126, 252],
        macd_pairs=[(8, 24), (16, 48), (32, 96)]
    )
    df = processor.process_features(df_raw) # Recalculate Range/Mean features
    
    feat_cols = ['ret_1', 'ret_21', 'ret_63', 'ret_126', 'ret_252', 'macd_8_24', 'macd_16_48', 'macd_32_96']
    
    # 2. Load Modulated Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleMomentumTransformer(
        input_dim=1, 
        num_vars=len(feat_cols) + 2, # signals + range + mean
        hidden_dim=160, 
        num_heads=4, 
        num_layers=2
    )
    
    model.load_state_dict(torch.load('weights/modulated_baseline_final.pt', map_location=device))
    model.to(device)
    model.eval()
    
    print(f"--- Running Final Paper-Scale Backtest ---")
    results = run_modulated_backtest(model, df, feat_cols, ann_factor=252, base_cost=0.0001)

    print(f"\n--- Final 24-Asset Residual Audit ---")
    print(f"  Full Period Net Sharpe (1990+): {results['sr']:.2f}")
    print(f"  OOS Net Sharpe (2025+):         {results['sr_oos']:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(results['dates'], results['cum_pnl'], label=f'Modulated Alpha (SR: {results["sr"]:.2f})', color='darkgreen')
    plt.axvline(pd.to_datetime('2025-01-01'), color='red', linestyle='--', label='OOS')
    plt.title("Final Wood et al. Replication: Residual Momentum Transformer")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("research/final_audit_results.png")
    print("\nDefinitive plot saved to research/final_audit_results.png")

if __name__ == "__main__":
    main()
