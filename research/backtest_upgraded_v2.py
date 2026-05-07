import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.architecture import EnsembleMomentumTransformer

def run_upgraded_backtest(model, df, feat_cols, ann_factor, seq_len=252, base_cost=0.0001):
    """
    Run backtest for the Upgraded V2 model across all assets.
    Matches the training logic in train_hpc.py.
    """
    symbols = df['symbol'].unique()
    portfolio_positions = {}
    
    # Mapping for Asset IDs (matching train_hpc.py)
    asset_id_map = {
        "ES=F": 0, "NQ=F": 0, "RTY=F": 0, "YM=F": 0, "ER2=F": 0, "^N225": 0, "^FTSE": 0, "^GDAXI": 0, "^FCHI": 0, "^HSI": 0, "^AXJO": 0, "^SSMI": 0, "DX=F": 0,
        "ZT=F": 1, "ZF=F": 1, "ZN=F": 1, "ZB=F": 1, "FGBL=F": 1, "FGBM=F": 1, "FGBS=F": 1, "TWE=F": 1,
        "CL=F": 2, "NG=F": 2, "RB=F": 2, "HO=F": 2, "BZ=F": 2, "QM=F": 2, "GC=F": 2, "SI=F": 2, "HG=F": 2, "PL=F": 2, "PA=F": 2,
        "ZC=F": 3, "ZW=F": 3, "ZS=F": 3, "CC=F": 3, "KC=F": 3, "CT=F": 3, "SB=F": 3, "OJ=F": 3, "LE=F": 3, "GF=F": 3, "HE=F": 3,
        "6E=F": 4, "6J=F": 4, "6B=F": 4, "6A=F": 4, "6C=F": 4, "6S=F": 4, "6N=F": 4, "6M=F": 4, "6L=F": 4
    }

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        for sym in symbols:
            print(f"  Backtesting {sym}...")
            group = df[df['symbol'] == sym].sort_index()
            group = group.ffill().bfill()
            
            if len(group) < seq_len:
                continue

            # Prepare Inputs (matching TimeSeriesDataset)
            x_vals = group[feat_cols].values.astype(np.float32)
            rets = group['scaled_returns'].values.astype(np.float32)
            spreads = group['spread'].values.astype(np.float32) if 'spread' in group.columns else np.ones_like(rets) * 0.0001
            pm = group['prior_mom'].values.astype(np.float32)
            mon = group.index.month.values.astype(np.int64)
            day = group.index.dayofweek.values.astype(np.int64)
            aid = asset_id_map.get(sym, 0)
            
            # Prepare windows
            num_steps = len(x_vals)
            windows_x = []
            windows_pm = []
            windows_mon = []
            windows_day = []
            
            for i in range(num_steps):
                if i < seq_len:
                    # Pad with zeros
                    wx = np.zeros((seq_len, x_vals.shape[1]), dtype=np.float32)
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
            
            # Batch Inference
            pos_list = []
            batch_size = 512
            for b in range(0, len(windows_x), batch_size):
                bx = torch.from_numpy(np.stack(windows_x[b:b+batch_size])).to(device)
                bpm = torch.from_numpy(np.stack(windows_pm[b:b+batch_size])).to(device)
                bmon = torch.from_numpy(np.stack(windows_mon[b:b+batch_size])).to(device)
                bday = torch.from_numpy(np.stack(windows_day[b:b+batch_size])).to(device)
                baid = torch.full((bx.size(0),), aid, dtype=torch.long).to(device)
                
                out = model(bx, asset_id=baid, month=bmon, day=bday, prior_mom=bpm)
                pos_list.append(out[:, -1, 0].cpu().numpy())
            
            positions = np.concatenate(pos_list)
            
            # Returns Calculation
            shifted_rets = np.roll(rets, -1)
            shifted_rets[-1] = 0
            
            pos_diff = np.abs(np.diff(positions, prepend=0))
            costs = pos_diff * (base_cost + spreads / 2.0)
            
            net_rets = (positions * shifted_rets) - costs
            portfolio_positions[sym] = pd.Series(net_rets, index=group.index)

    # Aggregate Portfolio
    combined_df = pd.DataFrame(portfolio_positions).ffill().fillna(0)
    portfolio_rets = combined_df.mean(axis=1)
    cum_pnl = portfolio_rets.cumsum()
    
    # Metrics
    sr = (portfolio_rets.mean() / (portfolio_rets.std() + 1e-9)) * np.sqrt(ann_factor)
    
    # OOS AUDIT (2025 onwards)
    oos_start = pd.to_datetime('2025-01-01')
    dates_no_tz = combined_df.index.tz_localize(None) if combined_df.index.tz is not None else combined_df.index
    oos_mask = dates_no_tz >= oos_start
    
    if oos_mask.any():
        oos_rets = portfolio_rets[oos_mask]
        sr_oos = (oos_rets.mean() / (oos_rets.std() + 1e-9)) * np.sqrt(ann_factor)
    else:
        sr_oos = 0.0

    return {
        'dates': combined_df.index,
        'cum_pnl': cum_pnl,
        'sr': sr,
        'sr_oos': sr_oos,
        'symbols': symbols,
        'portfolio_rets': portfolio_rets
    }

import argparse

def main():
    parser = argparse.ArgumentParser(description="Audit Momentum Transformer Performance")
    parser.add_argument("--epoch", type=int, default=None, help="Specific epoch weights to audit")
    parser.add_argument("--final", action="store_true", help="Audit the final weights")
    args = parser.parse_args()

    # 1. Setup
    data_path = 'data/processed/futures_1990_present.parquet'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_parquet(data_path)
    
    feat_cols = [f'ret_{k}' for k in [1, 21, 63, 126, 252]] + \
                [f'macd_{S}_{L}' for S, L in [[8, 24], [16, 48], [32, 96]]]
    
    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleMomentumTransformer(
        input_dim=1,
        num_vars=len(feat_cols),
        hidden_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        output_dropout=0.5,
        num_static_vars=5
    ).to(device)
    
    if args.final:
        weight_path = 'weights/strict_v2_final.pt'
    elif args.epoch:
        weight_path = f'weights/strict_v2_epoch_{args.epoch}.pt'
    else:
        # Auto-detect latest strict weight
        weight_files = [f for f in os.listdir('weights') if f.startswith('strict_v2_epoch_')]
        if not weight_files:
            print("No strict weights found.")
            return
        latest = sorted(weight_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        weight_path = os.path.join('weights', latest)
        print(f"Auto-detected latest weights: {weight_path}")

    if not os.path.exists(weight_path):
        print(f"Error: {weight_path} not found.")
        return
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    # Run Backtest
    print(f"--- Auditing Momentum Transformer (Weights: {weight_path}) ---")
    results = run_upgraded_backtest(model, df, feat_cols, ann_factor=252, seq_len=252)
    
    # ... (rest of the plotting/audit logic)

    print(f"\n--- 46-Asset Upgraded V2 Audit ---")
    print(f"  Full Period Net Sharpe: {results['sr']:.2f}")
    print(f"  OOS (2025+) Net Sharpe: {results['sr_oos']:.2f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results['dates'], results['cum_pnl'], label=f'Upgraded V2 Portfolio (SR: {results["sr"]:.2f})', color='darkgreen')
    
    oos_start = pd.to_datetime('2025-01-01')
    plt.axvline(oos_start, color='red', linestyle='--', label='OOS Split')
    
    plt.title("Upgraded Momentum Transformer V2: Performance (Net)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Net Returns")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("research/backtest_upgraded_v2.png")
    print("\nBacktest plot saved to research/backtest_upgraded_v2.png")

if __name__ == "__main__":
    main()
