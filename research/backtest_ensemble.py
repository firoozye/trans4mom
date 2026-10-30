import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.architecture import EnsembleMomentumTransformer
from research.backtest_upgraded_v2 import run_upgraded_backtest

def run_ensemble_backtest(model_paths, df, feat_cols, ann_factor, seq_len=252, base_cost=0.0001):
    """
    Load multiple models, average their position signals, and calculate ensemble performance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = df['symbol'].unique()
    
    # Mapping for Asset IDs
    asset_id_map = {
        "ES=F": 0, "NQ=F": 0, "RTY=F": 0, "YM=F": 0, "^N225": 0, "^FTSE": 0, "^GDAXI": 0, "^FCHI": 0, "^HSI": 0, "^AXJO": 0, "^SSMI": 0,
        "ZT=F": 1, "ZF=F": 1, "ZN=F": 1, "ZB=F": 1,
        "CL=F": 2, "NG=F": 2, "RB=F": 2, "HO=F": 2, "BZ=F": 2, "QM=F": 2,
        "GC=F": 2, "SI=F": 2, "HG=F": 2, "PL=F": 2, "PA=F": 2,
        "ZC=F": 3, "ZW=F": 3, "ZS=F": 3, "CC=F": 3, "KC=F": 3, "CT=F": 3, "SB=F": 3, "OJ=F": 3, "LE=F": 3, "GF=F": 3, "HE=F": 3,
        "6E=F": 4, "6J=F": 4, "6B=F": 4, "6A=F": 4, "6C=F": 4, "6S=F": 4, "6N=F": 4, "6M=F": 4, "6L=F": 4
    }

    # 1. Load All Models
    models = []
    for path in model_paths:
        print(f"Loading seed weights: {path}")
        m = EnsembleMomentumTransformer(
            input_dim=1, num_vars=len(feat_cols), hidden_dim=64,
            num_heads=4, num_layers=1, dropout=0.1, output_dropout=0.1, num_static_vars=5
        ).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)

    portfolio_rets = {}
    
    with torch.no_grad():
        for sym in symbols:
            print(f"  Ensembling {sym}...")
            group = df[df['symbol'] == sym].sort_index().ffill().bfill()
            if len(group) < seq_len: continue

            x_vals = group[feat_cols].values.astype(np.float32)
            rets = group['scaled_returns'].values.astype(np.float32)
            spreads = group['spread'].values.astype(np.float32) if 'spread' in group.columns else np.ones_like(rets) * 0.0001
            pm = group['prior_mom'].values.astype(np.float32)
            mon = group.index.month.values.astype(np.int64)
            day = group.index.dayofweek.values.astype(np.int64)
            aid = asset_id_map.get(sym, 0)
            
            # Prepare windows
            num_steps = len(x_vals)
            windows_x, windows_pm, windows_mon, windows_day = [], [], [], []
            for i in range(num_steps):
                if i < seq_len:
                    wx = np.zeros((seq_len, x_vals.shape[1]), dtype=np.float32)
                    wx[-(i+1):] = x_vals[:i+1]
                    wpm = np.zeros((seq_len, 1), dtype=np.float32)
                    wpm[-(i+1):] = pm[:i+1].reshape(-1, 1)
                    wmon = np.zeros(seq_len, dtype=np.int64)
                    wmon[-(i+1):] = mon[:i+1]
                    wday = np.zeros(seq_len, dtype=np.int64)
                    wday[-(i+1):] = day[:i+1]
                else:
                    wx, wpm, wmon, wday = x_vals[i-seq_len+1:i+1], pm[i-seq_len+1:i+1].reshape(-1,1), mon[i-seq_len+1:i+1], day[i-seq_len+1:i+1]
                windows_x.append(wx); windows_pm.append(wpm); windows_mon.append(wmon); windows_day.append(wday)
            
            # Inference and Average
            batch_size = 512
            ensemble_pos = np.zeros(len(windows_x))
            for b in range(0, len(windows_x), batch_size):
                bx = torch.from_numpy(np.stack(windows_x[b:b+batch_size])).to(device)
                bpm = torch.from_numpy(np.stack(windows_pm[b:b+batch_size])).to(device)
                bmon = torch.from_numpy(np.stack(windows_mon[b:b+batch_size])).to(device)
                bday = torch.from_numpy(np.stack(windows_day[b:b+batch_size])).to(device)
                baid = torch.full((bx.size(0),), aid, dtype=torch.long).to(device)
                
                # Average predictions from all 5 models
                batch_preds = []
                for m in models:
                    out = m(bx, asset_id=baid, month=bmon, day=bday, prior_mom=bpm)
                    batch_preds.append(out[:, -1, 0].cpu().numpy())
                
                ensemble_pos[b:b+batch_size] = np.mean(batch_preds, axis=0)

            # Performance
            # Performance and physical transaction costs
            vol_scale = group['vol_scale'].values.astype(np.float32) if 'vol_scale' in group.columns else np.ones_like(rets)
            physical_positions = ensemble_pos * vol_scale
            
            shifted_rets = np.roll(rets, -1); shifted_rets[-1] = 0
            costs = np.abs(np.diff(physical_positions, prepend=0)) * (base_cost + spreads/2.0)
            portfolio_rets[sym] = pd.Series((ensemble_pos * shifted_rets) - costs, index=group.index)

    # Aggregate
    combined_df = pd.DataFrame(portfolio_rets).ffill().fillna(0)
    p_rets = combined_df.mean(axis=1)
    
    # Metrics
    sr = (p_rets.mean() / (p_rets.std() + 1e-9)) * np.sqrt(ann_factor)
    oos_start = pd.to_datetime('2025-01-01')
    dates_no_tz = combined_df.index.tz_localize(None) if combined_df.index.tz is not None else combined_df.index
    oos_mask = dates_no_tz >= oos_start
    sr_oos = (p_rets[oos_mask].mean() / (p_rets[oos_mask].std() + 1e-9)) * np.sqrt(ann_factor) if oos_mask.any() else 0.0
    
    return {'dates': combined_df.index, 'cum_pnl': p_rets.cumsum(), 'sr': sr, 'sr_oos': sr_oos}

def main():
    df = pd.read_parquet('data/processed/futures_1990_present.parquet')
    feat_cols = [f'ret_{k}' for k in [1, 21, 63, 126, 252]] + [f'macd_{S}_{L}' for S, L in [[8, 24], [16, 48], [32, 96]]] + ['cpd_sev_21', 'cpd_sev_126']
    
    # Collect all ensemble best weights
    weights_dir = 'weights'
    model_paths = [os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.startswith('ensemble_best_seed_')]
    
    if not model_paths:
        print("No ensemble weights found.")
        return

    print(f"--- Running 5-Seed Ensemble Backtest ({len(model_paths)} models) ---")
    res = run_ensemble_backtest(model_paths, df, feat_cols, ann_factor=252)

    print(f"\n--- ENSEMBLE AUDIT RESULTS ---")
    print(f"  Full Period Net Sharpe: {res['sr']:.2f}")
    print(f"  OOS (2025+) Net Sharpe: {res['sr_oos']:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(res['dates'], res['cum_pnl'], label=f'5-Seed Ensemble (SR: {res["sr"]:.2f})', color='purple')
    plt.axvline(pd.to_datetime('2025-01-01'), color='red', linestyle='--')
    plt.title("Momentum Transformer: 5-Seed Ensemble Performance")
    plt.savefig("research/backtest_ensemble_v10.png")

if __name__ == "__main__":
    main()
