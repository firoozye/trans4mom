import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.architecture import EnsembleMomentumTransformer

def run_backtest_multi(model, df, feat_cols, ann_factor, base_cost=0.002):
    """
    Run backtest for multi-asset cross-sectional model.
    """
    symbols = df['symbol'].unique()
    num_assets = len(symbols)
    seq_len = 64
    
    # Pivot for multi-asset alignment
    pivoted = df.pivot_table(index=df.index, columns='symbol', values=feat_cols + ['returns', 'spread'])
    pivoted = pivoted.sort_index().ffill().bfill()
    
    # Prepare features
    x_list = [pivoted[f].values for f in feat_cols]
    x_vals = np.stack(x_list, axis=-1) # (time, assets, num_features)
    rets = pivoted['returns'].values # (time, assets)
    spreads = pivoted['spread'].values # (time, assets)
    dates = pivoted.index
    
    # Vectorized Inference
    model.eval()
    with torch.no_grad():
        num_steps = len(x_vals)
        windows = []
        for i in range(num_steps):
            if i < seq_len:
                win = np.zeros((seq_len, num_assets, len(feat_cols)))
                win[-(i+1):] = x_vals[:i+1]
            else:
                win = x_vals[i-seq_len+1 : i+1]
            windows.append(win)
        
        x_tensor = torch.tensor(np.array(windows), dtype=torch.float32)
        all_pos = []
        for b in range(0, len(x_tensor), 128):
            batch_x = x_tensor[b : b+128].to(next(model.parameters()).device)
            out = model(batch_x)
            all_pos.append(out[:, -1, :].cpu().numpy())
        
        positions = np.concatenate(all_pos) # (time, num_assets)

    # 2. Portfolio Returns
    shifted_rets = rets[1:, :]
    active_pos = positions[:-1, :]
    dates_backtest = dates[1:]

    # Precise per-asset trading costs
    pos_diff = np.abs(np.diff(positions, axis=0))
    asset_costs = pos_diff * (base_cost + spreads[1:, :] / 2.0)

    gross_rets_per_asset = active_pos * shifted_rets
    net_rets_per_asset = gross_rets_per_asset - asset_costs

    net_rets = np.sum(net_rets_per_asset, axis=1) / num_assets
    cum_pnl = np.cumsum(net_rets)

    # Full Period SR
    sr_full = (np.mean(net_rets) / (np.std(net_rets) + 1e-9)) * np.sqrt(ann_factor)

    # --- OOS AUDIT (2025+) ---
    oos_start = pd.to_datetime('2025-01-01').tz_localize('UTC')
    oos_mask = dates_backtest >= oos_start
    if oos_mask.any():
        oos_net_rets = net_rets[oos_mask]
        sr_oos = (np.mean(oos_net_rets) / (np.std(oos_net_rets) + 1e-9)) * np.sqrt(ann_factor)
    else:
        sr_oos = 0.0

    return {
        'dates': dates_backtest,
        'cum_pnl': cum_pnl,
        'sr': sr_full,
        'sr_oos': sr_oos,
        'turnover': np.mean(np.sum(pos_diff, axis=1) / num_assets) * ann_factor,
        'net_exposure': np.mean(positions, axis=1),
        'symbols': symbols
    }

def main():
    # 1. Setup
    data_path = 'data/processed/futures_1d.parquet'
    df = pd.read_parquet(data_path)
    ann_factor = 252 # Futures are 252 days
    feat_cols = [f'macd_{w}' for w in [10, 21, 63, 126, 252]]
    
    # Identify unique symbols present in the data
    symbols = df['symbol'].unique()
    num_assets = len(symbols)
    print(f"Detected {num_assets} assets in data: {symbols}")
    # 2. Load Model
    model = EnsembleMomentumTransformer(input_dim=len(feat_cols), num_vars=num_assets, hidden_dim=32, num_heads=4, output_dim=num_assets)
    model.load_state_dict(torch.load('weights/model_macro_ensemble.pt', map_location='cpu'))
    
    # 3. Backtest (Realistic)
    print(f"--- Running Cross-Sectional Backtest ({num_assets} Assets) ---")
    results = run_backtest_multi(model, df, feat_cols, ann_factor, base_cost=0.002)
    
    print(f"\n--- 15-Asset Macro Audit ---")
    print(f"  Full Period Net Sharpe: {results['sr']:.2f}")
    print(f"  OOS (2025+) Net Sharpe: {results['sr_oos']:.2f}")
    print(f"  Annualized Turnover:    {results['turnover']:.1f}x")
    print(f"\n--- Running Cross-Sectional Backtest (Pure AR Spread Only) ---")
    results_pure = run_backtest_multi(model, df, feat_cols, ann_factor, base_cost=0.0)
    print(f"  Sharpe Ratio (Pure AR): {results_pure['sr']:.2f}")

    # 5. Plot Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(results['dates'], results['cum_pnl'], label=f'Realistic (SR: {results["sr"]:.2f})', color='green')
    plt.plot(results_pure['dates'], results_pure['cum_pnl'], label=f'Pure AR (SR: {results_pure["sr"]:.2f})', color='blue', alpha=0.7)
    plt.axvline(pd.to_datetime('2025-01-01'), color='red', linestyle='--', label='OOS Split')
    plt.title("Cross-Sectional Momentum: Realistic vs. Pure AR Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("research/backtest_cs_comparison.png")
    
    # Save individual Pure AR plot for the 'Amazing' slide
    plt.figure(figsize=(12, 6))
    plt.plot(results_pure['dates'], results_pure['cum_pnl'], color='blue', linewidth=2)
    plt.title(f"Intrinsic Alpha: Pure AR Momentum Portfolio (Sharpe: {results_pure['sr']:.2f})")
    plt.ylabel("Cumulative Net Return")
    plt.grid(True, alpha=0.3)
    plt.savefig("research/backtest_cs_pure_ar.png")
    
    print("\nPlots saved to research/ (including backtest_cs_pure_ar.png)")

if __name__ == "__main__":
    main()
