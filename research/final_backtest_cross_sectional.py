import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.architecture import MomentumTransformer

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
    # Note: rets[t] is return of period (t-1, t). 
    # positions[t] is position decided at end of t, captures rets[t+1].
    # So we shift rets to align.
    shifted_rets = rets[1:, :]
    active_pos = positions[:-1, :]
    
    gross_rets = np.sum(active_pos * shifted_rets, axis=1) / num_assets
    
    # Costs
    turnover = np.sum(np.abs(np.diff(positions, axis=0)), axis=1) / num_assets
    avg_spread = np.mean(spreads, axis=1)
    trading_costs = turnover * (base_cost + avg_spread[:-1] / 2.0)
    
    net_rets = gross_rets - trading_costs
    
    # Performance
    cum_pnl = np.cumsum(net_rets)
    sr = (np.mean(net_rets) / (np.std(net_rets) + 1e-9)) * np.sqrt(ann_factor)
    
    return {
        'dates': dates[1:],
        'cum_pnl': cum_pnl,
        'sr': sr,
        'turnover': np.mean(turnover) * ann_factor,
        'net_exposure': np.mean(positions, axis=1)
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
    model = MomentumTransformer(input_dim=len(feat_cols), num_vars=num_assets, hidden_dim=64, num_heads=4, output_dim=num_assets)
    model.load_state_dict(torch.load('weights/model_macro_cs_15.pt', map_location='cpu'))
    
    # 3. Backtest
    print(f"--- Running Cross-Sectional Backtest ({num_assets} Assets) ---")
    results = run_backtest_multi(model, df, feat_cols, ann_factor)
    
    print(f"Portfolio Results:")
    print(f"  Sharpe Ratio: {results['sr']:.2f}")
    print(f"  Annualized Turnover: {results['turnover']:.1f}x")
    print(f"  Avg Net Exposure: {np.mean(results['net_exposure']):.4f}")

    # 4. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results['dates'], results['cum_pnl'], label=f'CS Portfolio (SR: {results["sr"]:.2f})')
    plt.axvline(pd.to_datetime('2025-01-01'), color='red', linestyle='--', label='OOS Split')
    plt.title("Cross-Sectional Momentum Portfolio Performance (Net)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("research/backtest_cross_sectional.png")
    print("\nPlot saved to research/backtest_cross_sectional.png")

if __name__ == "__main__":
    main()
