import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.architecture import MomentumTransformer

def estimate_spread_corwin_schultz(high, low):
    """
    Estimate bid-ask spread using Corwin-Schultz (2012) High-Low method.
    """
    # Calculate log(H/L) for 1-period and 2-period
    kh1 = np.log(high / low)**2
    # This is a simplified version for the backtest
    gamma = kh1.rolling(window=2).max()
    spread = 2 * (np.exp(np.sqrt(gamma)) - 1) / (1 + np.exp(np.sqrt(gamma)))
    return spread.fillna(spread.mean())

def estimate_spread_abdi_ranaldo(high, low, close):
    """
    Abdi & Ranaldo (2017) High-Low-Close spread estimator.
    More robust than Corwin-Schultz.
    """
    mid = (high + low) / 2.0
    # s = 2 * sqrt(E[(c_t - eta_t)(c_t - eta_{t-1})])
    # Term: (c_t - eta_t)(c_t - eta_{t-1})
    term = (close - mid) * (close - mid.shift(1))
    # Handle negative terms by taking max(0, term) before sqrt
    spread_raw = 2 * np.sqrt(term.clip(lower=0))
    # Normalize by mid price to get decimal percentage spread
    spread_pct = spread_raw / mid
    return spread_pct.fillna(spread_pct.mean())

def run_backtest(model, df, symbol, start_date, end_date, ann_factor, base_cost=0.002, mode='net'):
    """
    Run model backtest. 
    mode='gross': zero costs (The Fantasy)
    mode='net': dynamic spreads + fixed slippage (The Reality)
    """
    asset_df = df[(df['symbol'] == symbol) & (df.index >= start_date) & (df.index <= end_date)].sort_index()
    if len(asset_df) < 100:
        return None
    
    # Estimate dynamic spreads
    if 'high' in asset_df.columns and 'low' in asset_df.columns:
        ar_spread = estimate_spread_abdi_ranaldo(asset_df['high'], asset_df['low'], asset_df['close'])
    else:
        ar_spread = pd.Series(0.0, index=asset_df.index)

    # Prepare features
    feat_cols = [c for c in asset_df.columns if 'macd' in c]
    x_vals = asset_df[feat_cols].values
    rets = asset_df['returns'].values
    dates = asset_df.index
    
    seq_len = 64
    
    # Vectorized Inference
    model.eval()
    with torch.no_grad():
        # Create sliding window tensor
        num_steps = len(x_vals)
        windows = []
        for i in range(num_steps):
            if i < seq_len:
                win = np.zeros((seq_len, len(feat_cols)))
                win[-(i+1):] = x_vals[:i+1]
            else:
                win = x_vals[i-seq_len+1 : i+1]
            windows.append(win)
        
        # Batch inference (processing in chunks to avoid OOM if needed)
        x_tensor = torch.tensor(np.array(windows), dtype=torch.float32).unsqueeze(-1)
        # Assuming batch_size=512 for speed
        all_pos = []
        for b in range(0, len(x_tensor), 512):
            batch_x = x_tensor[b : b+512].to(next(model.parameters()).device)
            out = model(batch_x)
            all_pos.append(out[:, -1, 0].cpu().numpy())
        
        positions = np.concatenate(all_pos)
            
    # 2. Returns Calculation
    gross_rets = positions[:-1] * rets[1:]
    turnover = np.abs(np.diff(positions))
    
    if mode == 'gross':
        final_rets = gross_rets
    else:
        # Dynamic cost = Slippage (base_cost) + Half-Spread
        # Multiplied by turnover
        trading_costs = turnover * (base_cost + ar_spread.values[1:] / 2.0)
        final_rets = gross_rets - trading_costs
    
    # Performance
    cum_pnl = np.cumsum(final_rets)
    sr = (np.mean(final_rets) / (np.std(final_rets) + 1e-9)) * np.sqrt(ann_factor)
    
    return {
        'dates': dates[1:],
        'cum_pnl': cum_pnl,
        'positions': positions[1:],
        'sr': sr,
        'turnover': np.mean(turnover) * (ann_factor),
        'avg_spread': ar_spread.mean()
    }

def plot_backtest_with_pos(dates, pnl, positions, title, filename, sr, turnover, split_date=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # PnL Plot
    ax1.plot(dates, pnl, color='blue', linewidth=1.5)
    ax1.set_title(f"{title}\nSharpe: {sr:.2f} | Turnover: {turnover:.1f}x")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(True, alpha=0.3)
    if split_date:
        ax1.axvline(pd.to_datetime(split_date), color='red', linestyle='--', label='OOS Split')
        ax1.legend()

    # Position Plot
    ax2.fill_between(dates, 0, positions, color='orange', alpha=0.3, label='Model Position')
    ax2.plot(dates, positions, color='darkorange', linewidth=0.5)
    ax2.set_ylabel("Position")
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    # 1. Setup
    data_path = 'data/processed/crypto_data.parquet'
    df = pd.read_parquet(data_path)
    ann_factor = 2190 # 4h crypto
    symbols = df['symbol'].unique()
    
    # 2. Load Model
    model = MomentumTransformer(input_dim=1, num_vars=5, hidden_dim=64, num_heads=4)
    model.load_state_dict(torch.load('weights/model_slow_mom_100bps.pt', map_location='cpu'))
    
    # 3. Define Periods
    train_end = '2024-12-31'
    
    print(f"--- Generating Masterclass Visuals for {len(symbols)} symbols ---")
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        
        # Run Full Period (Train + OOS)
        gross_res = run_backtest(model, df, symbol, df.index.min(), df.index.max(), ann_factor, mode='gross')
        net_res = run_backtest(model, df, symbol, df.index.min(), df.index.max(), ann_factor, mode='net')
        
        if net_res is None:
            continue

        print(f"  FULL PERIOD STATS:")
        print(f"    Gross Sharpe Ratio: {gross_res['sr']:.2f}")
        print(f"    Net Sharpe Ratio: {net_res['sr']:.2f}")
        print(f"    Annualized Turnover: {net_res['turnover']:.1f}x")
        
        # Calculate OOS Sharpe (2025 onwards)
        oos_start = '2025-01-01'
        oos_mask = net_res['dates'] >= pd.to_datetime(oos_start)
        if oos_mask.any():
            oos_rets = np.diff(net_res['cum_pnl'][oos_mask]) # Approx returns
            # Re-calculate Sharpe for OOS
            oos_sr = (np.mean(oos_rets) / (np.std(oos_rets) + 1e-9)) * np.sqrt(ann_factor)
            print(f"    OOS Net Sharpe Ratio (2025+): {oos_sr:.2f}")
        
        # 4. Detailed Plotting
        sym_clean = symbol.replace('/', '_')
        
        # Plot A: Gross
        plot_backtest_with_pos(gross_res['dates'], gross_res['cum_pnl'], gross_res['positions'], 
                              f"Momentum Transformer: Gross ({symbol})", f"research/backtest_gross_{sym_clean}.png", 
                              gross_res['sr'], gross_res['turnover'], split_date=train_end)

        # Plot B: Net
        plot_backtest_with_pos(net_res['dates'], net_res['cum_pnl'], net_res['positions'], 
                              f"Momentum Transformer: Net ({symbol})", f"research/backtest_net_{sym_clean}.png", 
                              net_res['sr'], net_res['turnover'], split_date=train_end)

        # Plot C: Comparison (only for the primary asset BTC to keep presentation lean, or all)
        if symbol == 'BTC/USDT':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            ax1.plot(gross_res['dates'], gross_res['cum_pnl'], label=f"Gross (SR: {gross_res['sr']:.2f})", alpha=0.4, linestyle='--')
            ax1.plot(net_res['dates'], net_res['cum_pnl'], label=f"Net (SR: {net_res['sr']:.2f})", color='green', linewidth=2)
            ax1.axvline(pd.to_datetime(train_end), color='red', linestyle=':', label='OOS Split')
            ax1.set_title(f"Momentum Transformer: Gross vs Net Comparison ({symbol})")
            ax1.set_ylabel("Cumulative Return")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.fill_between(net_res['dates'], 0, net_res['positions'], color='green', alpha=0.2, label='Net Positions')
            ax2.set_ylabel("Position")
            ax2.set_ylim(-1.1, 1.1)
            ax2.grid(True, alpha=0.2)
            
            plt.tight_layout()
            plt.savefig(f"research/backtest_comparison_{sym_clean}.png")
            plt.close()
    
    print("\nSUCCESS: All symbol plots with position panels saved to research/")

if __name__ == "__main__":
    main()
