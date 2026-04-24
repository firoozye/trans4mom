import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.architecture import MomentumTransformer

def plot_multi_asset_positions(dates, positions, symbols, filename="research/cs_positions_heatmap.png"):
    """Plot a heatmap of positions for all assets over time."""
    plt.figure(figsize=(15, 8))
    # Taking a subset of dates for clarity if needed, or plotting full
    df_pos = pd.DataFrame(positions, index=dates, columns=symbols)
    sns.heatmap(df_pos.T, cmap='RdBu', center=0)
    plt.title("Cross-Sectional Model Positions (Long/Short Heatmap)")
    plt.xlabel("Date")
    plt.ylabel("Asset")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_portfolio_diagnostics(dates, cum_pnl, net_exposure, sr, turnover):
    """Plot equity curve and net exposure for the portfolio."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Equity Curve
    ax1.plot(dates, cum_pnl, color='green', linewidth=2)
    ax1.set_title(f"Cross-Sectional Portfolio\nSharpe: {sr:.2f} | Turnover: {turnover:.1f}x")
    ax1.set_ylabel("Cumulative Net Return")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(pd.to_datetime('2025-01-01'), color='red', linestyle='--', label='OOS Split')
    ax1.legend()

    # Net Exposure (Neutrality Check)
    ax2.fill_between(dates, 0, net_exposure, color='gray', alpha=0.3, label='Net Exposure')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_ylabel("Net Exposure")
    ax2.set_ylim(-0.2, 0.2)
    ax2.set_title("Market Neutrality Check (Target: 0)")
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("research/cs_portfolio_performance.png")
    plt.close()

def plot_attention(model, x, filename="research/cs_attention_map.png"):
    """Visualize self-attention weights for the cross-sectional model."""
    model.eval()
    with torch.no_grad():
        _, attn_weights = model(x, return_attention=True)
    
    # attn_weights shape: (batch, num_heads, time, time)
    if len(attn_weights.shape) == 4:
        weights = attn_weights[0, 0].cpu().numpy() # First batch, first head
    else:
        weights = attn_weights[0].cpu().numpy()
    
    # Ensure weights is 2D
    if len(weights.shape) == 3:
        weights = weights[0]
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, cmap='magma')
    plt.title("Transformer Attention Map (Temporal Dependencies)")
    plt.xlabel("Key (Past)")
    plt.ylabel("Query (Present)")
    plt.savefig(filename)
    plt.close()

def main():
    # 1. Load Data and Model
    data_path = 'data/processed/crypto_1d.parquet'
    df = pd.read_parquet(data_path)
    symbols = df['symbol'].unique()
    num_assets = len(symbols)
    feat_cols = [f'macd_{w}' for w in [10, 21, 63, 126, 252]]
    
    model = MomentumTransformer(input_dim=len(feat_cols), num_vars=num_assets, hidden_dim=64, num_heads=4, output_dim=num_assets)
    model.load_state_dict(torch.load('weights/model_cross_sectional_latest.pt', map_location='cpu'))
    
    # 2. Re-run small inference for visuals
    pivoted = df.pivot_table(index=df.index, columns='symbol', values=feat_cols + ['returns', 'spread'])
    pivoted = pivoted.sort_index().ffill().bfill()
    
    x_list = [pivoted[f].values for f in feat_cols]
    x_vals = np.stack(x_list, axis=-1)
    
    seq_len = 64
    # Just take the last 500 days for clear attention visuals
    sample_x = x_vals[-seq_len:]
    sample_tensor = torch.tensor(sample_x, dtype=torch.float32).unsqueeze(0)
    
    # Full inference for heatmap
    windows = []
    for i in range(len(x_vals)):
        if i < seq_len:
            win = np.zeros((seq_len, num_assets, len(feat_cols)))
            win[-(i+1):] = x_vals[:i+1]
        else:
            win = x_vals[i-seq_len+1 : i+1]
        windows.append(win)
    
    model.eval()
    with torch.no_grad():
        full_x = torch.tensor(np.array(windows), dtype=torch.float32)
        all_pos = []
        for b in range(0, len(full_x), 128):
            out = model(full_x[b:b+128])
            all_pos.append(out[:, -1, :].numpy())
        positions = np.concatenate(all_pos)

    # 3. Generate Visuals
    print("Generating Cross-Sectional Visuals...")
    plot_multi_asset_positions(pivoted.index, positions, symbols)
    plot_attention(model, sample_tensor)
    
    # For performance curve, we'll reuse the backtest logic but save the plot here
    # (Simplified for now as we already have backtest_cross_sectional.png)
    print("SUCCESS: Visuals saved to research/")

if __name__ == "__main__":
    main()
