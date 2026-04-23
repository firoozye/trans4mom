import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.architecture import MomentumTransformer
from data.processor import FeatureProcessor

def plot_cumulative_returns(df, symbols):
    """Plot cumulative returns of the assets using multiple y-axes to handle scaling diffs."""
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    axes = [ax1]
    
    # Create extra y-axes for other assets
    for i in range(len(symbols) - 1):
        axes.append(ax1.twinx())
        # Offset the extra axes to the right
        if i > 0:
            axes[-1].spines['right'].set_position(('outward', 60 * i))

    for i, symbol in enumerate(symbols):
        asset_df = df[df['symbol'] == symbol].sort_index()
        # Use log returns to compute cumulative price index starting at 1.0
        cum_ret = np.exp(asset_df['returns'].cumsum())
        
        ax = axes[i]
        ax.plot(asset_df.index, cum_ret, label=symbol, color=colors[i % len(colors)])
        ax.set_ylabel(f"{symbol} Index", color=colors[i % len(colors)])
        ax.tick_params(axis='y', labelcolor=colors[i % len(colors)])
    
    plt.title("Asset Cumulative Returns (Data Overview - Multiple Scales)")
    fig.tight_layout()
    plt.grid(True, alpha=0.1)
    plt.savefig("research/data_overview.png")
    plt.close()

def calculate_variance_ratio(returns, q):
    """
    Calculate the Variance Ratio for a given lag q.
    VR(q) = Var(r_t + ... + r_{t-q+1}) / (q * Var(r_t))
    VR > 1 implies Momentum (Trending)
    VR < 1 implies Mean Reversion
    """
    returns = returns.dropna()
    n = len(returns)
    mu = returns.mean()
    
    # 1-period variance
    sig_1 = returns.var(ddof=1)
    
    # q-period returns
    q_returns = returns.rolling(window=q).sum().dropna()
    sig_q = q_returns.var(ddof=1)
    
    vr = sig_q / (q * sig_1)
    return vr

def plot_momentum_feasibility(df, symbol, lags=50):
    """Plot autocorrelation and Variance Ratio to show if momentum exists."""
    asset_df = df[df['symbol'] == symbol].copy()
    returns = asset_df['returns'].dropna()
    
    # ACF Plot
    acf = [returns.autocorr(lag=i) for i in range(1, lags)]
    
    # VR Plot for different q
    q_values = [2, 5, 10, 21, 63, 126]
    vr_values = [calculate_variance_ratio(returns, q) for q in q_values]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Ax1: ACF
    ax1.bar(range(1, lags), acf, alpha=0.7, color='steelblue')
    ax1.axhline(y=0, color='black', linestyle='--')
    ax1.axhline(y=1.96/np.sqrt(len(returns)), color='red', linestyle='--', label='95% Confidence')
    ax1.axhline(y=-1.96/np.sqrt(len(returns)), color='red', linestyle='--')
    ax1.set_title(f"Autocorrelation of Returns ({symbol})")
    ax1.set_ylabel("Autocorrelation")
    ax1.legend()
    
    # Ax2: Variance Ratio
    ax2.plot(q_values, vr_values, marker='o', linestyle='-', color='darkorange')
    ax2.axhline(y=1.0, color='black', linestyle='--', label='Random Walk (VR=1)')
    ax2.set_title(f"Variance Ratio Test ({symbol})")
    ax2.set_xlabel("q (Lag Period)")
    ax2.set_ylabel("Variance Ratio")
    ax2.set_xscale('log')
    ax2.set_xticks(q_values)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"research/momentum_check_{symbol.replace('/', '_')}.png")
    plt.close()

def plot_strategy_performance(model, x, y_true, title="Strategy Performance"):
    """Plot model predicted positions vs actual returns and cumulative PnL."""
    model.eval()
    with torch.no_grad():
        # Get positions
        positions = model(x) # (batch, time, 1)
        
    # Flatten for plotting (taking the first batch)
    pos = positions[0, :, 0].cpu().numpy()
    rets = y_true[0, :, 0].cpu().numpy()
    
    # Simple PnL: Position * next period return
    # Note: in real backtest we shift rets, here we just show correlation
    strategy_rets = pos[:-1] * rets[1:]
    cum_pnl = np.cumsum(strategy_rets)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(pos, label='Predicted Position', color='blue', alpha=0.7)
    ax1.set_ylabel("Position (-1 to 1)")
    ax1.set_title("Model Output (Positions)")
    ax1.legend()
    
    ax2.plot(cum_pnl, label='Cumulative Strategy PnL', color='green')
    ax2.set_ylabel("Cumulative PnL")
    ax2.set_title("Equity Curve (Goodness of Fit)")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("research/strategy_performance.png")
    plt.close()

def plot_attention_map(model, x):
    """Visualize the attention weights to show what the model is looking at."""
    model.eval()
    with torch.no_grad():
        _, attn_weights = model(x, return_attention=True)
    
    # attn_weights shape: (batch, num_heads, time, time) or (batch, time, time)
    # Taking first batch, first head
    if len(attn_weights.shape) == 4:
        weights = attn_weights[0, 0].cpu().numpy()
    else:
        weights = attn_weights[0].cpu().numpy()
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, cmap='viridis')
    plt.title("Attention Map (Interpretability)")
    plt.xlabel("Lookback Time")
    plt.ylabel("Current Time")
    plt.savefig("research/attention_map.png")
    plt.close()

def main():
    # 1. Load Data
    data_path = "data/processed/crypto_data.parquet"
    if not os.path.exists(data_path):
        print("Data not found. Please run data_pipeline.py first.")
        return
    
    df = pd.read_parquet(data_path)
    symbols = df['symbol'].unique()
    
    # 2. Data Overview
    print("Generating Data Overview with Multi-Axes...")
    plot_cumulative_returns(df, symbols)
    
    # 3. Momentum Check
    print(f"Generating Momentum Check for {symbols[0]}...")
    plot_momentum_feasibility(df, symbols[0])
    
    # 4. Model Visuals
    print("Generating Model Diagnostics...")
    weight_path = 'weights/model_insane_penalty.pt'
    if os.path.exists(weight_path):
        model = MomentumTransformer(input_dim=1, num_vars=5, hidden_dim=64, num_heads=4)
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        print("  Loaded champion weights.")
        
        # Prepare a real sample from the data
        asset_df = df[df['symbol'] == symbols[0]].iloc[:500]
        feat_cols = [c for c in asset_df.columns if 'macd' in c]
        x_vals = torch.tensor(asset_df[feat_cols].values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        y_vals = torch.tensor(asset_df[['scaled_returns']].values, dtype=torch.float32).unsqueeze(0)
        
        plot_strategy_performance(model, x_vals, y_vals)
        plot_attention_map(model, x_vals)
    else:
        print("  Weights not found. Skipping model-specific diagnostics.")
    
    print("\nSUCCESS: All visualizations saved to research/ directory.")

if __name__ == "__main__":
    main()
