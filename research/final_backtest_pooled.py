import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.architecture import EnsembleMomentumTransformer

def run_pooled_backtest(model, df, feat_cols, ann_factor, seq_len=64, base_cost=0.002):
    """
    Run backtest for the Universal Pooled model across all assets.
    """
    symbols = df['symbol'].unique()
    all_rets = []
    all_net_rets = []
    
    # 1. Generate Positions for each asset independently
    portfolio_positions = {}
    
    # Pre-calculate asset IDs for the model
    asset_id_map = {
        "ES=F": 0, "NQ=F": 0, "RTY=F": 0, "YM=F": 0,
        "ZT=F": 1, "ZF=F": 1, "ZN=F": 1, "ZB=F": 1,
        "CL=F": 2, "NG=F": 2, "RB=F": 2, "HO=F": 2, "GC=F": 2, "SI=F": 2, "HG=F": 2, 
        "ZC=F": 2, "ZW=F": 2, "ZS=F": 2,
        "6E=F": 3, "6J=F": 3, "6B=F": 3, "6A=F": 3, "6C=F": 3, "6S=F": 3
    }

    with torch.no_grad():
        for sym in symbols:
            print(f"  Backtesting {sym}...")
            group = df[df['symbol'] == sym].sort_index()
            group = group.ffill().bfill()
            
            x_vals = group[feat_cols].values
            rets = group['scaled_returns'].values
            spreads = group['spread'].values if 'spread' in group.columns else np.ones_like(rets) * 0.0005
            
            # Prepare windows
            num_steps = len(x_vals)
            windows = []
            for i in range(num_steps):
                if i < seq_len:
                    win = np.zeros((seq_len, len(feat_cols)))
                    win[-(i+1):] = x_vals[:i+1]
                else:
                    win = x_vals[i-seq_len+1 : i+1]
                windows.append(win)
            
            x_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(next(model.parameters()).device)
            aid = asset_id_map.get(sym, 4)
            aid_tensor = torch.full((x_tensor.size(0),), aid, dtype=torch.long).to(x_tensor.device)
            
            # Inference in batches to save memory
            pos_list = []
            for b in range(0, len(x_tensor), 256):
                batch_x = x_tensor[b : b+256]
                batch_aid = aid_tensor[b : b+256]
                out = model(batch_x, asset_id=batch_aid)
                # out is (B, T, 1), we want last step
                pos_list.append(out[:, -1, :].cpu().numpy())
            
            positions = np.concatenate(pos_list).flatten()
            
            # Returns Calculation
            shifted_rets = np.roll(rets, -1) # Position at t earns return at t+1
            shifted_rets[-1] = 0
            
            pos_diff = np.abs(np.diff(positions, prepend=0))
            costs = pos_diff * (base_cost + spreads / 2.0)
            
            net_rets = (positions * shifted_rets) - costs
            portfolio_positions[sym] = pd.Series(net_rets, index=group.index)

    # 2. Aggregate Portfolio
    combined_df = pd.DataFrame(portfolio_positions).ffill().fillna(0)
    portfolio_rets = combined_df.mean(axis=1)
    cum_pnl = portfolio_rets.cumsum()
    
    # 3. Metrics
    sr = (portfolio_rets.mean() / (portfolio_rets.std() + 1e-9)) * np.sqrt(ann_factor)
    
    # OOS AUDIT
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
        'symbols': symbols
    }

def main():
    # 1. Setup
    data_path = 'data/processed/futures_1990_present.parquet'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_parquet(data_path)
    feat_cols = [
        'ret_1', 'ret_21', 'ret_63', 'ret_126', 'ret_252',
        'macd_8_24', 'macd_16_48', 'macd_32_96'
    ]
    
    # 2. Load Model (Hidden Dim: 64, Input Dim: 1, Num Vars: 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleMomentumTransformer(
        input_dim=1, 
        num_vars=len(feat_cols), 
        hidden_dim=64, 
        num_heads=4, 
        output_dim=1
    )
    
    weight_path = 'weights/macro_transformer_final.pt'
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"--- Running Pooled Ensemble Backtest (24 assets, 1990-2026) ---")
    results = run_pooled_backtest(model, df, feat_cols, ann_factor=252)

    print(f"\n--- 24-Asset Pooled Macro Audit ---")
    print(f"  Full Period Net Sharpe: {results['sr']:.2f}")
    print(f"  OOS (2025+) Net Sharpe: {results['sr_oos']:.2f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results['dates'], results['cum_pnl'], label=f'Pooled Portfolio (SR: {results["sr"]:.2f})', color='navy')
    
    oos_start = pd.to_datetime('2025-01-01')
    plt.axvline(oos_start, color='red', linestyle='--', label='OOS Split')
    
    plt.title("Universal Pooled Momentum Transformer: Performance (Net)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Net Returns")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("research/final_backtest_results.png")
    print("\nBacktest plot saved to research/final_backtest_results.png")

if __name__ == "__main__":
    main()
