import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.architecture import EnsembleMomentumTransformer

def run_backtest_ensemble(models, df, feat_cols, ann_factor, seq_len=252, base_cost=0.002):
    """
    Run backtest for an ensemble of multi-asset cross-sectional models.
    """
    symbols = df['symbol'].unique()
    num_assets = len(symbols)
    # Pivot for multi-asset alignment
    pivoted = df.pivot_table(index=df.index, columns='symbol', values=feat_cols + ['scaled_returns', 'spread'])
    pivoted = pivoted.sort_index().ffill().bfill()

    # Prepare features
    x_list = [pivoted[f].values for f in feat_cols]
    x_vals = np.stack(x_list, axis=-1) # (time, assets, features)
    rets = pivoted['scaled_returns'].values 
    spreads = pivoted['spread'].values 
    dates = pivoted.index

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

    all_pos_ensemble = []
    with torch.no_grad():
        for i, model in enumerate(models):
            print(f"  Generating positions for model {i}...")
            all_pos = []
            for b in range(0, len(x_tensor), 64):
                batch_x = x_tensor[b : b+64].to(next(model.parameters()).device)
                # Static asset_id temporarily disabled in model architecture to match weights
                out = model(batch_x)
                all_pos.append(out[:, -1, :].cpu().numpy())
            all_pos_ensemble.append(np.concatenate(all_pos))
            
    # Ensemble by averaging positions
    positions = np.mean(all_pos_ensemble, axis=0) 

    # Portfolio Returns
    shifted_rets = rets[1:, :]
    active_pos = positions[:-1, :]
    dates_backtest = dates[1:]
    
    pos_diff = np.abs(np.diff(positions, axis=0))
    asset_costs = pos_diff * (base_cost + spreads[1:, :] / 2.0)
    
    gross_rets_per_asset = active_pos * shifted_rets
    net_rets_per_asset = gross_rets_per_asset - asset_costs
    
    net_rets = np.sum(net_rets_per_asset, axis=1) / num_assets
    cum_pnl = np.cumsum(net_rets)
    
    sr_full = (np.mean(net_rets) / (np.std(net_rets) + 1e-9)) * np.sqrt(ann_factor)
    
    # OOS AUDIT
    oos_start = pd.to_datetime('2025-01-01')
    dates_no_tz = dates_backtest.tz_localize(None) if dates_backtest.tz is not None else dates_backtest
    oos_mask = dates_no_tz >= oos_start
    
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
        'symbols': symbols
    }

def main():
    # 1. Setup
    data_path = 'data/processed/futures_full_scale.parquet'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_parquet(data_path)
    
    # FILTER: Ensemble models were trained on 15 assets (excluding YM.c.0)
    original_15 = [
        "ES.c.0", "NQ.c.0", "RTY.c.0", "ZT.c.0", "ZF.c.0", "ZN.c.0", "ZB.c.0",
        "CL.c.0", "NG.c.0", "GC.c.0", "SI.c.0", "HG.c.0", "6E.c.0", "6J.c.0", "6B.c.0"
    ]
    df = df[df['symbol'].isin(original_15)]
    
    ann_factor = 252 
    # Match the 8 features expected by legacy weights: 5 ret_ + 3 macd_
    feat_cols = [
        'ret_1', 'ret_21', 'ret_63', 'ret_126', 'ret_252',
        'macd_8_24', 'macd_16_48', 'macd_32_96'
    ]
    
    symbols = df['symbol'].unique()
    num_assets = len(symbols)
    
    # 2. Load Ensemble of Models (matching checkpoint: hidden_dim: 64, input_dim: 8)
    seeds = 5
    models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for seed in range(seeds):
        weight_path = f'weights/model_ensemble_seed_{seed}.pt'
        if os.path.exists(weight_path):
            model = EnsembleMomentumTransformer(
                input_dim=len(feat_cols), 
                num_vars=num_assets, 
                hidden_dim=64, 
                num_heads=4, 
                output_dim=num_assets
            )
            # Load weights - architecture.py has been patched to skip static_embedding
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)
    
    if not models:
        print("No ensemble models found.")
        return
    
    print(f"--- Running Legacy Ensemble Backtest ({len(models)} models, {num_assets} assets) ---")
    results = run_backtest_ensemble(models, df, feat_cols, ann_factor, seq_len=252, base_cost=0.002)

    print(f"\n--- {num_assets}-Asset Macro Ensemble Audit ---")
    print(f"  Full Period Net Sharpe: {results['sr']:.2f}")
    print(f"  OOS (2025+) Net Sharpe: {results['sr_oos']:.2f}")
    print(f"  Annualized Turnover:    {results['turnover']:.1f}x")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results['dates'], results['cum_pnl'], label=f'Ensemble Portfolio (SR: {results["sr"]:.2f})', color='purple')
    
    oos_start = pd.to_datetime('2025-01-01')
    plt.axvline(oos_start, color='red', linestyle='--', label='OOS Split')
    
    plt.title("Cross-Sectional Ensemble Momentum Performance (Net)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("research/backtest_cs_ensemble.png")
    print("\nEnsemble plot saved to research/backtest_cs_ensemble.png")

if __name__ == "__main__":
    main()
