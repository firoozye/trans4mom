import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.architecture import EnsembleMomentumTransformer

def visualize_attention(model, df, symbol, feat_cols, seq_len=252):
    device = next(model.parameters()).device
    model.eval()
    
    # 1. Prepare Data for the specific symbol (OOS period)
    group = df[df['symbol'] == symbol].sort_index()
    # Take the very end of the data
    group = group.tail(seq_len + 100) 
    
    # Mapping for Asset IDs (matching train_hpc.py)
    asset_id_map = {
        "ES=F": 0, "NQ=F": 0, "RTY=F": 0, "YM=F": 0, "^N225": 0, "^FTSE": 0, "^GDAXI": 0, "^FCHI": 0, "^HSI": 0, "^AXJO": 0, "^SSMI": 0,
        "ZT=F": 1, "ZF=F": 1, "ZN=F": 1, "ZB=F": 1,
        "CL=F": 2, "NG=F": 2, "RB=F": 2, "HO=F": 2, "BZ=F": 2, "QM=F": 2,
        "GC=F": 2, "SI=F": 2, "HG=F": 2, "PL=F": 2, "PA=F": 2,
        "ZC=F": 3, "ZW=F": 3, "ZS=F": 3, "CC=F": 3, "KC=F": 3, "CT=F": 3, "SB=F": 3, "OJ=F": 3, "LE=F": 3, "GF=F": 3, "HE=F": 3,
        "6E=F": 4, "6J=F": 4, "6B=F": 4, "6A=F": 4, "6C=F": 4, "6S=F": 4, "6N=F": 4, "6M=F": 4, "6L=F": 4
    }
    aid = asset_id_map.get(symbol, 0)

    # Prepare single window
    x_vals = group[feat_cols].values[-seq_len:].astype(np.float32)
    pm = group['prior_mom'].values[-seq_len:].reshape(-1, 1).astype(np.float32)
    mon = group.index.month.values[-seq_len:].astype(np.int64)
    day = group.index.dayofweek.values[-seq_len:].astype(np.int64)
    dates = group.index[-seq_len:]

    bx = torch.from_numpy(x_vals).unsqueeze(0).to(device)
    bpm = torch.from_numpy(pm).unsqueeze(0).to(device)
    bmon = torch.from_numpy(mon).unsqueeze(0).to(device)
    bday = torch.from_numpy(day).unsqueeze(0).to(device)
    baid = torch.tensor([aid], dtype=torch.long).to(device)

    with torch.no_grad():
        pos, attn = model(bx, asset_id=baid, month=bmon, day=bday, prior_mom=bpm, return_attention=True)
    
    # attn shape: (1, T, T)
    attn_map = attn[0].cpu().numpy()
    
    # Plotting
    plt.figure(figsize=(12, 10))
    # We focus on the last prediction's attention (how it looks back)
    # or the full causal map
    sns.heatmap(attn_map, cmap='viridis')
    plt.title(f"Causal Attention Map: {symbol}")
    plt.xlabel("Lookback (Days)")
    plt.ylabel("Prediction Step")
    
    # Also plot the attention weights for the VERY LAST prediction
    plt.figure(figsize=(12, 4))
    plt.plot(dates, attn_map[-1, :], color='blue', label='Attention Weights (Last Prediction)')
    plt.title(f"What is the model looking at for {symbol} on {dates[-1].date()}?")
    plt.ylabel("Attention Weight")
    plt.grid(True, alpha=0.3)
    
    # Overlay the returns to see if it's looking at volatility
    ax2 = plt.gca().twinx()
    ax2.fill_between(dates, 0, group['scaled_returns'].values[-seq_len:], color='gray', alpha=0.2, label='Returns')
    ax2.set_ylabel("Scaled Returns")
    
    plt.tight_layout()
    plt.savefig(f"research/attention_{symbol.replace('=', '_')}.png")
    print(f"Saved attention visualization for {symbol}")

def main():
    # Load Model (Reproduction V4 - the one that overfitted slightly)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_cols = [f'ret_{k}' for k in [1, 21, 63, 126, 252]] + \
                [f'macd_{S}_{L}' for S, L in [[8, 24], [16, 48], [32, 96]]]
    
    model = EnsembleMomentumTransformer(
        input_dim=1, num_vars=8, hidden_dim=64, num_heads=4, num_layers=1, dropout=0.1, output_dropout=0.1, num_static_vars=5
    ).to(device)
    
    weight_path = 'weights/reproduction_v3_best.pt'
    if not os.path.exists(weight_path):
        print("Best weights not found.")
        return
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    
    df = pd.read_parquet('data/processed/futures_1990_present.parquet')
    
    # Visualize for a few key assets
    visualize_attention(model, df, "ES=F", feat_cols)
    visualize_attention(model, df, "GC=F", feat_cols)
    visualize_attention(model, df, "CL=F", feat_cols)

if __name__ == "__main__":
    main()
