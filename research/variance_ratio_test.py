import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def variance_ratio_test(data, k):
    """
    Computes the Variance Ratio for a given lag k.
    VR(k) = Var(r_t^k) / (k * Var(r_t^1))
    """
    log_prices = np.log(data)
    rets_1 = log_prices.diff().dropna()
    rets_k = log_prices.diff(k).dropna()
    
    vr = rets_k.var() / (k * rets_1.var())
    return vr

def main():
    df = pd.read_parquet('data/processed/futures_1d.parquet')
    symbols = df['symbol'].unique()
    lags = [2, 5, 10, 21, 63, 126, 252]
    
    results = {}
    print(f"{'Symbol':<10} | " + " | ".join([f"VR({k:3})" for k in lags]))
    print("-" * (13 + len(lags) * 10))

    for sym in symbols:
        asset_df = df[df['symbol'] == sym].sort_index()
        close = asset_df['close']
        
        vr_vals = []
        for k in lags:
            vr = variance_ratio_test(close, k)
            vr_vals.append(vr)
        
        results[sym] = vr_vals
        print(f"{sym:<10} | " + " | ".join([f"{v:6.2f}" for v in vr_vals]))

    # Visualization
    plt.figure(figsize=(12, 6))
    for sym, vr_vals in results.items():
        plt.plot(lags, vr_vals, marker='o', alpha=0.6, label=sym)
    
    plt.axhline(1, color='red', linestyle='--', label='Random Walk')
    plt.xscale('log')
    plt.xlabel('Lag k (Days)')
    plt.ylabel('Variance Ratio VR(k)')
    plt.title('Variance Ratio Test: Evidence of Momentum in Macro Futures')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('research/variance_ratio_tests.png')
    print("\nPlot saved to research/variance_ratio_tests.png")

if __name__ == "__main__":
    main()
