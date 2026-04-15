import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data.ingestion import DataIngestor

def calculate_variance_ratio(returns, q):
    """
    VR(q) = Var(r_t + ... + r_{t-q+1}) / (q * Var(r_t))
    VR > 1 implies Momentum
    VR < 1 implies Mean Reversion
    """
    returns = returns.dropna()
    if len(returns) < q*2:
        return np.nan
    
    sig_1 = returns.var(ddof=1)
    q_returns = returns.rolling(window=q).sum().dropna()
    sig_q = q_returns.var(ddof=1)
    
    vr = sig_q / (q * sig_1)
    return vr

def main():
    ingestor = DataIngestor()
    symbol = 'BTC/USDT'
    timeframes = ['1h', '4h', '6h', '12h', '1d']
    since = '2022-01-01T00:00:00Z'
    
    # Map timeframe strings to periods per day
    tf_to_ppd = {
        '1h': 24,
        '4h': 6,
        '6h': 4,
        '12h': 2,
        '1d': 1
    }
    
    # Lags in DAYS (normalized)
    # 6hr = 0.25d, 12hr = 0.5d
    days_lags = [0.25, 0.5, 1, 2, 3, 4, 5, 10, 15, 20]
    
    results = []
    
    print(f"--- Granular Multi-Frequency Momentum Scan for {symbol} ---")
    
    for tf in timeframes:
        ppd = tf_to_ppd[tf]
        print(f"Processing {tf}...")
        try:
            # Fetch more data to avoid NaNs at long lags (5000 bars)
            df = ingestor.fetch_ccxt_ohlcv(symbol=symbol, timeframe=tf, since=since, limit=5000)
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
            rets = df['returns'].dropna()
            
            tf_vrs = {'TF': tf}
            for days in days_lags:
                q = int(days * ppd)
                # If q is 0 (e.g. 6hr lag on 1d data), we mark as NaN or skip
                if q < 2:
                    tf_vrs[f'{days}d'] = np.nan
                else:
                    tf_vrs[f'{days}d'] = calculate_variance_ratio(rets, q)
            
            results.append(tf_vrs)
        except Exception as e:
            print(f"  Error fetching {tf}: {e}")

    # Display Results
    res_df = pd.DataFrame(results)
    # Rename columns for clarity in the table
    column_map = {f'{d}d': f'{int(d*24)}h' if d < 1 else f'{int(d)}d' for d in days_lags}
    res_df.rename(columns=column_map, inplace=True)
    
    print("\nVariance Ratio Table (VR > 1 is Momentum):")
    print(res_df.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))
    
    # Plotting
    plt.figure(figsize=(14, 7))
    for col in res_df.columns[1:]:
        plt.plot(res_df['TF'], res_df[col], marker='o', label=col)
    
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Random Walk')
    plt.title(f"Momentum Horizon Scan ({symbol})")
    plt.xlabel("Data Frequency")
    plt.ylabel("Variance Ratio (Momentum > 1)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("research/granular_momentum_scan.png")
    print("\nPlot saved to research/granular_momentum_scan.png")

if __name__ == "__main__":
    main()
