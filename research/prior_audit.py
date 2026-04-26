import pandas as pd
import numpy as np

def main():
    df = pd.read_parquet('data/processed/futures_1d.parquet')
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    symbols = df['symbol'].unique()
    feat_cols = [c for c in df.columns if 'macd' in c]
    
    all_port_rets = []
    
    for sym in symbols:
        asset_df = df[df['symbol'] == sym].sort_index()
        # OOS only
        asset_oos = asset_df[asset_df.index >= '2025-01-01']
        if asset_oos.empty: continue
        
        # Signal = mean of standardized trends
        signal = asset_oos[feat_cols].mean(axis=1)
        rets = asset_oos['returns']
        
        # Alignment: signal at t captures return at t+1
        pos = np.sign(signal.shift(1)).fillna(0)
        asset_rets = pos * rets
        all_port_rets.append(asset_rets)
        
    # Aggregate portfolio
    port_df = pd.concat(all_port_rets, axis=1).mean(axis=1)
    port_rets = port_df.values
    
    sr = (np.mean(port_rets) / (np.std(port_rets) + 1e-9)) * np.sqrt(252)
    print(f"Definitive OOS Static TSMOM Prior Sharpe (2025+): {sr:.2f}")

if __name__ == "__main__":
    main()
