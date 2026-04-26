import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    df = pd.read_parquet('data/processed/futures_1d.parquet')
    df.index = pd.to_datetime(df.index)
    oos = df[df.index >= '2025-01-01']
    
    plt.figure(figsize=(12, 6))
    symbols = ['ZN.c.0', 'CL.c.0', 'ES.c.0', 'GC.c.0']
    
    for sym in symbols:
        data = oos[oos['symbol'] == sym].sort_index()
        if not data.empty:
            plt.plot(data.index, data['close'] / data['close'].iloc[0], label=sym)
            
    plt.title('2025 Out-of-Sample Regime Shift: Normalized Asset Performance')
    plt.ylabel('Relative Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('research/oos_regime_autopsy.png')
    print("Autopsy plot saved to research/oos_regime_autopsy.png")

if __name__ == "__main__":
    main()
