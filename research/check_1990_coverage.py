import yfinance as yf
import pandas as pd

symbols = {
    "ES=F": "S&P 500", "NQ=F": "Nasdaq 100", "RTY=F": "Russell 2000", "YM=F": "Dow 30",
    "ZT=F": "2yr Note", "ZF=F": "5yr Note", "ZN=F": "10yr Note", "ZB=F": "30yr Bond",
    "CL=F": "Crude Oil", "NG=F": "Natural Gas", "RB=F": "Gasoline", "HO=F": "Heating Oil",
    "GC=F": "Gold", "SI=F": "Silver", "HG=F": "Copper",
    "ZC=F": "Corn", "ZW=F": "Wheat", "ZS=F": "Soybeans",
    "6E=F": "EUR/USD", "6J=F": "JPY/USD", "6B=F": "GBP/USD", "6A=F": "AUD/USD", "6C=F": "CAD/USD", "6S=F": "CHF/USD"
}

results = []
for sym, name in symbols.items():
    print(f"Checking {sym} ({name})...")
    ticker = yf.Ticker(sym)
    hist = ticker.history(start="1990-01-01")
    if not hist.empty:
        results.append({
            "Symbol": sym,
            "Name": name,
            "Start": hist.index.min().strftime('%Y-%m-%d'),
            "End": hist.index.max().strftime('%Y-%m-%d'),
            "Rows": len(hist)
        })
    else:
        results.append({"Symbol": sym, "Name": name, "Start": "N/A", "End": "N/A", "Rows": 0})

df = pd.DataFrame(results)
print("\n--- yfinance Coverage Since 1990 ---")
print(df.to_string(index=False))
