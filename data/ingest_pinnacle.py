#!/usr/bin/env python3
"""
Ingest Pinnacle Data Corp Ratio-Adjusted (.RAD) Continuous Contracts
and generate ex-ante features aligned with Wood, Roberts & Zohren (2022).
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

# Core balanced portfolio of 50 liquid futures across 4 asset classes
PINNACLE_ASSET_MAP = {
    # Equities (Asset ID 0)
    "EQUITY": [
        "ES", "SP", "EN", "ND", "YM", "DJ", "ER", "RL", "MD",
        "NK", "LX", "AX", "CA", "HS", "AP", "XU"
    ],
    # Fixed Income (Asset ID 1)
    "FIXED_INCOME": [
        "US", "TY", "FB", "TU", "FF", "ED", "DT", "UB", "UZ", "GS", "CB"
    ],
    # Energy & Metals (Commodities - Asset ID 2)
    "ENERGY_METALS": [
        "CL", "BC", "HO", "RB", "NG", "BG",
        "GC", "SI", "HG", "PL", "PA"
    ],
    # Agriculture (Commodities - Asset ID 3)
    "AGRICULTURE": [
        "C_", "W_", "KW", "S_", "BO", "SM", "CC", "KC", "CT", "SB", "JO", "LC", "FC", "LH"
    ],
    # Foreign Exchange (Asset ID 4)
    "FX": [
        "FN", "BN", "JN", "AN", "CN", "SN", "MP", "DX"
    ]
}

def get_asset_id_and_category(ticker: str):
    for cat_idx, (cat, symbols) in enumerate(PINNACLE_ASSET_MAP.items()):
        if ticker in symbols:
            return cat_idx, cat
    return 2, "OTHER" # Default to commodity

def process_single_rad_file(filepath: str, return_windows=[1, 21, 63, 126, 252], macd_pairs=[(8, 24), (16, 48), (32, 96)], vol_span=60):
    ticker = os.path.basename(filepath).replace(".RAD", "")
    
    # Format: YYYYMMDD, Open, High, Low, Close, Volume, OpenInterest
    df = pd.read_csv(
        filepath,
        header=None,
        names=["Date", "open", "high", "low", "close", "volume", "open_interest"],
        dtype={"Date": str, "open": float, "high": float, "low": float, "close": float, "volume": float, "open_interest": float}
    )
    
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"])
    df = df.set_index("Date")
    
    # Filter valid trading days from 1985 onwards (to give warm-up before 1990)
    df = df.loc["1985-01-01":]
    if len(df) < 300:
        return None
        
    df["symbol"] = ticker
    asset_id, category = get_asset_id_and_category(ticker)
    df["asset_id"] = asset_id
    df["category"] = category
    
    # Returns (log returns)
    df["returns"] = np.log(df["close"] / df["close"].shift(1))
    
    # 60-day EWMA volatility (ex-ante)
    df["sigma_d"] = df["returns"].ewm(span=vol_span).std()
    
    # Outlier clipping at 5 standard deviations
    thresh = 5.0
    df["returns"] = df["returns"].clip(lower=-thresh * df["sigma_d"], upper=thresh * df["sigma_d"])
    
    # Volatility target scaling (15% annualised)
    sigma_tgt = 0.15 / np.sqrt(252)
    df["vol_scale"] = sigma_tgt / (df["sigma_d"] + 1e-9)
    df["scaled_returns"] = df["returns"] * df["vol_scale"]
    
    # Multi-scale normalized returns
    for k in return_windows:
        roll_ret = df["returns"].rolling(window=k).sum()
        df[f"ret_{k}"] = roll_ret / (df["sigma_d"] * np.sqrt(k) + 1e-9)
        
    # Multi-scale normalized MACDs (Baz et al. 2015)
    macd_cols = []
    for S, L in macd_pairs:
        ema_s = df["close"].ewm(span=S).mean()
        ema_l = df["close"].ewm(span=L).mean()
        macd = (ema_s - ema_l) / (df["sigma_d"] * df["close"] + 1e-9)
        macd_std = macd.rolling(252).std() + 1e-9
        col = f"macd_{S}_{L}"
        df[col] = macd / macd_std
        macd_cols.append(col)
        
    # Classical TSMOM Prior: average of normalized MACDs
    df["prior_mom"] = df[macd_cols].mean(axis=1)
    
    # Spread proxy
    mid = (df["high"] + df["low"]) / 2.0
    df["spread"] = np.abs(df["close"] - mid) / (mid + 1e-9)
    
    # Changepoint Severity Proxies (Rolling volatility shocks)
    df["cpd_sev_21"] = np.maximum(0.0, (df["returns"].rolling(21).std() / (df["sigma_d"] + 1e-9)) - 1.0)
    df["cpd_sev_126"] = np.maximum(0.0, (df["returns"].rolling(126).std() / (df["sigma_d"] + 1e-9)) - 1.0)
    
    # Drop warm-up NaNs
    df = df.dropna()
    
    # Keep only data from 1990 onwards
    df = df.loc["1990-01-01":]
    return df

def build_pinnacle_dataset(clc_dir: str, output_parquet: str):
    all_tickers = []
    for cat, syms in PINNACLE_ASSET_MAP.items():
        all_tickers.extend(syms)
        
    print(f"Scanning {len(all_tickers)} target tickers in {clc_dir}...")
    dfs = []
    
    for ticker in all_tickers:
        fpath = os.path.join(clc_dir, f"{ticker}.RAD")
        if not os.path.exists(fpath):
            print(f"  Warning: {ticker}.RAD not found, skipping.")
            continue
            
        processed_df = process_single_rad_file(fpath)
        if processed_df is not None and len(processed_df) > 500:
            print(f"  Loaded {ticker:5s} ({processed_df['category'].iloc[0]:15s}): {len(processed_df):5,d} bars ({processed_df.index.min().strftime('%Y-%m-%d')} to {processed_df.index.max().strftime('%Y-%m-%d')})")
            dfs.append(processed_df)
            
    if not dfs:
        raise RuntimeError("No files were successfully processed!")
        
    combined = pd.concat(dfs, axis=0).sort_index()
    print(f"\nSuccessfully combined {len(dfs)} assets: {len(combined):,d} total records.")
    print(f"Date range: {combined.index.min().strftime('%Y-%m-%d')} to {combined.index.max().strftime('%Y-%m-%d')}")
    
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    combined.to_parquet(output_parquet)
    print(f"Saved dataset to {output_parquet} ({os.path.getsize(output_parquet) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    CLC_DIR = "/cs/research/fca/home0/firoozye/dev/trans4mom/data/PinnacleData/CLCDATA"
    OUT_PARQUET = "/cs/research/fca/home0/firoozye/dev/trans4mom/data/processed/pinnacle_futures_1990_2026.parquet"
    build_pinnacle_dataset(CLC_DIR, OUT_PARQUET)
