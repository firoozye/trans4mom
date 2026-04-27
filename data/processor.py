import pandas as pd
import numpy as np
from typing import List, Optional

class FeatureProcessor:
    """
    Academic-grade feature processor aligned with Wood et al. (2022).
    Implements 60-day EWMA volatility scaling and multi-scale MACD crossovers.
    """
    def __init__(
        self, 
        return_windows: List[int] = [1, 21, 63, 126, 252],
        macd_pairs: List[tuple] = [(8, 24), (16, 48), (32, 96)],
        vol_span: int = 60
    ):
        self.return_windows = return_windows
        self.macd_pairs = macd_pairs
        self.vol_span = vol_span

    def compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """60-day EWMA volatility as per Lim et al. (2019). Lagged by 1 for ex-ante."""
        # Shift(1) is critical to ensure no lookahead bias in volatility estimation
        df['sigma_d'] = df['returns'].ewm(span=self.vol_span).std().shift(1)
        return df

    def compute_normalized_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns scaled by vol and sqrt(k) for multi-scale stationarity."""
        for k in self.return_windows:
            # Cumulative log returns over window k
            roll_ret = df['returns'].rolling(window=k).sum()
            df[f'ret_{k}'] = roll_ret / (df['sigma_d'] * np.sqrt(k) + 1e-9)
        return df

    def compute_macd_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility-normalized EWMA crossovers."""
        for S, L in self.macd_pairs:
            ema_s = df['close'].ewm(span=S).mean()
            ema_l = df['close'].ewm(span=L).mean()
            macd = ema_s - ema_l
            # Normalize by daily vol to ensure scale-invariance
            df[f'macd_{S}_{L}'] = macd / (df['sigma_d'] * df['close'] + 1e-9)
        return df

    def compute_additional_priors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Range and Mean features for broader prior modulation."""
        # 1. Vol-scaled Range: (High - Low) / Sigma
        # Normalized by price to keep it scale-invariant
        df['range_60'] = (df['high'] - df['low']).rolling(60).mean() / (df['sigma_d'] * df['close'] + 1e-9)
        
        # 2. Vol-scaled Mean: Rolling average of returns / Sigma
        df['mean_60'] = df['returns'].rolling(60).mean() / (df['sigma_d'] + 1e-9)
        
        # 3. Target Priors (The baseline signals to be modulated)
        # We use the 252-day return and the average of MACD signals as our two "priors"
        df['prior_mom'] = df['ret_252']
        macd_cols = [f'macd_{S}_{L}' for S, L in self.macd_pairs]
        df['prior_trend'] = df[macd_cols].mean(axis=1)
        
        return df

    def process_features(self, df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
        df = df.sort_index()
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 1. Asset ID Mapping from Config
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else "UNKNOWN"
        asset_id = 4 # Default category
        
        if config and 'asset_map' in config:
            for idx, (category, syms) in enumerate(config['asset_map'].items()):
                if symbol in syms:
                    asset_id = idx
                    break
        df['asset_id'] = asset_id
        
        # 2. Volatility and Multi-scale Returns
        df = self.compute_volatility(df)
        
        # Outlier cleaning for robust training
        if config and config.get('features', {}).get('clean_outliers', False):
            thresh = config['features'].get('outlier_threshold', 5.0)
            df['returns'] = df['returns'].clip(lower=-thresh*df['sigma_d'], upper=thresh*df['sigma_d'])

        df = self.compute_normalized_returns(df)
        df = self.compute_macd_signals(df)
        df = self.compute_additional_priors(df)
        
        # 3. Volatility Target Scaling (Target: 15% Annualized)
        sigma_tgt = 0.15 / np.sqrt(252)
        df['scaled_returns'] = df['returns'] * (sigma_tgt / (df['sigma_d'] + 1e-9))
        
        # 4. Feature Set and Cleanup
        # Simple bid-ask spread estimation from High-Low range
        mid = (df['high'] + df['low']) / 2.0
        df['spread'] = np.abs(df['close'] - mid) / (mid + 1e-9)
        
        # Impute NaNs if requested
        if config and config.get('features', {}).get('impute_nans', False):
            df = df.ffill().fillna(0)
            return df
            
        return df.dropna()

if __name__ == "__main__":
    print("FeatureProcessor configured for paper-aligned reproduction with Broader Priors.")
