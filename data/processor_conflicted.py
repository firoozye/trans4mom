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
        df['sigma_d'] = df['returns'].ewm(span=self.vol_span).std().shift(1)
        return df

    def compute_normalized_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns scaled by vol and sqrt(k) for multi-scale stationarity."""
        for k in self.return_windows:
            # We must shift(1) here to ensure these features are available at time t
            # to decide the position for t -> t+1.
            roll_ret = df['returns'].rolling(window=k).sum().shift(1)
            df[f'ret_{k}'] = roll_ret / (df['sigma_d'] * np.sqrt(k) + 1e-9)
        return df

    def compute_macd_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility-normalized EWMA crossovers. Lagged for ex-ante."""
        for S, L in self.macd_pairs:
            ema_s = df['close'].ewm(span=S).mean()
            ema_l = df['close'].ewm(span=L).mean()
            macd = ema_s - ema_l
            # Shift(1) ensures no lookahead
            df[f'macd_{S}_{L}'] = (macd / (df['sigma_d'] * df['close'] + 1e-9)).shift(1)
        return df

    def compute_additional_priors(self, df: pd.DataFrame) -> pd.DataFrame:
<<<<<<< HEAD
        """Add baseline TSMOM signal as the sole prior modulation. Lagged for ex-ante."""
        # Baseline Prior (TSMOM) = 12-month return (ret_252)
        df['prior_mom'] = df['ret_252']
=======
        """Add Range and Mean features for broader prior modulation. Lagged for ex-ante."""
        # 1. Vol-scaled Range (Lagged)
        df['range_60'] = ((df['high'] - df['low']).rolling(60).mean() / (df['sigma_d'] * df['close'] + 1e-9)).shift(1)
        
        # 2. Vol-scaled Mean (Lagged)
        df['mean_60'] = (df['returns'].rolling(60).mean() / (df['sigma_d'] + 1e-9)).shift(1)
        
        # 3. Target Priors (Lagged)
        # prior_mom uses the already lagged ret_252
        df['prior_mom'] = df['ret_252']
        
        # prior_trend uses the already lagged macd columns
        macd_cols = [f'macd_{S}_{L}' for S, L in self.macd_pairs]
        df['prior_trend'] = df[macd_cols].mean(axis=1)
        
>>>>>>> e6b1e3150b63b7e8eeeef3b6e063fde8a5a64696
        return df

    def process_features(self, df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
        df = df.sort_index()
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 1. Asset ID Mapping
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else "UNKNOWN"
        asset_id = 4
        if config and 'asset_map' in config:
            for idx, (category, syms) in enumerate(config['asset_map'].items()):
                if symbol in syms:
                    asset_id = idx
                    break
        df['asset_id'] = asset_id
        
        # 2. Volatility
        df = self.compute_volatility(df)
        
        # Outlier cleaning (Use raw returns for cleaning to avoid circularity)
        if config and config.get('features', {}).get('clean_outliers', False):
            thresh = config['features'].get('outlier_threshold', 5.0)
            df['returns'] = df['returns'].clip(lower=-thresh*df['sigma_d'], upper=thresh*df['sigma_d'])

        # 3. Signals (All now lagged inside methods)
        df = self.compute_normalized_returns(df)
        df = self.compute_macd_signals(df)
        df = self.compute_additional_priors(df)
        
        # 4. Volatility Target Scaling (Target: 15% Annualized)
        sigma_tgt = 0.15 / np.sqrt(252)
        df['scaled_returns'] = df['returns'] * (sigma_tgt / (df['sigma_d'] + 1e-9))
        
        # 5. Spread Estimation
        mid = (df['high'] + df['low']) / 2.0
        df['spread'] = np.abs(df['close'] - mid) / (mid + 1e-9)
        
        if config and config.get('features', {}).get('impute_nans', False):
            df = df.ffill().fillna(0)
            return df
            
        return df.dropna()

if __name__ == "__main__":
    print("FeatureProcessor: EX-ANTE SIGNALS ENABLED (Lookahead Fixed).")
