import pandas as pd
import numpy as np
from typing import List, Union

class FeatureProcessor:
    """
    Feature Engineering for Momentum Transformer.
    Includes Vol-scaling and Multi-scale Momentum features.
    """

    def __init__(self, window_sizes: List[int] = [10, 21, 63, 126, 252]):
        self.window_sizes = window_sizes

    def compute_returns(self, df: pd.DataFrame, col: str = 'close') -> pd.DataFrame:
        """Compute daily log returns."""
        df['returns'] = np.log(df[col] / df[col].shift(1))
        return df

    def compute_vol_scaling(self, df: pd.DataFrame, window: int = 22) -> pd.DataFrame:
        """
        Compute rolling 22-day standard deviation and scale returns.
        Scaled Return = r_t / (sigma_t * sqrt(252))
        """
        # Daily rolling volatility
        df['sigma_d'] = df['returns'].rolling(window=window).std()
        
        # Scale to unit annualised volatility
        # Note: Plan says r_t / (sigma_t * sqrt(252))
        df['scaled_returns'] = df['returns'] / (df['sigma_d'] * np.sqrt(window))
        return df

    def compute_macd_signals(self, df: pd.DataFrame, col: str = 'close') -> pd.DataFrame:
        """
        Compute MACD-style trend signals at multiple lookback windows.
        Calculates the ratio of two moving averages.
        """
        for window in self.window_sizes:
            # Simple MACD-like signal: Price relative to rolling average
            df[f'macd_{window}'] = df[col] / df[col].rolling(window=window).mean() - 1
            
            # Alternate: Standard MACD signal (EMA-diff)
            # EMA_fast = df[col].ewm(span=window // 2, adjust=False).mean()
            # EMA_slow = df[col].ewm(span=window, adjust=False).mean()
            # df[f'macd_ema_{window}'] = (EMA_fast - EMA_slow) / df[col]
            
        return df

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature processing pipeline."""
        df = self.compute_returns(df)
        df = self.compute_vol_scaling(df)
        df = self.compute_macd_signals(df)
        
        # Cleanup NaNs from rolling windows
        df.dropna(inplace=True)
        return df

if __name__ == "__main__":
    # Example toy data
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    prices = np.exp(np.cumsum(np.random.normal(0, 0.01, size=500)))
    toy_df = pd.DataFrame({'close': prices}, index=dates)
    
    processor = FeatureProcessor()
    processed_df = processor.process_features(toy_df)
    print("Features processed.")
    print(processed_df.tail())
