import pandas as pd
import numpy as np
from typing import List, Union

class FeatureProcessor:
    """
    Feature Engineering for Momentum Transformer.
    Includes Vol-scaling and Multi-scale Momentum features.
    """

    def __init__(self, window_sizes: List[int] = [10, 21, 63, 126, 252], 
                 outlier_threshold: float = 5.0, 
                 impute_nans: bool = True):
        self.window_sizes = window_sizes
        self.outlier_threshold = outlier_threshold
        self.impute_nans = impute_nans

    def clean_outliers(self, df: pd.DataFrame, col: str = 'returns') -> pd.DataFrame:
        """
        Identify and clip outliers based on a Z-score threshold.
        """
        if col not in df.columns:
            return df
        
        # Calculate Z-score for returns
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        
        # Clip outliers to threshold
        df[col] = df[col].clip(
            lower=df[col].mean() - self.outlier_threshold * df[col].std(),
            upper=df[col].mean() + self.outlier_threshold * df[col].std()
        )
        return df

    def handle_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values via imputation or removal.
        """
        if self.impute_nans:
            # Forward fill followed by backward fill
            df = df.ffill().bfill()
        else:
            # Remove all rows with any NaNs
            df = df.dropna()
        return df

    def compute_returns(self, df: pd.DataFrame, col: str = 'close') -> pd.DataFrame:
        """Compute daily log returns."""
        df['returns'] = np.log(df[col] / df[col].shift(1))
        return df

    def compute_vol_scaling(self, df: pd.DataFrame, window: int = 22) -> pd.DataFrame:
        """
        Compute rolling 22-day standard deviation and scale returns.
        Scaled Return = r_t / (sigma_t * sqrt(window))
        """
        # Daily rolling volatility
        df['sigma_d'] = df['returns'].rolling(window=window).std()
        
        # Scale to unit annualised volatility
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
            
        return df

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature processing pipeline."""
        df = self.compute_returns(df)
        df = self.clean_outliers(df)
        df = self.compute_vol_scaling(df)
        df = self.compute_macd_signals(df)
        df = self.handle_nans(df)
        
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
