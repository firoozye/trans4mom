import numpy as np
import pandas as pd
from typing import Tuple

class ChangepointDetector:
    """
    Generalized Likelihood Ratio (GLR) based Changepoint Detector.
    Identifies structural breaks in mean and variance of returns.
    Aligned with the 'Regime Awareness' requirements of Wood et al. (2022).
    """
    def __init__(self, window_len: int = 126, min_regime_len: int = 10):
        self.window_len = window_len
        self.min_regime_len = min_regime_len

    def compute_severity(self, returns: np.ndarray) -> float:
        """
        Calculates the changepoint severity (Log-Likelihood Ratio).
        Compares: H0 (Single stationary regime) vs H1 (Two regimes).
        """
        n = len(returns)
        if n < self.min_regime_len * 2:
            return 0.0

        # H0: Constant Mean and Variance
        mu_full = np.mean(returns)
        var_full = np.var(returns) + 1e-9
        ll_h0 = -0.5 * n * (np.log(2 * np.pi * var_full) + 1)

        # H1: Search for optimal split point tau
        max_ll_h1 = -np.inf
        
        for tau in range(self.min_regime_len, n - self.min_regime_len):
            r1 = returns[:tau]
            r2 = returns[tau:]
            
            var1 = np.var(r1) + 1e-9
            var2 = np.var(r2) + 1e-9
            
            ll_tau = -0.5 * (
                tau * (np.log(2 * np.pi * var1) + 1) + 
                (n - tau) * (np.log(2 * np.pi * var2) + 1)
            )
            
            if ll_tau > max_ll_h1:
                max_ll_h1 = ll_tau

        # Severity = Log Likelihood Ratio
        severity = max_ll_h1 - ll_h0
        return max(0.0, severity)

    def run_online(self, series: pd.Series) -> pd.Series:
        """Vectorized-ish rolling severity calculation."""
        # Use a rolling window and apply the severity calculation
        return series.rolling(window=self.window_len).apply(
            lambda x: self.compute_severity(x.values),
            raw=False
        )

def add_cpd_features(df: pd.DataFrame, fast_len: int = 21, slow_len: int = 126) -> pd.DataFrame:
    """Add multi-scale CPD severity features to the dataframe."""
    print(f"  Computing CPD features (Fast: {fast_len}, Slow: {slow_len})...")
    
    fast_detector = ChangepointDetector(window_len=fast_len, min_regime_len=5)
    slow_detector = ChangepointDetector(window_len=slow_len, min_regime_len=20)
    
    # Apply per symbol
    def process_group(group):
        group = group.sort_index()
        # Scale severity to roughly [0, 1] using a softplus or normalization
        # In practice, we take the sqrt of the LLR as per standard detection theory
        f_sev = fast_detector.run_online(group['returns'])
        s_sev = slow_detector.run_online(group['returns'])
        
        group[f'cpd_sev_{fast_len}'] = np.sqrt(f_sev.fillna(0))
        group[f'cpd_sev_{slow_len}'] = np.sqrt(s_sev.fillna(0))
        return group

    return df.groupby('symbol', group_keys=False).apply(process_group)

if __name__ == "__main__":
    # Quick test
    data = np.random.normal(0, 1, 100)
    data[50:] = data[50:] * 3 + 2 # Massive shift
    
    detector = ChangepointDetector(window_len=100)
    print(f"Detection Severity: {detector.compute_severity(data):.4f}")
