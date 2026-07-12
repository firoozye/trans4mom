import numpy as np
import pandas as pd
import torch
from typing import Tuple

class GPChangepointDetector:
    """
    Gaussian Process based Changepoint Detector (GP-CPD) using Matern 3/2 kernel.
    Compares the marginal log-likelihood of a stationary GP vs a split (two-regime) GP.
    Aligned with Wood et al. (2022).
    """
    def __init__(self, window_len: int = 126, min_regime_len: int = 10, lengthscale: float = 20.0):
        self.window_len = window_len
        self.min_regime_len = min_regime_len
        self.lengthscale = lengthscale
        self.signal_variance = 1.0
        self.noise_variance = 0.5

    def _gp_mll(self, y: torch.Tensor, lengthscale: float) -> float:
        n = len(y)
        if n == 0:
            return 0.0
        t = torch.arange(n, dtype=torch.float32, device=y.device).unsqueeze(1)
        d = torch.abs(t - t.T)
        
        # Matern 3/2 Kernel
        sqrt3 = np.sqrt(3.0)
        K = self.signal_variance * (1.0 + (sqrt3 * d / lengthscale)) * torch.exp(-sqrt3 * d / lengthscale)
        K = K + self.noise_variance * torch.eye(n, device=y.device)
        
        try:
            L = torch.linalg.cholesky(K)
            y_col = y.unsqueeze(1)
            alpha = torch.cholesky_solve(y_col, L)
            fit_term = -0.5 * torch.matmul(y_col.T, alpha)
            complexity_term = -torch.sum(torch.log(torch.diagonal(L)))
            constant_term = -0.5 * n * np.log(2.0 * np.pi)
            return (fit_term + complexity_term + constant_term).item()
        except RuntimeError:
            # Fallback if Cholesky fails (add jitter)
            K = K + 1e-4 * torch.eye(n, device=y.device)
            try:
                L = torch.linalg.cholesky(K)
                y_col = y.unsqueeze(1)
                alpha = torch.cholesky_solve(y_col, L)
                fit_term = -0.5 * torch.matmul(y_col.T, alpha)
                complexity_term = -torch.sum(torch.log(torch.diagonal(L)))
                constant_term = -0.5 * n * np.log(2.0 * np.pi)
                return (fit_term + complexity_term + constant_term).item()
            except RuntimeError:
                return -9999.0

    def compute_severity(self, returns: np.ndarray) -> float:
        n = len(returns)
        if n < self.min_regime_len * 2:
            return 0.0

        # Normalize returns in the window for numerical stability
        ret_mean = np.mean(returns)
        ret_std = np.std(returns) + 1e-9
        norm_returns = (returns - ret_mean) / ret_std

        y = torch.from_numpy(norm_returns).float()

        # H0: Single stationary GP regime
        ll_h0 = self._gp_mll(y, self.lengthscale)

        # H1: Search for optimal split point tau
        max_ll_h1 = -np.inf
        
        # Grid search step optimization for execution speed
        step = 1 if self.window_len <= 30 else (2 if self.window_len <= 100 else 5)
        
        for tau in range(self.min_regime_len, n - self.min_regime_len, step):
            y1 = y[:tau]
            y2 = y[tau:]
            
            # Combined marginal likelihood of independent GP models
            ll_tau = self._gp_mll(y1, self.lengthscale) + self._gp_mll(y2, self.lengthscale)
            
            if ll_tau > max_ll_h1:
                max_ll_h1 = ll_tau

        # Severity = Log Likelihood Ratio
        severity = max_ll_h1 - ll_h0
        return float(max(0.0, severity))

    def run_online(self, series: pd.Series) -> pd.Series:
        """Rolling window severity calculation."""
        return series.rolling(window=self.window_len).apply(
            lambda x: self.compute_severity(x.values),
            raw=False
        )

def add_cpd_features(df: pd.DataFrame, fast_len: int = 21, slow_len: int = 126) -> pd.DataFrame:
    """Add multi-scale GP-CPD severity features to the dataframe."""
    print(f"  Computing GP-CPD features (Fast: {fast_len}, Slow: {slow_len})...")
    
    fast_detector = GPChangepointDetector(window_len=fast_len, min_regime_len=5, lengthscale=5.0)
    slow_detector = GPChangepointDetector(window_len=slow_len, min_regime_len=20, lengthscale=30.0)
    
    def process_group(group):
        group = group.sort_index()
        f_sev = fast_detector.run_online(group['returns'])
        s_sev = slow_detector.run_online(group['returns'])
        
        group[f'cpd_sev_{fast_len}'] = np.sqrt(f_sev.fillna(0))
        group[f'cpd_sev_{slow_len}'] = np.sqrt(s_sev.fillna(0))
        return group

    return df.groupby('symbol', group_keys=False).apply(process_group)

if __name__ == "__main__":
    # Test script execution
    data = np.random.normal(0, 1, 100)
    data[50:] = data[50:] * 3 + 2 # Mean and variance shift
    
    detector = GPChangepointDetector(window_len=100, min_regime_len=10, lengthscale=20.0)
    print(f"GP-CPD Detection Severity: {detector.compute_severity(data):.4f}")
