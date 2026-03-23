import pandas as pd
import numpy as np
import pytest
from data.processor import FeatureProcessor

@pytest.fixture
def toy_data():
    """Generate deterministic toy data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
    # Simple random walk for prices
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, size=300)))
    return pd.DataFrame({'close': prices}, index=dates)

def test_compute_returns(toy_data):
    processor = FeatureProcessor()
    df = processor.compute_returns(toy_data.copy())
    assert 'returns' in df.columns
    assert not df['returns'].isna().all()
    # Check log return formula: log(p1/p0)
    expected = np.log(toy_data['close'].iloc[1] / toy_data['close'].iloc[0])
    np.testing.assert_approx_equal(df['returns'].iloc[1], expected)

def test_compute_vol_scaling(toy_data):
    processor = FeatureProcessor()
    df = processor.compute_returns(toy_data.copy())
    df = processor.compute_vol_scaling(df, window=22)
    
    assert 'sigma_d' in df.columns
    assert 'scaled_returns' in df.columns
    # First 22 rows should have NaN for sigma
    assert df['sigma_d'].iloc[:22].isna().all()
    assert not df['sigma_d'].iloc[22:].isna().any()

def test_compute_macd_signals(toy_data):
    windows = [10, 21]
    processor = FeatureProcessor(window_sizes=windows)
    df = processor.compute_macd_signals(toy_data.copy())
    
    for w in windows:
        col = f'macd_{w}'
        assert col in df.columns
        # Value at index w should not be NaN
        assert not np.isnan(df[col].iloc[w])

def test_process_features(toy_data):
    processor = FeatureProcessor(window_sizes=[10, 21])
    df = processor.process_features(toy_data.copy())
    
    # Check all columns exist
    expected_cols = ['close', 'returns', 'sigma_d', 'scaled_returns', 'macd_10', 'macd_21']
    for col in expected_cols:
        assert col in df.columns
    
    # Check no NaNs remain
    assert not df.isna().any().any()
    # Length should be reduced by the largest window (21) + returns (1) = 22
    assert len(df) == len(toy_data) - 22
