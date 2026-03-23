import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from data.ingestion import DataIngestor

def test_load_local_csv(tmp_path):
    """Verify local CSV loading works."""
    d = tmp_path / "data"
    d.mkdir()
    file = d / "test.csv"
    df_orig = pd.DataFrame({"close": [100, 101]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    df_orig.index.name = "timestamp"
    df_orig.to_csv(file)
    
    ingestor = DataIngestor()
    df_loaded = ingestor.load_local_csv(str(file))
    
    assert len(df_loaded) == 2
    assert "close" in df_loaded.columns

@patch("ccxt.coinbase")
def test_fetch_coinbase_ccxt(mock_coinbase):
    """Mock CCXT to verify data formatting."""
    mock_exchange = MagicMock()
    # Mock return: [timestamp, open, high, low, close, volume]
    mock_exchange.fetch_ohlcv.return_value = [
        [1609459200000, 29000, 29500, 28000, 29200, 100],
    ]
    mock_coinbase.return_value = mock_exchange
    
    ingestor = DataIngestor()
    df = ingestor.fetch_coinbase_ccxt(symbol="BTC/USDT")
    
    assert len(df) == 1
    assert df.index.name == "timestamp"
    assert df["close"].iloc[0] == 29200

@patch("yfinance.download")
def test_fetch_yfinance(mock_yf):
    """Mock yfinance download."""
    mock_yf.return_value = pd.DataFrame({"Close": [100]}, index=pd.to_datetime(["2020-01-01"]))
    
    ingestor = DataIngestor()
    df = ingestor.fetch_yfinance(["AAPL"], "2020-01-01", "2020-01-02")
    
    assert not df.empty
    assert "Close" in df.columns
