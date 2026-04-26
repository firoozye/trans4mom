import os
import pandas as pd
import numpy as np
import databento as db
import ccxt
import yfinance as yf
from datetime import datetime
from typing import List, Optional, Union

class DataIngestor:
    """
    Unified Data Ingestion for Momentum Transformer Project.
    Supports DataBento (institutional), CCXT (Coinbase), and yfinance (Free).
    """

    def __init__(self, databento_api_key: Optional[str] = None):
        self.db_api_key = databento_api_key or os.getenv("DATABENTO_API_KEY")
        self.db_client = db.Historical(self.db_api_key) if self.db_api_key else None

    def fetch_databento(
        self,
        symbols: List[str],
        start: str,
        end: str,
        dataset: str = "GLBX.MDP3",
        schema: str = "ohlcv-1h",
        stype_in: str = "continuous"
    ) -> pd.DataFrame:
        """Fetch historical data from DataBento."""
        if not self.db_client:
            raise ValueError("DataBento API key required.")

        data = self.db_client.timeseries.get_range(
            dataset=dataset,
            symbols=symbols,
            schema=schema,
            start=start,
            end=end,
            stype_in=stype_in
        )
        df = data.to_df()
        return df

    def fetch_ccxt_ohlcv(
        self,
        exchange_id: str = 'binance',
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        since: Optional[str] = None,
        limit: int = 1000,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV from any CCXT-supported exchange.
        since: ISO8601 string (e.g., '2023-01-01T00:00:00Z')
        """
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'enableRateLimit': True})
        
        since_ms = exchange.parse8601(since) if since else None
        all_ohlcv = []
        
        print(f"Fetching {symbol} from {exchange_id} ({timeframe})...")
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_ms, limit)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                
                # Update since_ms to the last timestamp + 1ms to get the next batch
                since_ms = ohlcv[-1][0] + 1
                
                # If we got fewer than limit, we've reached the end
                if len(ohlcv) < limit:
                    break
                    
                print(f"  Downloaded up to {exchange.iso8601(since_ms)}")
            except Exception as e:
                print(f"Error: {e}")
                break

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    # --- OLD CODE (Commented out per instructions) ---
    # def fetch_yfinance(
    #     self,
    #     symbols: List[str],
    #     start: str,
    #     end: str,
    #     interval: str = "1d"
    # ) -> pd.DataFrame:
    #     """Fetch historical data from yfinance (Free)."""
    #     df = yf.download(symbols, start=start, end=end, interval=interval)
    #     return df

    def fetch_yfinance(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data from yfinance (Free).
        Standardizes columns to lowercase to match the rest of the pipeline.
        """
        # Cleanup timestamp if it has the ISO T...Z suffix which yfinance dislikes
        clean_start = start.split('T')[0] if 'T' in start else start
        clean_end = end.split('T')[0] if 'T' in end else end
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=clean_start, end=clean_end, interval=interval)
        
        if df.empty:
            return df
            
        # Standardize columns to lowercase
        df.columns = [c.lower() for c in df.columns]
        
        # Ensure 'symbol' column is present
        df['symbol'] = symbol
        
        return df

    def load_local_csv(self, file_path: str) -> pd.DataFrame:
        """Load local data from Bloomberg or other sources."""
        df = pd.read_csv(file_path, parse_dates=True, index_index=0)
        return df

if __name__ == "__main__":
    pass
