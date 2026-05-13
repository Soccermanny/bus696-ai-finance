"""
Fetch historical prices for stablecoins and save as CSVs.
Simplified version with better error handling.
"""

import yfinance as yf
import pandas as pd
import os

# Apply pandas groupby compatibility patch
from pandas.core.groupby import DataFrameGroupBy

_original_apply = DataFrameGroupBy.apply

def _patched_apply(self, func, *args, **kwargs):
    """Preserve groupby keys in the group dataframe for newer pandas versions."""
    grouper = self.grouper
    group_keys = [g for g in grouper.names if g is not None]
    
    if not group_keys:
        return _original_apply(self, func, *args, **kwargs)
    
    results = []
    for key, group in self:
        if isinstance(key, tuple):
            for k, v in zip(group_keys, key):
                group[k] = v
        else:
            group[group_keys[0]] = key
        
        result = func(group, *args, **kwargs)
        results.append(result)
    
    if results and isinstance(results[0], pd.DataFrame):
        return pd.concat(results, ignore_index=True)
    return _original_apply(self, func, *args, **kwargs)

DataFrameGroupBy.apply = _patched_apply

# Create data directory if it doesn't exist
data_dir = 'data/stablecoins'
os.makedirs(data_dir, exist_ok=True)

# Define coins and their Yahoo Finance tickers
coins = {
    'USDT': 'USDT-USD',
    'USDC': 'USDC-USD',
    'DAI':  'DAI-USD',
    'UST':  'UST-USD',
}

# Date range
start_date = '2021-01-01'
end_date = '2023-12-31'

print(f"Fetching stablecoin data from {start_date} to {end_date}...\n")

for coin_name, ticker in coins.items():
    print(f"Downloading {coin_name} ({ticker})...", end=' ')
    
    try:
        # Download from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"✗ No data available")
            continue
        
        # Handle multi-level column index from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten the MultiIndex columns by dropping the ticker level
            data.columns = data.columns.droplevel(1)  # Remove ticker level, keep price type
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Add Coin column
        data['Coin'] = coin_name
        
        # Keep only standard OHLCV columns
        data = data[['Date', 'Coin', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Drop rows with NaN Close
        data = data.dropna(subset=['Close'])
        
        # Save to CSV
        filepath = os.path.join(data_dir, f'{coin_name.lower()}.csv')
        data.to_csv(filepath, index=False)
        
        print(f"✓ {len(data)} days saved")
        print(f"  Range: {data['Date'].min().date()} to {data['Date'].max().date()}")
        print(f"  Prices: ${data['Close'].min():.6f} - ${data['Close'].max():.6f}\n")
        
    except Exception as e:
        print(f"✗ Error: {e}\n")

print("Done! CSV files saved to data/stablecoins/")
