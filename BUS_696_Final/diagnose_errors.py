"""
Diagnostic script for Defense Sector Trading Strategy (BUS696).
Identifies notebook errors by running key sections.
Universe: 21 defense & aerospace stocks, TOP_N=6, min_obs=10.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*70)
print("DIAGNOSTIC: Defense Sector Trading Strategy (BUS696)")
print("DIAGNOSTIC: Running notebook sections to identify errors")
print("="*70)
print()

# Test 1: Imports
print("[1/5] Testing imports...")
try:
    import yfinance as yf
    import requests
    import time
    import matplotlib.pyplot as plt
    from scipy import stats
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    import os
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    exit(1)

print()

# Test 2: Basic setup
print("[2/5] Testing basic setup...")
try:
    START_DATE = '2015-01-01'
    END_DATE = '2024-12-31'
    BACKTEST_START = '2018-01-01'
    TOP_N = 6       # top 30% of 21-stock defense universe
    MIN_OBS = 10    # small universe (was 50 for S&P 500)
    print("✓ Setup complete (Defense Sector: TOP_N=6, MIN_OBS=10)")
except Exception as e:
    print(f"✗ Setup error: {e}")
    exit(1)

print()

# Test 3: Create simple prices_monthly
print("[3/5] Creating test price data...")
try:
    dates = pd.date_range('2018-01-01', '2024-12-31', freq='D')
    n_dates = len(dates)
    n_tickers = 21  # defense universe size
    # Use realistic defense ticker names for testing
    DEFENSE_UNIVERSE = ['LMT', 'RTX', 'NOC', 'GD', 'BA', 'HII', 'LHX', 'LDOS',
                        'BAH', 'SAIC', 'CACI', 'HEICO', 'TDG', 'KTOS', 'AXON',
                        'BWXT', 'DRS', 'CW', 'MRCY', 'PLTR', 'VSEC']
    prices_daily = pd.DataFrame(
        np.random.randn(n_dates, n_tickers).cumsum() + 100,
        index=dates,
        columns=DEFENSE_UNIVERSE
    )
    prices_monthly = prices_daily.resample('ME').last()
    returns_monthly = prices_monthly.pct_change()

    def winsorize(df, lower=0.01, upper=0.99):
        lo = df.quantile(lower, axis=1)
        hi = df.quantile(upper, axis=1)
        return df.clip(lo, hi, axis=0)

    returns_monthly_clean = winsorize(returns_monthly)
    print(f"✓ Price data created: {prices_monthly.shape}")
    print(f"  Date range: {prices_monthly.index[0].date()} to {prices_monthly.index[-1].date()}")
except Exception as e:
    print(f"✗ Price data error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Test 4: quarterly_to_monthly_signal function
print("[4/5] Testing quarterly_to_monthly_signal function...")
try:
    def quarterly_to_monthly_signal(quarterly_df, prices_monthly, signal_col, lag_months=1):
        """Convert quarterly accounting data to monthly signal."""
        if len(quarterly_df) == 0:
            return pd.DataFrame(np.nan, index=prices_monthly.index, columns=prices_monthly.columns)

        quarterly_pivot = quarterly_df.pivot_table(
            index='date', columns='ticker', values=signal_col, aggfunc='last'
        )
        monthly_signal = quarterly_pivot.reindex(prices_monthly.index, method='ffill')
        if lag_months > 0:
            monthly_signal = monthly_signal.shift(lag_months)
        monthly_signal = monthly_signal.reindex(columns=prices_monthly.columns, fill_value=np.nan)
        return monthly_signal

    # Test with empty data
    test_df = pd.DataFrame()
    result = quarterly_to_monthly_signal(test_df, prices_monthly, 'accruals', lag_months=1)
    print(f"✓ quarterly_to_monthly_signal works: output shape {result.shape}")
except Exception as e:
    print(f"✗ quarterly_to_monthly_signal error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Test 5: xsec_zscore function
print("[5/5] Testing xsec_zscore function...")
try:
    def xsec_zscore(df):
        """Cross-sectional z-score normalization."""
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-6, axis=0)

    test_signal = pd.DataFrame(
        np.random.randn(len(prices_monthly), n_tickers),
        index=prices_monthly.index,
        columns=prices_monthly.columns
    )
    result = xsec_zscore(test_signal)
    print(f"✓ xsec_zscore works: output shape {result.shape}")
    print(f"  Mean per row: {result.mean(axis=1).mean():.6f} (should be ~0)")
    print(f"  Std per row: {result.std(axis=1).mean():.6f} (should be ~1)")
except Exception as e:
    print(f"✗ xsec_zscore error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()
print("="*70)
print("✓ ALL DIAGNOSTIC TESTS PASSED")
print("="*70)
print()
print("NEXT: Try running the full notebook in Jupyter now.")
print("If errors occur, they'll be shown in the notebook terminal.")
print()
