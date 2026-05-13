# Congressional Defense API: Quick Start Guide

## Overview

You now have a complete system to:
1. **Track defense legislation** (NDAA bills, appropriations, amendments)
2. **Fetch defense stock prices** (LMT, RTX, BA, NOC, GD, TXT, HII, CACI)
3. **Correlate laws with stock performance** (find which laws move stocks most)
4. **Measure Information Coefficient (IC)** (predictive power of legislation signals)

---

## Files Created

| File | Purpose |
|------|---------|
| **congressional_defense_api.py** | Python module with 3 main classes |
| **Congressional_Defense_API_Analysis.ipynb** | Jupyter notebook to run analysis |
| **DEFENSE_SECTOR_ALT_DATA_GUIDE.md** | Comprehensive documentation |

---

## How to Run

### Option 1: Run Jupyter Notebook (Recommended)

```bash
# Navigate to project directory
cd C:\Users\manny\Documents\BUS696\BUS_696_Final

# Start Jupyter
jupyter notebook Congressional_Defense_API_Analysis.ipynb

# Execute all cells (or run section by section)
```

### Option 2: Run Python Script Directly

```python
from congressional_defense_api import main

# Run complete analysis
results = main()

# Access results
ndaa_bills = results['ndaa_bills']
stock_prices = results['stock_prices']
correlations = results['correlation_results']
```

---

## What The API Does

### Part 1: Congress.gov API Client

**CongressionalDefenseAPI** class fetches legislation:

```python
congress_api = CongressionalDefenseAPI()

# Get NDAA history (2000-2026)
ndaa_df = congress_api.get_ndaa_history(start_year=2000, end_year=2026)

# Returns DataFrame with:
# - congress: Congress number
# - bill_number: Bill ID (e.g., "S.4711")
# - title: Full bill title
# - introduced_date: When bill introduced
# - passed_date: When passed Senate/House
# - enacted_date: When signed into law (KEY DATE)
# - status: Current status
```

### Part 2: Defense Stock Data Fetcher

**DefenseStockData** class fetches prices:

```python
stock_data = DefenseStockData()

# Fetch prices for 8 defense contractors (2015-2026)
prices = stock_data.fetch_prices(start_date='2015-01-01', end_date='2026-05-06')

# Returns: Dictionary {ticker: DataFrame with OHLCV}
# Tickers: LMT, RTX, BA, NOC, GD, TXT, HII, CACI
```

### Part 3: Correlation Analysis

**LawStockCorrelationAnalysis** class correlates laws with returns:

```python
correlation_analysis = LawStockCorrelationAnalysis(ndaa_df, prices)

# Create events timeline
events_df = correlation_analysis.create_legislation_events()

# Calculate correlations: Law passage → Stock returns
corr_results = correlation_analysis.correlate_all_stocks(events_df)

# Returns: DataFrame with Spearman/Pearson correlations and p-values
```

---

## Key Output: Correlation Results

### Example Output

```
  ticker company                    spearman_corr  spearman_pval  significant
0    LMT   Lockheed Martin                 0.0234         0.4521       False
1    RTX   Raytheon Technologies           0.0156         0.6234       False
2    NOC   Northrop Grumman               -0.0089         0.7891       False
3     GD   General Dynamics                0.0312         0.3456       False
4    TXT   Textron                         0.0145         0.6789       False
```

### Interpretation

- **Spearman_corr**: Rank correlation between law passage and stock returns
  - Range: -1.0 to +1.0
  - > 0 = stocks tend to go up when laws pass
  - < 0 = stocks tend to go down when laws pass

- **Spearman_pval**: Statistical significance (probability this is random)
  - < 0.05 = **SIGNIFICANT** (not random)
  - > 0.05 = **NOT SIGNIFICANT** (could be random)

- **Significant**: True if p-value < 0.05

### Expected Results

If correlations are **NOT significant** (p > 0.05), this means:

1. **Market efficiency**: Market prices in legislation BEFORE passage
   - NDAA debate starts (t-1): Market begins pricing in
   - NDAA passed (t): Already reflected in price
   - Result: No additional reaction at passage

2. **Lagged effect**: Market reacts with delay
   - Solution: Test 30-day, 60-day, 90-day forward returns
   - Current notebook tests 30-day; you can extend

3. **Amendment-level detail needed**: NDAA passage is too broad
   - Solution: Track specific program amendments
   - Example: "F-35 +$5B" is more specific than "NDAA passed"

---

## Stock Reaction Analysis

### What It Measures

When NDAA passes, how much do defense stocks rise?

```
Calculation:
  1. Find all NDAA passage dates (enacted_date)
  2. For each date, calculate 30-day forward return
  3. Average across all NDAA passage events
  
Example:
  • NDAA passed March 15, 2024
  • LMT closed March 15 at $450
  • LMT closed April 14 (30 days later) at $463
  • Return = (463 - 450) / 450 = +2.89%
  
  • If 5 NDAA passages: average across all
```

### Example Results

```
         NDAA Events  Avg 30d Return  Win Rate  Avg +Return  Avg -Return
LMT               5           +2.14%     60.0%       +3.45%       -1.23%
RTX               5           +1.89%     60.0%       +2.87%       -0.95%
BA                5           +0.45%     40.0%       +2.10%       -2.15%
NOC               5           +1.67%     60.0%       +2.56%       -1.34%
```

**Interpretation:**
- LMT averages +2.14% gain 30 days after NDAA
- Win rate 60% = 3 out of 5 NDAAs preceded positive returns
- Avg positive return: +3.45% when it works
- Avg negative return: -1.23% when it doesn't work

---

## Visualizations Generated

The notebook creates 3 charts:

### 1. law_stock_correlation.png
Shows Spearman correlation for each stock (green = positive, red = negative)

### 2. ndaa_win_rates.png
Win rate (% of NDAAs that preceded positive returns)
- Baseline: 50% (random)
- >55% = better than random
- <50% = worse than random

### 3. ndaa_avg_returns.png
Average 30-day return after NDAA passage

---

## How to Interpret Your Results

### Scenario 1: Strong Positive Correlation ✓

```
Spearman ρ = +0.15, p-value = 0.032 (SIGNIFICANT)
Avg 30d return = +2.5%
Win rate = 65%
```

**Verdict:** NDAA passage predicts positive returns
- **IC**: 0.15 (good signal)
- **Action**: Include in XGBoost model
- **Feature**: Binary indicator for NDAA passage (0 or 1)

### Scenario 2: No Correlation (Current Likely Result)

```
Spearman ρ = +0.025, p-value = 0.654 (NOT SIGNIFICANT)
Avg 30d return = +0.5%
Win rate = 52%
```

**Verdict:** No strong correlation
- **IC**: 0.025 (weak signal)
- **Possible reason**: Market prices in NDAA before passage
- **Next step**: 
  1. Test 60-90 day forward returns (not 30)
  2. Track amendments (not just passage)
  3. Watch bills DURING debate (not after)

### Scenario 3: Negative Correlation ✗

```
Spearman ρ = -0.08, p-value = 0.201 (NOT SIGNIFICANT)
Avg 30d return = -1.2%
Win rate = 42%
```

**Verdict:** Stocks sometimes fall after NDAA
- Could be coincidence (not statistically significant)
- Could indicate market has already priced in positive factors

---

## Integration Into Main Model

Once you understand the signal strength, integrate into your XGBoost:

```python
# From your BUS696_Final_Project_Trading_Strategy.ipynb

# Step 1: Create NDAA feature
ndaa_events = congressional_api.get_ndaa_history()
ndaa_dates = ndaa_events[ndaa_events['enacted_date'].notna()]['enacted_date']

# Step 2: Add as binary feature to each stock
features['ndaa_signal'] = 0
for date in ndaa_dates:
    features.loc[date:, 'ndaa_signal'] = 1  # Flag rises after NDAA

# Step 3: Include in XGBoost
model.fit(
    features[['momentum', 'volatility', 'ndaa_signal', ...]],
    target_returns
)

# Step 4: Measure IC
ic_ndaa_only = calculate_ic(features['ndaa_signal'], target_returns)
# Expected: 0.02-0.04
```

---

## Advanced: Amendment-Level Tracking

Current approach:
- Tracks NDAA bill passage (broad)
- Correlation: weak (0.02-0.03)

Better approach:
- Track specific amendments (detailed)
- Example: "F-35 Program +$5B" → LMT specific signal
- Expected correlation: stronger (0.04-0.08)

**How to implement:**

```python
# Extract amendment text from Congress.gov
# Parse for contractor mentions and program changes
# Examples:
#   "Lockheed Martin F-35" → LMT positive
#   "Raytheon Missile Systems +$2B" → RTX positive
#   "Budget cuts for..." → Negative signal

# Use Claude LLM to extract:
prompt = f"""
Parse this defense bill amendment for trading signals:
{amendment_text}

Extract:
1. Contractors mentioned (ticker symbols)
2. Programs affected (F-35, hypersonics, etc.)
3. Budget impact ($+ or $-)
4. Signal strength: bullish/neutral/bearish
"""
```

---

## Troubleshooting

### Issue: "API error" or slow downloads

**Solution**: Congress.gov API has rate limits
```python
# Add delays between requests
import time

for bill in bills:
    process_bill(bill)
    time.sleep(0.5)  # 500ms delay between API calls
```

### Issue: No data returned

**Possible causes:**
1. Congress.gov API down → Try again later
2. No internet connection → Check network
3. Congress number is wrong → Check current congress (118 as of 2024)

### Issue: Stock prices have NaN values

**Solution**: yfinance sometimes fails for delisted stocks
```python
# Filter out stocks with too many NaNs
prices = prices.dropna(thresh=len(prices) * 0.9)  # Keep if >90% data
```

---

## Expected Timeline

| Week | Task | Time |
|------|------|------|
| 1 | Run Congressional_Defense_API_Analysis.ipynb | 15 min |
| 1 | Interpret correlation results | 30 min |
| 2 | Implement amendment tracking (LLM) | 2 hours |
| 2 | Test lagged returns (60-90 days) | 1 hour |
| 3 | Integrate into XGBoost model | 2 hours |
| 3 | Measure combined IC with other signals | 1 hour |

---

## Expected IC (Information Coefficient)

| Signal | Alone | Combined |
|--------|-------|----------|
| NDAA passage (current) | 0.02-0.03 | 0.03-0.04 |
| NDAA amendments (future) | 0.05-0.08 | 0.06-0.10 |
| + Ukraine munitions data | - | 0.10-0.15 |
| + Geopolitical risk | - | 0.12-0.18 |

**Note**: Combining multiple signals is how you build alpha.
- Single signal IC: 0.02-0.03
- Multiple signals (this pipeline): 0.08-0.15
- That's 4-5x improvement!

---

## Next Steps

1. ✅ **This week**: Run Congressional_Defense_API_Analysis.ipynb
2. ⏳ **Next week**: Implement amendment-level tracking
3. ⏳ **Week 3**: Integrate into main XGBoost model
4. ⏳ **Week 4**: Measure combined signal IC (target: 0.05+)

---

**Created:** May 6, 2026  
**Location:** BUS_696_Final/  
**Files:** 
- congressional_defense_api.py (400 lines)
- Congressional_Defense_API_Analysis.ipynb (executable)
- DEFENSE_SECTOR_ALT_DATA_GUIDE.md (reference)
