# Historical Validation Study Guide
## Out-of-Sample Model Testing Across 10-Year Windows

---

## Overview

This validation study tests the XGBoost cross-sectional stock prediction model against historical market data spanning multiple decades. The key feature: **crisis periods are excluded** to isolate model performance in normal market conditions.

**Test Design:**
- 4 independent 10-year rolling windows
- Non-overlapping training and test periods
- Crisis periods removed: 2001 (9/11), 2008 (Financial Crisis), 2020 (COVID-19)
- Model trained with no forward-looking information

---

## Test Windows

### Window 1: 1995-2005
- **Training:** 1990-1994
- **Testing:** 1995-2000 (before 9/11)
- **Purpose:** Test pre-crisis performance in 1990s bull market

### Window 2: 2003-2007
- **Training:** 2000-2002
- **Testing:** 2003-2007 (before financial crisis)
- **Purpose:** Test recovery period and expansion (2003-2007)

### Window 3: 2010-2020
- **Training:** 2009-2009 (post-crisis)
- **Testing:** 2010-2020 (before COVID)
- **Purpose:** Test post-crisis recovery and expansion (2010-2020)

### Window 4: 2022-2026
- **Training:** 2021-2021 (post-COVID)
- **Testing:** 2022-2026 (recent period)
- **Purpose:** Test current market conditions

---

## Performance Metrics

### 1. **Mean Absolute Error (MAE)**
- Measures average magnitude of prediction errors
- **Lower is better**
- Interpretation: If MAE = 0.02, average prediction error is 2% per month

### 2. **Root Mean Squared Error (RMSE)**
- Emphasizes larger errors more than MAE
- **Lower is better**
- Interpretation: Penalizes outlier predictions

### 3. **R² (Coefficient of Determination)**
- Percentage of return variance explained by model
- **Range: 0 to 1, higher is better**
- Interpretation: R² = 0.05 means model explains 5% of return variance

### 4. **Directional Accuracy**
- Percentage of predictions with correct sign (up/down)
- **Baseline: 50% (random chance)**
- Interpretation: > 50% = better than random; 55% = good; 60%+ = excellent

### 5. **Information Coefficient (IC)**
- Rank correlation between predictions and actual returns
- **Range: -1 to +1**
- **Interpretation:**
  - IC > 0.05: Meaningful signal
  - IC > 0.10: Strong signal
  - IC < 0.02: Negligible
  - Typical: IC = 0.02-0.04 for good models

### 6. **Correlation**
- Linear correlation between predicted and actual returns
- **Range: -1 to +1**
- **Interpretation:** How closely predictions track actuals

---

## How to Interpret Results

### Case 1: Strong Model (Desired Outcome)
```
MAE:                    0.03
RMSE:                   0.05
R²:                     0.06
Directional Accuracy:   55%+
IC:                     +0.04+
Correlation:            +0.15+
```
**Verdict:** Model has genuine predictive power across 10-year periods

### Case 2: Weak Model (Over-Optimized)
```
MAE:                    0.15+
RMSE:                   0.20+
R²:                     0.00-0.02
Directional Accuracy:   48-50%
IC:                     -0.02 to +0.02
Correlation:            0.00-0.05
```
**Verdict:** Model is no better than random; likely over-optimized to specific periods

### Case 3: Regime-Dependent Model
```
Upmarket IC:            +0.08
Downmarket IC:          -0.02
Upmarket Accuracy:      60%
Downmarket Accuracy:    45%
```
**Verdict:** Model works well in rising markets but fails in downturns; needs risk management

---

## Running the Validation Study

### Quick Start
```python
# Execute all cells from top to bottom
# Notebook will:
# 1. Load 35 years of historical price data
# 2. Train model on each window independently
# 3. Test on 10-year holdout periods
# 4. Generate performance metrics
# 5. Create visualization dashboards
```

### Output Files
- `validation_prices_cache.parquet` — Cached price data (speeds up re-runs)
- `historical_validation_dashboard.png` — 6-panel performance summary
- `predicted_vs_actual_scatter.png` — Scatter plots for each window

### Typical Runtime
- First run: 10-15 minutes (includes data download)
- Subsequent runs: 2-3 minutes (uses cache)

---

## Interpreting Visualizations

### Dashboard (Panel 1-6)
1. **MAE by Window** — Lower bars = better predictions
2. **RMSE by Window** — Squared error; penalizes outliers
3. **R² by Window** — Model fit; >0 preferred, higher better
4. **Directional Accuracy** — Percentage beating 50% random
5. **Information Coefficient** — Green (+) is good, red (-) is bad
6. **Correlation** — How closely predictions track actuals

### Scatter Plots (Per-Window)
- **X-axis:** Actual returns
- **Y-axis:** Predicted returns
- **Red dashed line:** Perfect prediction
- **Blue points:** Individual stock predictions
- **Statistics shown:** Correlation and R²

---

## Key Findings to Look For

### ✓ Model is Robust If:
- IC is positive and consistent across all windows
- Directional accuracy > 53% (statistically significant vs 50%)
- R² > 0.02 (explains some variance)
- Performance is similar across different decades
- Model works in both up and down markets

### ✗ Model May Be Over-Optimized If:
- IC varies wildly between windows (0.10 in one, -0.05 in another)
- Performance degrades significantly in recent periods
- Directional accuracy close to 50%
- R² < 0.01
- Model only works in specific market conditions (e.g., bull markets)

---

## Crisis Period Exclusion: Why It Matters

**Without exclusion:**
- Model would need to predict during extreme market dislocations
- Would over-emphasize tail risk management vs signal quality
- Results would be dominated by 2008 financial crisis performance

**With exclusion:**
- Isolates "normal" market conditions
- Tests whether signals actually work for regular trading
- Prevents crisis periods from masking or exaggerating skill

**Note:** Real trading must handle crises! This study validates signals in normal conditions.

---

## Next Steps After Validation

### If Model Shows Strong Signal (IC > 0.05):
1. ✓ Deploy model with confidence
2. ✓ Monitor for regime changes
3. ✓ Add real transaction costs
4. ✓ Test on very recent data (2024-2026)

### If Model Shows Weak Signal (IC < 0.03):
1. Add more features (value, quality, liquidity, sentiment)
2. Ensemble with other models
3. Try non-linear models (more complex)
4. Test on sector-specific universes

### If Model Shows Regime Dependence:
1. Add explicit regime identification
2. Use separate models for bull/bear markets
3. Scale position sizing by regime
4. Add defensive positioning in crisis periods

---

## Technical Details

### Why 10-Year Windows?
- Long enough to capture meaningful predictions
- Short enough to avoid too much data drift
- Standard in institutional asset management
- Allows 4 independent tests from 1990-2026

### Why Non-Overlapping Periods?
- True out-of-sample test (no data leakage)
- Different market conditions each period
- No artificial confidence from overlapping data

### Why Exclude Crises?
- Anomalies can't be predicted (by definition)
- Real alphas show in normal markets
- Prevents model from being tuned to rare events
- Aligns with institutional practice

---

## Comparison to Your Main Model

This validation study is **intentionally simplified** compared to your main BUS696 project:
- Uses only 2 signals (momentum, low-vol) vs 5
- No insider data, earnings quality, or macro regime scaler
- Single XGBoost model vs walk-forward ensemble

**Why simplified?**
- Easier to replicate and understand
- Focuses on core signal robustness
- Provides baseline to compare against main model

**Expected result:** 
- Main model IC should be **higher** (more signals)
- Historical validation IC should be **lower** (fewer signals)
- If equal/close → main model signals may not add value

---

## References & Further Reading

**Key Academic Papers:**
- Jegadeesh & Titman (1993) — Momentum anomaly
- Frazzini & Pedersen (2014) — Low-vol anomaly
- Lopez de Prado (2018) — Backtesting pitfalls

**Related Validation Approaches:**
- Walk-forward backtesting (uses train/test splits)
- Walk-back testing (retrain as new data arrives)
- Monte Carlo permutation tests (shuffle data labels)
- Cross-validation k-folds (multiple train/test combinations)

---

## Questions to Ask About Your Results

1. **Is the IC consistent?** (Good: varies by <0.05 across windows)
2. **Is accuracy > 50%?** (Statistically testable with t-stat)
3. **Does it work in both markets?** (Check upmarket vs downmarket IC)
4. **Does it hold in recent data?** (2022-2026 window most important)
5. **Can I afford the costs?** (IC must be high enough to cover trading costs)

---

Created: May 6, 2026
Model: XGBoost Cross-Sectional Prediction
Test Period: 1990-2026 (excluding crises)
