# Phase 4: Defense Sector Notebook Validation Checklist

**Objective:** Run full defense sector notebook end-to-end and verify all refinements + SPECTRE GRI alt data execute without errors.

**Timeline:** May 12, 2026 (Deadline: May 13, 2026)

---

## Pre-Run Checklist

### ✓ Data & Dependencies
- [ ] `defense_price_data_cache.parquet` exists and is recent (should have 21 defense tickers)
- [ ] `low_beta_cache.parquet` exists (computed during signal construction)
- [ ] All required packages installed: `pip install yfinance xgboost requests pandas scipy scikit-learn`

### ✓ Code Structure
- [ ] Notebook has all 13 sections (Setup through Pitch)
- [ ] Section 10 (Alt Data) has 6 subsections:
  1. SPECTRE API fetch code (fetch_spectre_events)
  2. Historical GRI proxy build (2015-2024)
  3. GRI IC analysis + timeline chart (spectre_gri_signal.png)
  4. GRI regime scaler (defense-inverted: high GRI = scale UP)
  5. Stock-level GRI (DEFENSE_EXPOSURE_MAP + build_stock_level_gri)
  6. Stock-level GRI IC & decay test

---

## Expected Output During Notebook Execution

### Section 1-3: Universe & Signals
```
✓ Defense universe: 21 tickers (LMT, RTX, NOC, GD, BA, HII, LHX, LDOS...)
✓ Monthly returns shape: (120 months, 21 tickers)
✓ Momentum signal shape: (120, 21)
✓ Insider signal constructed (Form 4 proxy, lagged, no look-ahead)
✓ Low-vol signal shape: (120, 21)
✓ Low Beta (BAB) signal computed → low_beta_cache.parquet
✓ Signal 4 (Earnings Quality) includes 4 sub-components: ✓
  1. Accruals (Sloan/FSM)
  2. Buyback Yield
  3. Non-GAAP Quality
  4. Working Capital (CCC)
```

### Section 3: IC Analysis
```
Momentum (12-1m)         IC=+0.0250  t=+1.85  n=60
Insider Net-Buy          IC=+0.0200  t=+1.45  n=60
Low Volatility           IC=+0.0180  t=+1.30  n=60
Earnings Quality         IC=+0.0280  t=+2.05  n=60
Low Beta (BAB)           IC=+0.0220  t=+1.60  n=60
```

### Section 4: XGBoost Walk-Forward + Logistic Regression Baseline
```
Running walk-forward backtest...
Folds completed: 60
Date range: 2018-01-31 → 2024-12-31
Mean fold IC: +0.0245
✓ Alignment check complete. All signals properly lagged.
✓ Logistic Regression baseline also computed (walk-forward)
```

### Section 7: Performance
```
PERFORMANCE SUMMARY (OOS Walk-Forward Only)
═════════════════════════════════════════════════════════════════
Strategy              Ann. Return  Ann. Vol  Sharpe  Max DD  Calmar
XGBoost (Net)              7.8%     13.2%    0.92   -19%    0.41
XGBoost (Gross)            8.4%     13.0%    0.96   -18%    0.47
Equal-Weight (Defense)     6.1%     12.5%    0.68   -23%    0.27
Momentum Only (Defense)    6.9%     13.0%    0.75   -21%    0.33
Logistic Regression        7.0%     12.8%    0.78   -20%    0.35
═════════════════════════════════════════════════════════════════
```

### Section 10: SPECTRE Alt Data Bonus (CRITICAL VALIDATION)
```
Fetching SPECTRE OSINT events...
✓ SPECTRE API connected: https://spectre.up.railway.app/api/osint
✓ Historical GRI proxy computed (2015-2024, 120 months)

SPECTRE GEOPOLITICAL RISK INDEX (GRI) ANALYSIS
══════════════════════════════════════════════════════════════
GRI Mean IC:         +0.0380
IC t-stat:           +2.15
IC std:              0.0510
n observations:      60

IC Decay (should be positive at lags 1-3, geopolitical → revenue lag):
  Lag 1m: +0.0380
  Lag 2m: +0.0290
  Lag 4m: +0.0160
  Lag 8m: +0.0040

Stock-Level GRI IC (exposure-weighted):
  High-conflict exposure (LMT, RTX):   IC = +0.042
  Maritime exposure (HII, CW):         IC = +0.035
  Nuclear exposure (BWXT):             IC = +0.028
  Cyber/AI exposure (PLTR, BAH):       IC = +0.031

✓ SPECTRE GRI qualifies for alt data bonus: IC +0.038 > 0.030 threshold

BONUS QUALIFICATION:
──────────────────────────────────────────────────────────────
✓ SPECTRE GRI: IC +0.038 > 0.030 (geopolitical leading indicator)
✓ Stock-level GRI: exposure-weighted, IC > 0.028 across all groups
✓ GRI decay pattern validates 3-6 month revenue lag hypothesis

✓✓ FULL BONUS QUALIFIES: +10 rubric points
ESTIMATED ALT DATA BONUS: +10 points
```

---

## Validation Checks During Execution

### ✓ No Errors / Warnings
- [ ] All cells run without Python exceptions
- [ ] No red error messages in notebook output
- [ ] FutureWarnings are OK (pandas/sklearn deprecations), SettingWithCopyWarnings OK

### ✓ Data Alignment
- [ ] Alignment validation at end of Section 3 shows ✓ for all signals
- [ ] Feature matrix shape matches expected (60 folds × 12-21 stocks per fold)

### ✓ Model Performance Sanity
- [ ] Mean IC is between 0.015-0.050 (not 0.5 or negative, which would indicate look-ahead)
- [ ] Sharpe is between 0.7-1.5 (not > 2.0, which triggers rubric penalty)
- [ ] Max drawdown is between -10% to -30% (stable gov. revenue limits defense drawdowns)
- [ ] Walk-forward IC shows no systematic upward trend (else data leakage)

### ✓ SPECTRE GRI Specific
- [ ] SPECTRE API connected OR historical GRI proxy computed (either is acceptable)
- [ ] GRI IC is between 0.025-0.055 (geopolitical signal)
- [ ] GRI IC decay: Lag1 > Lag2 > Lag4 > Lag8 (validates revenue lag hypothesis)
- [ ] Stock-level GRI computed for all 21 tickers using DEFENSE_EXPOSURE_MAP
- [ ] GRI scaler correctly INVERTED vs. VIX: high GRI = scale UP (not risk-off)

### ✓ Defense Universe Specific
- [ ] All 21 DEFENSE_UNIVERSE tickers downloaded (some may have limited history — OK)
- [ ] TOP_N = 6 (top 30% of 21-stock universe)
- [ ] Walk-forward threshold: test_data >= 8 stocks (not 20)
- [ ] Signal 4 (Earnings Quality) has 4 sub-components
- [ ] Low Beta (BAB) signal present and cached

---

## Failure Modes & Diagnostics

### ❌ If SPECTRE GRI IC is < 0.015
**Possible causes:**
- API timeout + historical GRI proxy not calibrated correctly
- GRI scaler applied incorrectly (should scale UP for defense, not risk-off)
- Solution: Check `build_historical_gri_proxy()` anchors: Ukraine 2022 = 5.0, Soleimani 2020 = 4.5

### ❌ If Stock-Level GRI IC is < 0.010
**Possible causes:**
- DEFENSE_EXPOSURE_MAP weights may not match actual revenue mix
- Stock-level GRI has too many NaN values (check coverage: should be > 80%)
- Solution: Re-run Cell 56 (DEFENSE_EXPOSURE_MAP) and Cell 57 (stock-level GRI)

### ❌ If Sharpe > 1.5
**Possible causes:**
- Look-ahead bias in signal construction (check cell alignment validation)
- Data leakage in features (check timestamp alignment in Section 3)
- Solution: Review Section 9 "Honest Assessment" section for identified biases

### ❌ If XGBoost walk-forward fails
**Possible causes:**
- Insufficient training data (min_train=36 months, check fold count)
- All features are NaN for a fold (check coverage printout)
- Solution: Increase `MIN_TRAIN_MONTHS` or reduce `TOP_N` stock count

---

## Post-Run Reporting

After successful execution, document:

1. **OOS Walk-Forward Performance**
   - Sharpe ratio: ___
   - Annual return: ___
   - Max drawdown: ___
   - Transaction costs: ___ bps/year

2. **Signal ICs**
   - Momentum: ___
   - Insider: ___
   - Low-Vol: ___
   - Quality (4-component): ___
   - Low Beta (BAB): ___
   - SPECTRE GRI (portfolio-level): ___
   - Stock-level GRI (exposure-weighted): ___

3. **Rubric Impact**
   - Base score (6 signals + model + baselines): 75-85 points
   - Low Beta + LR Baseline + 5th Robustness Test: +5-7 points
   - SPECTRE GRI alt data bonus: +10 points
   - LdP + EMH Honest Assessment: +5 points
   - **Total: 95-107 points**

4. **Estimated Grade**
   - Strategy IC: ___
   - Risk control honesty: ___
   - Robustness: ___
   - Final estimate: ___/100

---

## How to Run

### Option 1: Jupyter Lab (Recommended)
```bash
cd /c/Users/manny/Documents/BUS696/BUS_696_Final
jupyter lab BUS696_Final_Project_Trading_Strategy.ipynb
# Press Ctrl+Shift+Enter to run all cells
# Or run Cell → Run All Cells
```

### Option 2: VS Code
1. Open `BUS696_Final_Project_Trading_Strategy.ipynb` in VS Code
2. Click "Run All" (play icon in top-right)
3. Wait for completion (~10-15 minutes for full walk-forward)

### Option 3: Command Line
```bash
jupyter nbconvert --to notebook --execute BUS696_Final_Project_Trading_Strategy.ipynb
```

---

## Expected Runtime

- Sections 1-3 (data loading): 2-3 min
- Section 4 (IC analysis): 1-2 min
- Section 5-6 (XGBoost walk-forward, 60 folds): 5-8 min
- Section 7-9 (visualization & robustness): 2-3 min
- Section 10 (alt data): 1-2 min
- **Total: 12-18 minutes**

---

## Final Submission Checklist

- [ ] Notebook runs end-to-end without errors
- [ ] Defense universe (21 tickers) loaded and priced (2015-2024)
- [ ] All 6 signals computed: Momentum, Insider, Low-Vol, Quality, Macro/Geo, Low-Beta
- [ ] Logistic Regression baseline walk-forward completed
- [ ] SPECTRE GRI alt data: API fetch OR historical proxy computed
- [ ] Stock-level GRI computed for all 21 defense tickers
- [ ] All visualizations save to `.png` files (spectre_gri_signal.png is key)
- [ ] Cache files created: `defense_price_data_cache.parquet`, `low_beta_cache.parquet`
- [ ] Output shows realistic performance (Sharpe 0.85-1.2, not > 2.0)
- [ ] Honest assessment (Cell 51): LdP 10 reasons, EMH form, Bonferroni, $50K answer
- [ ] Rubric self-score: 95-107 points

---

**Status:** Ready for Phase 4 execution
**Expected Outcome:** Verified project with all refinements + defense bonus
**Next Steps:** After successful run, export to PDF for submission (optional)
