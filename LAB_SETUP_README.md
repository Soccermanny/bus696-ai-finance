# Week 10 Lab: Setup Instructions

## Problem Solved

Newer versions of pandas changed how `groupby().apply()` handles group dataframes. The lab code expects the groupby key (e.g., 'Coin') to be present in each group, but newer pandas strips it out.

## Solution

**Add this single line to the FIRST code cell of the notebook** (before any other code runs):

```python
import setup_lab_environment
```

Complete example of what your first code cell should look like:

```python
import setup_lab_environment  # ← ADD THIS LINE

# ============================================================
# SETUP: Libraries, color palette, plot defaults
# ============================================================
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
# ... rest of your setup code
```

This line must appear:
- **Before** the setup cell runs (cell 3 of the lab)
- **Before** `engineer_features` and `make_target` are called

---

## Why This Works

The `setup_lab_environment.py` patch makes pandas preserve the groupby key in group dataframes, which is what the lab code expects. You only need to import it once per notebook session.

## Alternative (if you prefer not to modify the notebook)

Run this in a terminal before opening the notebook:

```bash
python setup_lab_environment.py
```

This imports the module, applies the patch globally to your Python session, and confirms success.

---

## Data Files

The CSV files for USDT, USDC, DAI, and UST are already in `data/stablecoins/`:
- `data/stablecoins/usdt.csv`
- `data/stablecoins/usdc.csv`
- `data/stablecoins/dai.csv`
- `data/stablecoins/ust.csv`

You're ready to run the lab!
