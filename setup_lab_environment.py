"""
Setup script to make the lab notebook compatible with modern pandas.

This patches pandas groupby().apply() to preserve groupby keys in the group dataframes,
which is expected by the lab code. Run this BEFORE opening the lab notebook.

Usage:
    python setup_lab_environment.py
    
Then open Week_10_Lab_Stablecoin_Peg_Risk.ipynb and add this to the first cell:
    import setup_lab_environment
"""

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

# Store the original apply method
_original_groupby_apply = DataFrameGroupBy.apply

def _patched_apply(self, func, *args, **kwargs):
    """
    Patched groupby.apply() that preserves groupby keys in the group dataframe.
    
    This makes newer pandas versions compatible with code that expects the groupby
    key(s) to be present in the group dataframe passed to the function.
    """
    grouper = self.grouper
    group_keys = [g for g in grouper.names if g is not None]
    
    if not group_keys:
        # No named groupby keys, use original behavior
        return _original_groupby_apply(self, func, *args, **kwargs)
    
    # Manually iterate through groups and re-add the groupby key(s)
    results = []
    for key, group in self:
        # Re-add the groupby key(s) to the group
        if isinstance(key, tuple):
            for k, v in zip(group_keys, key):
                group[k] = v
        else:
            group[group_keys[0]] = key
        
        result = func(group, *args, **kwargs)
        results.append(result)
    
    # Concatenate results
    if results and isinstance(results[0], pd.DataFrame):
        return pd.concat(results, ignore_index=True)
    return _original_groupby_apply(self, func, *args, **kwargs)

# Apply the patch
DataFrameGroupBy.apply = _patched_apply

print("✓ Lab environment patched: groupby().apply() will preserve groupby keys")
