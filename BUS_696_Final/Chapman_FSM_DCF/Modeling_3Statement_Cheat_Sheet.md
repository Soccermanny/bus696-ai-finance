# 3-Statement Modeling Cheat Sheet (IS + BS + CFS)

## 1) Build Order (Recommended)
1. Input at least 2-3 years of historicals (left to right).
2. Forecast Income Statement operating lines (revenue, COGS, opex).
3. Build Balance Sheet supporting schedules (working capital, PP&E, intangibles, debt, equity).
4. Build Cash Flow Statement with all non-cash add-backs and working capital deltas.
5. Set cash and revolver as plugs after all other schedules are linked.
6. Check the balance sheet balances every forecast year.

## 2) Income Statement Forecast Drivers
- Revenue:
  - Simple case: growth-rate assumption.
  - Advanced case: segment-level price x volume buildup.
- COGS: usually forecast via gross margin or COGS margin.
- Opex (SG&A, R&D): forecast as a percent of revenue unless there is a specific thesis.
- D&A: forecast from PP&E and intangible schedules, then link back to IS.
- SBC: typically as percent of revenue (or opex), then link through equity and CFS add-back.
- Interest expense: debt balance x interest rate.
- Interest income: cash balance x yield.
- Taxes: usually effective tax rate straight-line unless there is a reasoned change.

## 3) Balance Sheet Forecast Drivers
### Working Capital
- AR: grow with revenue or override via DSO.
- Inventory: grow with COGS or override via turnover.
- AP: grow with COGS (or revenue if mixed) or override via days payable.
- Prepaids, accrueds, other current assets/liabilities: usually tie to revenue or SG&A.
- Deferred revenue: usually grows with revenue.
- Taxes payable: tie to tax expense growth.

### Long-Term Assets
- PP&E roll-forward:
  - PP&E EOP = PP&E BOP + capex - depreciation - asset sales
- Intangibles roll-forward:
  - Intangibles EOP = Intangibles BOP + purchases - amortization
- Goodwill: usually straight-line absent explicit M&A or impairment thesis.
- Other long-term assets/liabilities: straight-line if disclosure is weak.

### Debt and Equity
- Long-term debt:
  - Do not assume only contractual paydown.
  - Often hold roughly stable or refinance to target capital structure.
- Common stock and APIC:
  - New issuance only with explicit case.
  - SBC increases APIC.
- Treasury stock:
  - Buybacks reduce cash and reduce equity (contra-equity effect).
- Retained earnings roll-forward:
  - RE EOP = RE BOP + Net Income - Dividends
- OCI:
  - Usually straight-line unless there is a specific forecast view.

## 4) Cash Flow Statement Linking Rules
- Net income from IS is the CFS starting point.
- Add back non-cash items (D&A, SBC, non-cash charges).
- Include working capital changes from BS deltas.
- Include investing flows (capex, asset/intangible purchases, asset sales).
- Include financing flows (debt issued/repaid, buybacks, dividends, equity issuance).
- Cash and short-term borrowing (revolver/commercial paper) are final balancing plugs.

## 5) Circularity: Practical Handling
- Average-balance interest is conceptually cleaner but can create circular references.
- Beginning-balance interest is often used to reduce circularity in training and interview models.
- If using circularity with iteration on, document the convention clearly.

## 6) Fast Quality Checks
- Sign convention is consistent (especially capex, debt flows, buybacks, dividends).
- No hardcodes in formula cells intended to calculate.
- Every BS movement has a matching CFS cash impact.
- Debt schedule links to interest expense.
- Cash schedule links to interest income.
- Shares schedule links to EPS.
- Assets = Liabilities + Equity in every projected year.

## 7) Common Reasons Models Do Not Balance
- Wrong sign on capex, debt repayment, or buybacks.
- Mislink between schedules (for example, dividends linked where SBC should be).
- Missing CFS line for a BS movement (often in other assets/liabilities).
- Mixing period-average and period-end conventions inconsistently.

## 8) 30-Second Debug Process
1. Start at AR and move line by line down the BS.
2. For each line, compute the year-over-year change.
3. Confirm that exact change appears correctly in CFS adjustments.
4. Cross-check each matched pair and continue until mismatch is found.
5. Fix the first mismatch, then recheck balance.

## 9) Interview-Ready Soundbites
- Revenue is usually the key value driver; everything else should map to operating reality.
- Supporting schedules do the work; statements should mostly present linked outputs.
- A clean, fully linked model with explicit assumptions beats a complex but fragile model.
