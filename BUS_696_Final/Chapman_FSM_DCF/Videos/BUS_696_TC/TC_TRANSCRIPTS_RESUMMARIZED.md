# TC Transcripts — Resummarized
**Course:** Trading Comps Modeling (T1–T84) + Transaction Comps Modeling (T85–T129)
**Format:** 4–8 sentence conceptual summary per transcript

---

## Trading Comps Modeling (T1–T84)

---

### T1. Trading-Comps-Modeling-Introduction-to-Trading-Comps
Trading comps (comparable company analysis) values a company by benchmarking it against publicly traded peers with similar business profiles. The method uses standardized multiples — primarily EV/EBITDA, EV/Revenue, EV/EBIT, and P/E — so that companies of different sizes can be compared on a like-for-like basis. Enterprise value multiples are preferred because they are leverage-neutral, while equity value multiples (P/E) implicitly factor in a company's capital structure. The course case study uses Extreme Networks (EXTR), a $1.3B enterprise networking company, as the target being valued.

---

### T2. Trading-Comps-Modeling-Introduction-to-the-Template
The trading comps template is organized into three main tabs: (1) an Input tab that serves as "the brain," feeding all other tabs; (2) an Operating Metrics tab showing revenue, EBITDA, and margins across LTM and forecast periods; and (3) a Valuation Metrics tab displaying EV multiples and P/E for each peer. Data flows via HLOOKUP formulas from the Input tab so that a change in one place updates all downstream outputs automatically. The Input tab captures the full bridge from basic shares → diluted shares → market cap → net debt → EV, along with both historical and calendarized consensus estimates. Understanding the template architecture prevents data-entry errors and ensures consistency across the peer group.

---

### T3. Trading-Comps-Modeling-Understanding-Enterprise-Value
Enterprise value (EV) represents the total value of a company's operating assets net of operating liabilities — conceptually, it is what an acquirer would pay to own the entire business free of its non-operating items. The standard formula is EV = Market Cap + Net Debt, where net debt = gross debt + preferred stock + minority interest − cash − investments. Because cash is a non-operating asset that can immediately pay down debt, practitioners net it against gross debt rather than treating it separately. Understanding EV from the left side of the balance sheet (operating assets minus operating liabilities) reinforces why it is a leverage-neutral measure useful for cross-company comparison.

---

### T4. Trading-Comps-Modeling-Types-of-Multiples
The four primary EV multiples are EV/Revenue, EV/EBIT, EV/EBITDA, and industry-specific variants. EV/EBITDA is the most widely used because it eliminates both the leverage effect (interest expense) and the accounting noise of depreciation methods, making it ideal for capital-intensive companies with similar asset bases. EV/EBIT isolates operating profitability without the D&A add-back, making it better for service businesses with little capital expenditure. EV/Revenue is the multiple of last resort — it implicitly assumes identical cost structures across the peer group and is mainly used when EBITDA is negative (early-stage or distressed businesses). P/E (equity multiple) factors in leverage and is best for mature, profitable companies with similar capital structures; PEG = P/E ÷ long-term growth rate and is used when comparing companies at different growth stages.

---

### T5. Trading-Comps-Modeling-Peer-Selection-Overview
Comparable companies must share the same industry, customer base, competitive position, seasonality, and similar growth and margin trajectories. The core selection sources are the target's 10-K Item 1 (competition section), Section 7 (MD&A), equity research reports, and screening databases (FactSet, Bloomberg, Cap IQ). Peer selection is as much art as science — a company from a slightly different sub-industry may be included if it competes for the same customers, while a nominal industry peer may be excluded if its business model or scale is too different. For EXTR, the chosen peers are CSCO, ANET, HPE, and JNPR; tier-2 candidates (Sienna, Fabronet, Lumentum, etc.) were excluded as insufficiently similar.

---

### T6. Trading-Comps-Modeling-The-Input-Tab
The Input tab starts with the company's basic shares outstanding (from the cover page of the latest 10Q), then adds RSUs/PSUs and outstanding options (after applying the treasury stock method) to arrive at fully diluted shares outstanding (FDSO). Multiplying FDSO by the current share price gives equity value (market cap); adding net debt gives EV. The same tab captures the LTM income statement (revenue through net income), D&A from the cash flow statement, and consensus forecasts for the upcoming two fiscal years. All downstream operating and valuation metrics tabs pull their data via HLOOKUP, so the Input tab must be built precisely to ensure accurate outputs.

---

### T7. Trading-Comps-Modeling-Understanding-FDSO
Fully diluted shares outstanding (FDSO) is calculated by starting with basic shares and adding all in-the-money dilutive securities. Stock options that are exerciseable and in-the-money are included using the treasury stock method: the company is assumed to receive option proceeds and use them to repurchase shares at the current price, so only the net dilution is added. RSUs and PSUs are immediately and fully dilutive in a change-of-control context (they are included above-the-line). Convertible securities require two tests: (1) are they in-the-money (market price > conversion price)?; (2) is conversion dilutive (diluted EPS < basic EPS)? Only if both conditions are met do they enter the diluted count. Restricted stock that has already vested is already in the basic count; unvested restricted stock is typically excluded (with a caveat for materiality).

---

### T8. Trading-Comps-Modeling-Stock-Options-and-Treasury-Stock-Method
The treasury stock method models the net dilutive impact of in-the-money exerciseable options. Gross dilution = total in-the-money options; assumed proceeds = gross dilution × exercise price. Those proceeds are used to repurchase shares at the current stock price, giving shares repurchased = proceeds ÷ current price. Net dilution = gross dilution − shares repurchased. Options that are out-of-the-money (exercise price > current price) are excluded entirely because a rational holder would not exercise them. The resulting net dilution — not the gross option count — is what gets added to basic shares.

---

### T9. Trading-Comps-Modeling-Understanding-GAAP-vs-Non-GAAP
GAAP financials include non-cash and non-recurring items that obscure core operating performance and make cross-company comparison difficult. Analysts adjust to non-GAAP by adding back stock-based compensation (non-cash, assumption-driven), amortization of acquired intangibles (non-cash, acquisition accounting artifact), and one-time operating items such as restructuring, litigation, and acquisition costs. Critically, when you exclude a pre-tax expense, you must also reverse the associated tax shield — otherwise you overstate the after-tax earnings benefit. The primary source for non-GAAP adjustments is the earnings press release (8K filed concurrent with or just before the 10K/Q), which typically includes a formal GAAP-to-non-GAAP reconciliation table.

---

### T10. Trading-Comps-Modeling-The-Double-Counting-Fix
A common mistake occurs when a company's non-GAAP EBIT already excludes amortization of intangibles (so EBIT effectively equals EBIT-A), but then the analyst also adds back the full D&A line to get EBITDA. This double-counts the amortization. The fix is to subtract the amortization component from the D&A row before adding it back to EBIT, so only true depreciation is added in the EBITDA bridge. The correct formula is: Normalized EBITDA = Normalized EBIT + (Total D&A − Amortization already excluded from EBIT). This issue is ubiquitous in tech and networking companies where amortization of purchased intangibles is routinely excluded from non-GAAP EBIT.

---

### T11. Trading-Comps-Modeling-LTM-Concept
LTM (Last Twelve Months) = the most recent completed fiscal year (12 months from the 10-K) + the most recent YTD stub period (from the latest 10-Q) − the same stub period from the prior year's 10-Q (also found in the same filing). This formula converts a fiscal year end result into a rolling 12-month figure that captures the most recent performance. For companies whose fiscal year end coincides with their most recent 10-Q period end (e.g., ANET, HPE, JNPR), LTM = FYE with no stub calculation required. For companies like EXTR (FYE June 30) that have filed a subsequent quarter, the stub periods are needed. It is critical to pull both stub periods from the same 10-Q filing to ensure they are on a comparable basis.

---

### T12. Trading-Comps-Modeling-Calendarization-Overview
Calendarization adjusts all peers' operating metrics to a single, consistent year-end date (typically December 31) so that forward multiples are truly comparable. Without calendarization, a company with a June 30 fiscal year end would have a "forward" multiple based on results only 3 months away, while a December 31 company's forward period is 9 months out — making the June company appear cheaper on a forward multiple. The adjustment uses YEARFRAC weighting: the calendar-year metric = (fraction of last FY within CY) × last FY metric + (fraction of next FY within CY) × next FY estimate. For EXTR (FYE June 30), CY0 = 50% × FY24 + 50% × FY25 across all metrics. In some deals, the target company's own fiscal year end is used as the standardization date instead of December 31.

---

### T13. Trading-Comps-Modeling-Inputting-CSCO
Cisco (CSCO) has a non-calendar fiscal year end of the last Saturday of July (approximately July 29, 2023 for the most recent full year). The most recent quarterly period ends January 27, 2024. Share price used for the analysis is $48.86 as of the price refresh date of February 23, 2024. All amounts for Cisco are in millions (already in the right scale in their 10K), so no manual scaling is needed — unlike EXTR, which reports in thousands and must be divided by 1,000 before entry.

---

### T14. Trading-Comps-Modeling-Inputting-EXTR
Extreme Networks (EXTR) has a fiscal year end of June 30, 2023, with the most recent quarterly period ended December 31, 2023. The share price is $12.38 as of the February 23, 2024 price refresh date. EXTR reports in thousands, so all financial figures must be divided by 1,000,000 (or scaled) when entering into the template. The company has a non-calendar FYE, so calendarization will be required to align its metrics to a December 31 calendar year.

---

### T15. Trading-Comps-Modeling-Operating-Metrics-Tab
The Operating Metrics tab displays, for each peer, revenue and EBITDA in four periods: Historical FY, LTM, CY0 (current calendar year forecast), and CY1 (next calendar year forecast), plus revenue and EBITDA CAGRs (2023–2025) and EBITDA margins. The CAGR formula is (FV/PV)^(1/n) − 1, where n = 2 for a 2023–2025 two-year growth rate. Companies are sorted from largest to smallest by revenue, with the case company (EXTR) at the bottom. Grouping in Excel can collapse or expand peer rows; if rows are not hidden despite being grouped, the grouping must be corrected manually. The tab serves as the primary reference for identifying outliers (e.g., Arista's 14% CAGR) and assessing relative growth and profitability across the peer set.

---

### T16. Trading-Comps-Modeling-Valuation-Metrics-Tab
The Valuation Metrics tab displays EV/Revenue, EV/EBITDA, and P/E multiples for each peer on a CY0 and CY1 forward basis, along with the group's min, max, mean, and median statistics. The PE ratio has two legitimate definitions: price per share ÷ EPS (using weighted average diluted shares for EPS) vs. market cap ÷ net income (using current shares outstanding for market cap). A subtle difference arises if the share count changed significantly during the year; otherwise both are effectively equivalent. Multiples are best read in the context of the peer's growth profile — a premium multiple is expected for a high-growth outlier like Arista, while CSCO/HPE/JNPR trade at similar, compressed multiples to EXTR.

---

### T17. Trading-Comps-Modeling-YTD-Non-GAAP-Adjustments
YTD (year-to-date) non-GAAP adjustments for the stub periods follow the same logic as the full-year adjustments but use the 8K quarterly press release corresponding to each stub period. For EXTR's six-month stub ending December 31, 2023, the source is the 8K for that period, and you pull both the current YTD period and the prior-year YTD period (both in the same 10-Q filing). The non-GAAP reconciliation tables in each press release show which items to exclude from each stub. The double-counting fix for amortization must also be applied consistently in the stub periods. LTM non-GAAP results are then assembled as: FYE normalized + current stub normalized − prior-year stub normalized.

---

### T18. Trading-Comps-Modeling-EXTR-Background
Extreme Networks was founded in 1996 and is headquartered in Morrisville, North Carolina. The company sells enterprise networking equipment (switches, routers, wireless LAN) and competes directly with Cisco, Arista, Juniper, and HPE Networking. With roughly $1.3B in revenue and an EBITDA margin of approximately 6% on a GAAP basis, EXTR is a mid-size player in a market dominated by Cisco. Understanding the competitive landscape (who EXTR names as competitors in its 10-K and MD&A) is the foundation for peer selection.

---

### T19. Trading-Comps-Modeling-SBC-and-Amortization-Theory
Stock-based compensation (SBC) is excluded from non-GAAP earnings because it is a non-cash expense with value driven by assumptions (option pricing models) rather than cash outflows. Most analysts argue that SBC distorts period-to-period comparability and is better treated as a financing cost rather than an operating expense. Amortization of acquired intangibles is also excluded because it reflects a purchase-accounting artifact of past M&A activity and does not reflect ongoing operational cash consumption. When both are excluded, the result is sometimes called "cash EPS" or "cash EBIT." As always, excluding the expense requires also reversing the associated tax shield, since pretending the expense never happened means pretending the tax deduction never happened either.

---

### T20. Trading-Comps-Modeling-Locating-Filings-ANET
Arista Networks (ANET) has a December 31 fiscal year end. Its most recent 10-K was filed February 13 of the following year, and the corresponding 8K earnings press release was filed one day earlier on February 12. For ANET, because its fiscal year end matches the most recent quarter end, LTM = FYE and no stub periods are needed. The filing dates and period end dates should always be noted with a Shift-F2 source comment in the model cell for auditability.

---

### T21. Trading-Comps-Modeling-Locating-Filings-HPE
Hewlett Packard Enterprise (HPE) has a fiscal year end of October 31. Its most recent 10-K was filed December 22 of the calendar year, with a corresponding 8K earnings press release filed around the same time. Because HPE's fiscal year end does not match a recent quarter end, its LTM = FYE (the last full fiscal year is as recent as Q4 FY). Calendarization of HPE's consensus estimates to December 31 uses approximately 83% of the prior FY and 17% of the first forecast year.

---

### T22. Trading-Comps-Modeling-Locating-Filings-JNPR
Juniper Networks (JNPR) has a December 31 fiscal year end and files its 10-K around February 7 of the following year. Like ANET, JNPR's LTM = FYE with no stub adjustments required. Both JNPR and HPE have equity method investments in their balance sheets (JNPR ~$452M total, some disclosed in prepaid/other LT assets; HPE ~$2.1B in H3C Technologies), which analysts may optionally exclude from net debt as non-core operating assets depending on team convention.

---

### T23. Trading-Comps-Modeling-EXTR-Balance-Sheet
Extreme Networks' balance sheet shows: cash $221M, AR $112M, inventory $152.5M, goodwill + intangibles $408.7M, PP&E $47.1M. Debt consists of current portion of LTD ($9.3M) and long-term debt ($182.9M). Market cap = 128.73M basic shares × $12.38/share = $1.596B. Net debt = $192.2M gross debt − $221M cash = −$28.8M (slight net cash position), so EV ≈ $1.567B. All balance sheet items should be entered as zero (not blank) if not found — a zero confirms you checked and found nothing, while a blank is ambiguous.

---

### T24. Trading-Comps-Modeling-EXTR-Basic-Shares
EXTR's basic share count is found on the cover page of its most recent 10-Q: 128.73M shares as of January 26, the cover date (which is slightly after the period end of December 31 — this is normal). RSUs are found on page 19 of the same 10-Q: 7.745M non-vested RSUs outstanding, which are immediately dilutive in a change-of-control context and therefore included above-the-line. Because EXTR reports in thousands, any per-share counts expressed as single shares (e.g., "7,745,000 RSUs") must be divided by 1,000,000 to convert to millions for the model.

---

### T25. Trading-Comps-Modeling-CSCO-Calendarization
Cisco's fiscal year ends in late July (58% of its fiscal year falls within the July–December calendar second half, and 42% falls within January–June of the calendar year). To calendarize to December 31, the formula is: CY metric = (7/12) × last FY + (5/12) × next FY, using YEARFRAC to calculate the precise fractions. For CY24 CSCO: approximately 58% of FY24 (July 2023–July 2024) + 42% of FY25 (July 2024–July 2025) = CY24 estimate. This ensures that CSCO's "forward" period is anchored to the same December 31 endpoint as peers with a calendar year end.

---

### T26. Trading-Comps-Modeling-EXTR-Consensus
EXTR's consensus estimates from FactSet show: LTM revenue $1.318B; FY1 (FY June 2024) EBITDA $135M, EBIT $119M, NI $75M, EPS $0.57; FY2 EBITDA $174M, EBIT $162M, NI $109M, EPS $0.83; FY3 EBITDA $229M, EBIT $210M, NI $149M, EPS $1.14. These are pulled from FactSet's consensus estimates tab; "EBITDA non-GAAP" in FactSet is actually EBIT-A (EBIT excluding amortization) so the label may be misleading. Historical non-GAAP NI of $146M can be used as a cross-check when confirming that the LTM adjustments are being calculated correctly.

---

### T27. Trading-Comps-Modeling-CSCO-FYE-Non-GAAP
CSCO's full-year non-GAAP adjustments (from the FY24 8K earnings release): SBC +$2.347B, amortization of intangibles +$0.912B, acquisition/integration costs +$0.259B, remediation reversal −$0.009B, other +$0.531B = total operating adjustment ~$4B. Adjusted EBIT = GAAP EBIT + $4B = $19.071B. The double-counting fix: subtract $912M of amortization from the D&A row so it is not added back again in the EBITDA calculation. This is the most common non-GAAP modeling error in technology sector comps.

---

### T28. Trading-Comps-Modeling-CSCO-Non-Operating-and-Tax
Non-operating non-GAAP adjustments for CSCO: gain/loss on investments +$133M; income tax effect −$107M (reversing the tax shield on excluded items). After all adjustments, CSCO's normalized net income is $15.979B — confirmed against the company-disclosed non-GAAP net income figure in the same press release. The tax effect line must always be a negative number (a reversal of the tax benefit that would have been associated with the excluded expenses), because pretending the expense never occurred means pretending the related tax deduction also never occurred.

---

### T29. Trading-Comps-Modeling-EXTR-FYE-Non-GAAP
EXTR's full-year (FYE June 30, 2023) non-GAAP operating adjustments total $90.6M (cell row 166 in the template, summing SBC, acquisition/integration, restructuring, litigation, system transition, amortization). The amortization component is $14.916M and must be subtracted from the D&A row (double-counting fix). After all adjustments, normalized EBIT = $198.9M — consistent with broker consensus figures for EXTR non-GAAP EBIT. The 8K filed August 2, 2023 contains all of these non-GAAP reconciliation details and is the primary source document.

---

### T30. Trading-Comps-Modeling-EXTR-FYE-Income-Statement
EXTR's FYE June 30, 2023 income statement (from the 10-K page 50 or corresponding 8K): revenue $1.318B, COGS −$557M, gross profit $754.7M, operating expenses −$646M, GAAP EBIT $108.3M, non-operating items (+$3.155M interest income −$17.385M interest expense +$0.023M other), pre-tax income $94.1M, tax provision −$16.032M, GAAP net income $78M. Diluted weighted average shares outstanding: 133.649M (slightly higher than basic because of dilutive securities). D&A must be pulled from the cash flow statement (not the income statement) since it is embedded in operating expense line items.

---

### T31. Trading-Comps-Modeling-EXTR-FYE-Income-Statement-Part-2
Continuing from T30: D&A for EXTR FYE is found on page 53 of the 10-K in the operating activities section of the cash flow statement. D&A is broken into depreciation and amortization components; the amortization component must be netted out when computing the EBITDA add-back to avoid the double-count fix. GAAP net income $78M confirms input accuracy when the EPS formula ($78M ÷ 133.649M) produces $0.584, consistent with reported GAAP diluted EPS of approximately $0.58. Always sanity check the bottom-line EPS against the reported figure before moving to non-GAAP adjustments.

---

### T32. Trading-Comps-Modeling-EXTR-Options
EXTR has 1.1M outstanding stock options at a weighted average exercise price of $6.57. With the current stock price at $12.38, all options are in-the-money. Using the treasury stock method: proceeds = 1.1M × $6.57 = $7.23M; shares repurchased at $12.38 = $7.23M ÷ $12.38 = 0.584M; net dilution = 1.1M − 0.584M = approximately 0.5M dilutive shares. This 0.5M is added to basic shares and RSUs to complete the FDSO calculation. Source comments (Shift-F2) must document the page number and filing date for every data point entered.

---

### T33. Trading-Comps-Modeling-EXTR-YTD
EXTR's most recent 10-Q (period ending December 31, 2023) provides the stub period data needed for the LTM calculation. The income statement on page 4 of the 10-Q shows: six months ended December 31, 2023 revenue $649.5M, COGS −$253M, gross profit $396M, operating expenses −$350M, non-operating items −$5.9M, pre-tax $40.3M, tax provision −$7.632M, GAAP net income $32.664M. Diluted weighted average shares: 132.786M; D&A for the six-month stub: $12.549M (from the cash flow section of the same 10-Q). The corresponding prior-year stub (six months ended December 31, 2022) is in the same 10-Q filing's comparative column.

---

### T34. Trading-Comps-Modeling-CSCO-Balance-Sheet
Cisco's balance sheet: cash + $11.9B short-term investments = total cash; AR $4.8B, inventory $3.2B, goodwill + intangibles $40.7B, PP&E $2B. Debt: CPLTD $4.9B, with the remainder in long-term. Shareholders' equity $46.251B. At ~$48.86/share × 4,105M diluted shares, CSCO's market cap is approximately $201B; adding net debt of ~−$10B (net cash) gives EV ~$190B. CSCO is a net cash company — its enormous cash balances exceed its gross debt, so EV is actually less than market cap. All figures are already in millions in CSCO's filings, so no scaling is needed.

---

### T35. Trading-Comps-Modeling-EXTR-YTD-Non-GAAP
EXTR's YTD non-GAAP adjustments for the six months ended December 31, 2023 are sourced from the 8K press release filed January 31, 2024. The total operating adjustments for the current stub period are $60.2M; after reversing the tax effect (−$14.857M), adjusted net income for the stub = $78M with diluted EPS of $0.59. The prior-year stub adjustments are found in the same 8K filing or the corresponding prior-period press release. Both stubs are entered to enable the LTM non-GAAP calculation: FYE normalized + current stub normalized − prior-year stub normalized.

---

### T36. Trading-Comps-Modeling-Net-Debt-Components
Net debt components include: CPLTD, notes payable, capital leases (treated as debt equivalents), and long-term debt on the gross debt side. Minority interest (non-controlling interest) is included at its book value as a quasi-debt item. Preferred stock is also included in net debt since preferred holders have a claim senior to common equity. For convertible securities: if conversion into shares is assumed in the FDSO calculation, exclude the convertible principal from net debt to avoid double-counting (it can't be both a share and a debt). CSCO's net cash position (~$10B excess cash over gross debt) results in EV ≈ $190B, which is less than its market cap.

---

### T37. Trading-Comps-Modeling-VMware-Non-Recurring-Example
The VMware example illustrates a complete GAAP-to-non-GAAP income statement walkthrough. VMware's GAAP EPS was $2.34 while non-GAAP EPS was $3.37 — a large difference driven by SBC and amortization exclusions. Revenue is unchanged between GAAP and non-GAAP (no non-GAAP revenue adjustments). The tax line must be adjusted separately: if pre-tax expenses are excluded, the corresponding tax benefit embedded in the GAAP tax provision must also be reversed. The process produces a higher-quality, more comparable earnings figure that aligns with how the analyst community actually discusses and benchmarks the company.

---

### T38. Trading-Comps-Modeling-Non-Recurring-Items-GAAP-vs-Analyst
FASB recognizes three types of truly non-recurring items that are reported below net income: discontinued operations, extraordinary items (now eliminated under ASC 225), and accounting changes. These are universally excluded by both GAAP rules and analysts. Items that FASB considers merely "unusual or infrequent" (but not both) — such as restructuring, asset write-downs, litigation, and severance — are embedded within pre-tax income statement line items. Analysts want to exclude these too but must do so manually via non-GAAP adjustments, and must reverse the associated tax shield. The Walmart example shows restructuring charges of $260M embedded in SG&A; removing them requires a corresponding upward tax adjustment.

---

### T39. Trading-Comps-Modeling-Normalizing-Earnings-Exercise-1
Practice exercise: a company has a $14M inventory writedown in 2011 and an $18M writedown + $12M litigation gain in 2012, with a 35% tax rate. Students must identify the pre-tax adjustments, compute the after-tax impact on reported EPS, and assess whether these items affect forward forecasts (they typically do not, since the goal is to exclude them from the earnings base used for forecasting). The key concept is that non-recurring items appear in historical periods and must be excluded from normalized earnings before applying forward growth rates. Correctly computing the tax impact (35% × pre-tax adjustment) is essential to arriving at the right adjusted net income.

---

### T40. Trading-Comps-Modeling-Lemonade-Stand-Exercise
The Lemonade Stand exercise computes normalized EBITDA and EPS for a hypothetical business. Non-GAAP operating adjustments: SBC ($7M in COGS + $2M in SG&A), restructuring ($4M in COGS + $1M in SG&A). Non-operating adjustments: loss on sale of assets ($12M) and litigation gain ($2M). After applying these adjustments and a 40% tax rate, normalized EPS = $11.34 vs. GAAP EPS = $9.64. EBITDA excludes the non-operating items (interest, non-recurring gains/losses) because EBIT is defined as income from continuing operations and non-operating items fall below EBIT. This exercise reinforces the sign conventions and line-item placement rules for each type of adjustment.

---

### T41. Trading-Comps-Modeling-Football-Field-Overview
The football field chart is a standard valuation output that shows a range of implied share prices from multiple methodologies side by side. For EXTR, the football field includes: 52-week trading range (high/low), broker price target range, DCF range (WACC 7–9%, TGR 0–2%), EV/Revenue range (CY24E), and EV/EBITDA range (CY24E). The chart is built as a stacked bar: an invisible "riser bar" positions each range at the right height, and the visible bar shows only the high-minus-low range. A backup table drives the football field values; the high and low labels in the chart are derived from that backup table using cell references. The chart provides a visual synthesis of how different valuation approaches converge or diverge around a central implied value.

---

### T42. Trading-Comps-Modeling-Balance-Sheet-Items-for-EV
Only balance sheet items relevant to the EV bridge need to be spread in the comps template — not every line item. The key items are: total cash and cash equivalents + short-term and long-term investments (netted against debt), gross debt (CPLTD + notes payable + capital leases + long-term debt), minority interest (at book value), and preferred stock. AR and inventory are useful only for supplemental analyses (e.g., ABL facility sizing). Goodwill and intangibles are useful for tangible book value analysis. Equity method investments (e.g., HPE's H3C stake, JNPR's holdings) represent non-core assets that some analysts adjust out of net debt to isolate operating EV.

---

### T43. Trading-Comps-Modeling-EXTR-Non-GAAP-Overview
EXTR's total non-GAAP operating adjustments for FY June 2023 sum to $90.6M, as derived from the August 2023 8K press release. The company itself guides to $198.9M in adjusted EBIT (= GAAP EBIT $108.3M + $90.6M adjustments). Broker consensus estimates should be used as a cross-check: if analysts are estimating non-GAAP EBIT close to $199M, that confirms your adjustment calculation is directionally correct. Any material discrepancy between your modeled non-GAAP EBIT and the consensus figure signals a data entry or sign convention error that must be investigated before proceeding.

---

### T44. Trading-Comps-Modeling-LTM-Concept
The LTM concept is visualized as: FYE (12 months) + Current YTD (6 months) − Prior Year YTD (6 months) = LTM (12 months). The prior-year YTD period is always available in the same 10-Q filing as the current YTD, so only one document is needed for the stub periods. The subtraction removes the overlap between the full-year 10-K period and the current stub, ensuring the result is a clean rolling 12-month window ending at the most recent quarter. For EXTR (FYE June 30, latest 10-Q December 31), LTM = FY2023 (12 months) + Q1-Q2 FY2024 (6 months ending Dec 31, 2023) − Q1-Q2 FY2023 (6 months ending Dec 31, 2022).

---

### T45. Trading-Comps-Modeling-PE-and-PEG
The PE ratio is fundamentally driven by three variables: return on equity (↑ ROE → higher PE), long-term growth rate (↑ growth → higher PE), and cost of equity (↑ cost → lower PE). A PE comps analysis is only valid if the peer group generally shares similar ROE, growth, and cost of capital profiles — otherwise different PEs reflect genuine fundamental differences, not mispricing. The PEG ratio (PE ÷ long-term growth rate) solves the growth-rate problem, enabling comparison of companies at different stages of their lifecycle. However, PEG inherits all of EPS's limitations: it's an accounting-based, single-period measure susceptible to non-recurring item distortion and is meaningless for negative-EPS companies.

---

### T46. Trading-Comps-Modeling-Peer-Characteristics
Good comparable companies share: the same industry and product/service type, the same customer base and competitive position, similar seasonality and cyclicality, similar growth trajectories and margin profiles, and similar financial leverage. Secondary factors (sometimes used to narrow a large pool) include: stage in the business lifecycle, geographic mix, and R&D intensity. When two companies compete for the same customers and name each other as competitors in their 10-K, that is the strongest indicator of comparability. Even a company that is larger or smaller can be a valid comp if it is in the same competitive orbit — you are comparing multiples (size-normalized) rather than absolute values.

---

### T47. Trading-Comps-Modeling-Picking-Comps
The four main sources for identifying comps are: (1) the target's 10-K Item 1 competition section, (2) Section 7 MD&A (where management discusses competitive dynamics), (3) equity research reports from sell-side analysts (Oppenheimer, B. Riley, Needham for EXTR), and (4) FactSet/Bloomberg/Cap IQ peer screening tools using industry codes and size filters. A merger proxy (S-4 or DEF 14A) fairness opinion is another useful source because the investment bank advising the deal has already done the comparable company work. Client input (directly asking management who their competitors are) is also valuable when available.

---

### T48. Trading-Comps-Modeling-CSCO-FYE-Income-Statement
CSCO's full-year income statement (from 10-K Item 8, page 54): revenue $56.9B, COGS (negative), total operating expenses $20.7B, GAAP EBIT, non-operating benefit, taxes $2.7B, GAAP NI $12.6B. Diluted weighted average shares: 4,105M. Diluted EPS: $3.07. D&A is pulled from the cash flow statement, not the income statement, since it is embedded in various COGS and opex line items. For CSCO, all amounts are already in millions, so no scaling adjustment is needed before entering into the template.

---

### T49. Trading-Comps-Modeling-Comps-Process
The end-to-end trading comps process has four steps: (1) select a peer group of comparable public companies; (2) gather SEC filings (10-K, 10-Q, 8K earnings releases) for each peer; (3) spread and standardize financial data (LTM, calendarized forecasts, non-GAAP adjustments) to calculate EV multiples; and (4) apply the peer group multiple range to the target's operating metrics to derive an implied share price range. The output feeds directly into the football field chart. The process requires both mechanical precision (correct sign conventions, scaling, double-counting fixes) and judgment (peer selection, which adjustments to include, how to interpret outliers).

---

### T50. Trading-Comps-Modeling-Screening-Resources
Resources for building a peer set include: the target's own 10-K and equity research reports (primary); FactSet/Bloomberg/Cap IQ industry screens (secondary); merger proxies and fairness opinions from related deals (tertiary); NAICS/SIC code-based screens; and direct client or management guidance on who they view as competitors. Each resource has blind spots — 10-Ks focus on named competitors but may omit emerging rivals; database screens may include false positives from broad industry codes. The best peer sets are built by triangulating across multiple sources and then culling based on the qualitative comparability criteria in T46.

---

### T51. Trading-Comps-Modeling-Balance-Sheet-Review
The split-screen technique (Ctrl+Alt+S in Excel) keeps ticker symbols visible while scrolling through long balance sheets, reducing the chance of entering data for the wrong company. HPE and JNPR both have equity method investments that must be noted with a source comment — whether to adjust them out of net debt depends on materiality and team convention. All items should be entered as zero if not present (not blank) to signal that the item was reviewed and confirmed absent. For companies reporting in thousands (like EXTR), values must be converted to millions to maintain consistent scaling across the template.

---

### T52. Trading-Comps-Modeling-Calendarization-Review
The calendarization section of the Input tab uses a YEARFRAC function to calculate the precise fraction of each peer's fiscal year that falls within the desired calendar year window. An IF statement wraps the YEARFRAC formula to show #N/A when the fiscal year date input is blank, preventing silent errors. For EXTR (FYE June 30), CY0 = 50% × FY24 + 50% × FY25 for all income statement metrics. For CSCO (FYE late July), CY0 = approximately 58% × FY24 + 42% × FY25. The same date inputs in the header row drive all calendarized metrics automatically, so the analyst only needs to enter the fiscal year end date once.

---

### T53. Trading-Comps-Modeling-Consensus-Review
When pulling consensus estimates from FactSet, use the "total revenues" line (not segment revenues) to get the correct top-line figure. "EBITDA non-GAAP" as labeled in FactSet for this peer group actually represents EBIT-A (EBIT excluding amortization) — analysts in the networking sector consistently exclude amortization of purchased intangibles from non-GAAP EBIT, not just from net income. This means FactSet's non-GAAP EBITDA label can be misleading. The historical non-GAAP NI of $146M (per FactSet) can be used to confirm that the manual non-GAAP calculation was done correctly.

---

### T54. Trading-Comps-Modeling-Wrap-Up
All peer spreads are now complete: CSCO, ANET, HPE, and JNPR have been fully entered with LTM financials, calendarized consensus estimates, non-GAAP adjustments, and net debt. HPE's calendarization: 83% of prior FY + 17% of first forecast year (FYE October 31 → to December 31 is 2 months = 17%). With all inputs complete, the template automatically populates the Operating Metrics and Valuation Metrics output tabs. The next step is to analyze the outputs to identify outliers, understand multiple dispersion, and select an appropriate valuation range for the football field.

---

### T55. Trading-Comps-Modeling-HPE-Equity-Method
HPE's $2.1B equity method investment is its 49% stake in H3C Technologies (a China-based networking JV), disclosed in Note 20 of HPE's 10-K. Equity method investments represent a non-controlling ownership stake in a separately managed entity — the returns flow through the income statement as a share of earnings rather than as revenue. In the comps template, this can optionally be excluded from net debt (treated as a non-core asset) if the analyst believes it does not represent the company's core operational value. Whether or not to make this adjustment should be noted in a source comment and applied consistently across all peers with similar investments.

---

### T56. Trading-Comps-Modeling-Income-Statement-Spread-Review
All peers in the EXTR comp set (CSCO, ANET, HPE, JNPR) share the same EBIT-A issue: their non-GAAP EBIT already excludes amortization of intangibles. This means the D&A add-back to reach EBITDA must use only depreciation (not full D&A), applying the double-counting fix consistently across all five companies including EXTR. For EXTR specifically, LTM = FYE because the June 30 fiscal year end aligns with the period end of its most recent quarter (December 31, 2023 represents Q2, not year-end), so stubs are needed. Consistent application of the EBIT-A convention across all peers is what makes the EBITDA comparison valid.

---

### T57. Trading-Comps-Modeling-JNPR-Equity-Method
JNPR's equity method investments total approximately $452M but are disclosed across multiple balance sheet line items — approximately $110.2M is buried in "prepaid expenses and other current/LT assets," making it easy to miss. Whether to adjust these investments out of net debt is a team judgment call. When making this adjustment, the analyst should add the investment back to enterprise value (or remove it from the cash offset) and note the source in a Shift-F2 comment. Including a partial or complete adjustment for equity method investments has the effect of increasing the company's EV/EBITDA multiple by the amount of the excluded asset.

---

### T58. Trading-Comps-Modeling-Dilutive-Securities-Review
ANET has 2.5M outstanding (not exerciseable) options at a weighted average exercise price of $19.83, all deeply in-the-money given Arista's high stock price. For trading comps, the relevant column is exerciseable options, not total outstanding — only vested, exerciseable options are included. HPE and JNPR disclosed no stock options in their 10-Qs (some companies have discontinued option grants in favor of RSUs). For any peer where options exist, the treasury stock method must be applied to compute net dilution from in-the-money exerciseable options.

---

### T59. Trading-Comps-Modeling-Steps-to-Spread-Peer-Group
The systematic process to spread a peer's financials: (1) locate the latest 10-K and 10-Q to determine whether stubs are needed; (2) if the most recent quarter end = FYE, LTM = FYE (no stubs required); (3) enter the income statement (revenue, COGS, opex, non-op items, taxes, NI) with correct sign conventions; (4) locate and apply non-GAAP adjustments from the corresponding earnings 8K; (5) source D&A from the cash flow section; (6) spread the balance sheet for net debt items; (7) locate dilutive securities and apply treasury stock method. Source comments (Shift-F2) on every cell that has a specific filing reference are non-negotiable.

---

### T60. Trading-Comps-Modeling-YTD-Non-GAAP-Review
The prior-year YTD non-GAAP adjustments for EXTR (six months ended December 31, 2022) show operating adjustments of $42.67M and a tax effect of −$9.6M, resulting in prior-year stub non-GAAP NI of $63.6M and EPS of $0.47. The LTM non-GAAP calculation can then be confirmed: FYE NI + current stub NI − prior stub NI should equal the LTM NI; testing this against the known LTM NI is a critical sanity check before moving to outputs. Any mismatch indicates an error in period selection, sign convention, or scaling that must be corrected.

---

### T61. Trading-Comps-Modeling-EXTR-8K-Review
EXTR's 8K filed August 2, 2023 (reporting FY June 2023 results) contains the most complete non-GAAP reconciliation. The company explicitly discloses adjustments for: SBC, acquisition/integration costs, amortization of intangibles, restructuring charges, litigation charges, system transition costs, debt refinancing charges, and the tax effect. GAAP operating income (EBIT) is $103M per the 8K, which reconciles to the $108.3M EBIT from the 10-K (small timing/reclassification difference). Using the 8K as the primary non-GAAP source ensures you capture items that may not be explicitly broken out in the 10-K footnotes.

---

### T62. Trading-Comps-Modeling-CSCO-FYE-IS-Review
CSCO FY2024 income statement: revenue $56.9B, COGS (negative in template), operating expenses $20.7B subtotal, GAAP EBIT, non-operating benefit (CSCO is a net cash company with significant interest income), provision for income taxes $2.7B, GAAP NI $12.6B. Diluted WA shares 4,105M; diluted EPS $3.07 — this is the GAAP check figure. D&A is sourced from the cash flow statement. CSCO's high cash balance ($25B+) generates substantial interest income that partially offsets interest expense, making its non-operating line an income item rather than a cost.

---

### T63. Trading-Comps-Modeling-Final-Peer-Selection
Final peer selection rationale: CSCO (✓ — named in EXTR's 10-K as competitor; larger scale is acceptable for multiples comparison), ANET (✓ — mutual competitors, similar product focus, comparable revenue scale), HPE (✓ — partial overlap; competes through HPE Networking segment), JNPR (✓ — mutual competitor, explicitly named in broker equity research). Excluded: Sienna, Fabronet, Lumentum, Avi, InFinra, Luna — all deemed too different in business model or scale, or insufficiently data-rich for comps spreading. The final peer set of four is typical for a focused comps analysis in a relatively concentrated industry.

---

### T64. Trading-Comps-Modeling-Operating-Metrics-Tab-Review
The completed Operating Metrics tab shows the peer set sorted by revenue (largest to smallest): CSCO $57B, HPE $28B, ANET $5.5B, JNPR $5.3B, EXTR $1.6B (at bottom). Key observation: Arista is the growth outlier — 14% revenue CAGR (vs. 0-5% for peers) and the highest EBITDA margins (~40%+). CSCO has declining revenue (restructuring/transition). HPE and JNPR show slow growth with similar margins to EXTR. EXTR itself shows declining revenue in FY24 but growing EBITDA — a margin expansion story. Grouping rows (Excel) must be verified; rows should hide when grouped is activated. Arista's outlier status will have significant implications for the valuation metrics tab.

---

### T65. Trading-Comps-Modeling-Valuation-Metrics-Tab-Review
The Valuation Metrics tab confirms the expected outlier pattern: CSCO, HPE, JNPR, and EXTR all trade at EV/Revenue of 1–3.6x and EV/EBITDA of 8–16x, clustering together. Arista trades at a substantial premium across all metrics (EV/Revenue ~13x, EV/EBITDA ~40x) due to its superior growth profile. Because Arista is such an extreme outlier, the median (rather than mean) is the appropriate central tendency statistic. Excluding Arista from the median calculation would be analytically justifiable given its fundamentally different growth trajectory. The football field ranges for EV/Revenue (2–3x) and EV/EBITDA (11–15x) are derived from the range excluding Arista's outlier values.

---

### T66. Trading-Comps-Modeling-CSCO-YTD-Review
CSCO's six-month stub period for the YTD non-GAAP adjustments: Q2 FY2024 (six months ended January 27, 2024), showing revenue approximately $27.5B and GAAP NI $6.272B with diluted EPS $1.54. Both the current six-month period and the prior-year six-month period (Q2 FY2023) are available in the same 10-Q filing's comparative columns. For CSCO, the 10-Q is filed quarterly, and the large company size means non-GAAP adjustment amounts are also very large (multiple billions), requiring careful sign convention management.

---

### T67. Trading-Comps-Modeling-CSCO-Stock-Options
CSCO's most recent 10-Q does not include options disclosures (they are not required in quarterly filings). Checking the annual 10-K reveals that CSCO has no stock options outstanding — the company discontinued its option grant program years ago in favor of RSUs. This can also be verified via FactSet's equity compensation data. When a company has no options, the only dilutive securities are RSUs/PSUs, which are entered directly above-the-line in the diluted shares section without treasury stock method calculation.

---

### T68. Trading-Comps-Modeling-Football-Field-Backup
The football field backup table drives the high and low values of each valuation range bar. For EXTR: EV/Revenue range of 2–3x applied to CY24E revenue → implied price $16.41 (low) to $24.73 (high); EV/EBITDA range of 11–15x applied to CY24E EBITDA → implied price $12.19 (low) to $16.70 (high). The bar height = high − low; the invisible riser = low value. The chart is built as a stacked bar with two series: the invisible riser (formatted with no fill, no border) and the visible range bar. High/low labels are linked directly to backup table cells so they update automatically when ranges change.

---

### T69. Trading-Comps-Modeling-Share-Counts-Recap
EXTR's complete diluted share count: 128.73M basic shares (cover page, 10-Q as of January 26), plus 7.745M RSUs (page 19, 10-Q), plus 0.5M net dilution from 1.1M options at $6.57 (treasury stock method at $12.38), totaling approximately 136.975M FDSO. Every data point has a Shift-F2 source comment documenting the exact filing, page, and date. The RSUs are included above the treasury stock method section because they convert 1-for-1 to common shares with no exercise price required (zero proceeds, full gross dilution = net dilution).

---

### T70. Trading-Comps-Modeling-Convertible-Debt
Convertible debt is treated exactly like convertible preferred stock for dilution testing purposes. Test 1 (in-the-money): market price > conversion price (= face value of convertible bonds ÷ conversion ratio). Test 2 (anti-dilution): diluted EPS < basic EPS when conversion is assumed. For the diluted EPS numerator when testing convertible debt, add back the after-tax interest expense saved if the debt converts (no interest on debt that no longer exists). Only if both tests are met is the convertible included in diluted shares. If the convertible is included in FDSO, its principal must be excluded from net debt (to avoid double-counting it as both debt and equity).

---

### T71. Trading-Comps-Modeling-Colgate-Convertible-Exercise
Colgate exercise: $65 redemption value ÷ 8:1 conversion ratio = $8.13 conversion price < $66.94 market price → in-the-money (Test 1 passes). Basic EPS $4.44 vs. diluted EPS $4.34 → diluted < basic → dilutive (Test 2 passes). Therefore, the 2.4M preferred shares × 8 = 19.2M common shares are included in FDSO. The in-the-money test fails at a redemption value per share of approximately $530 (when $530 ÷ 8 = $66.25 > $66.94, the math tips over). Test 2 fails (becomes anti-dilutive) at a conversion ratio ≤ 3, because at that point the preferred dividend benefit to basic EPS exceeds the dilutive impact of new shares.

---

### T72. Trading-Comps-Modeling-Convertible-Preferred-Theory
Convertible preferred stock requires two sequential tests before including in the diluted share count. Test 1 (in-the-money): compare the market share price to the conversion price (= liquidation/redemption value per preferred share ÷ conversion ratio). Test 2 (anti-dilution): calculate diluted EPS assuming conversion; if diluted EPS < basic EPS, it is dilutive and should be included. A high preferred dividend yield can make conversion anti-dilutive: preferred holders receive a larger pro-rata share of earnings via dividends than common holders would receive via the converted shares — in that case, preferred shareholders would rationally refuse to convert. When both tests pass, the convertible preferred is included in FDSO and any preferred dividends are removed from the basic EPS numerator.

---

### T73. Trading-Comps-Modeling-Restricted-Stock
Restricted stock that has already vested is included in the basic share count by definition — it is just common stock. Unvested restricted stock (RSUs and RSAs that have not yet passed their vesting conditions) is typically excluded from the diluted share count in trading comps and standalone DCF analysis. A theoretically sounder approach would apply an illiquidity discount to unvested shares and include them, since most will ultimately vest. However, the prevailing practitioner convention is to exclude them entirely, with a caveat that this understates market cap when the unvested pool is material. For transaction comps, the convention flips: unvested restricted stock is generally included because change-of-control often triggers automatic vesting.

---

### T74. Trading-Comps-Modeling-Splits-and-Dual-Classes
Stock splits must be verified before entering any share count — if a split occurred between the 10-K/Q cover date and today's share price, the historical share count is stale (it reflects pre-split shares) while the observable price is post-split. Bloomberg CACS (corporate actions) is the easiest verification tool; alternatively, search SEC filings for 8-K filings announcing splits. Companies with dual-class share structures (e.g., Google Class A and Class B) should include both classes in the diluted share count, since both classes have identical economic claims on earnings and dividends — only their voting rights differ. Source comments on the cover page share count should specify the exact date of the share count disclosure.

---

### T75. Trading-Comps-Modeling-Shares-Outstanding-Overview
The "pie slice" analogy: the observable share price is the price per slice of the total equity pie; to reconstruct the full pie value (market cap), you must count all slices correctly — including dilutive securities that represent future slices. Dilutive securities include: stock options (in-the-money exerciseable; treasury stock method applies), warrants (mechanically identical to options), convertible bonds (bond disappears upon conversion, shares appear), and convertible preferred (preferred disappears, shares appear). The fundamental reason diluted shares matter: an investor observing a $10 stock price is observing the market's estimate of per-share value accounting for all potential future dilution; using basic shares to reconstruct market cap will consistently understate it.

---

### T76. Trading-Comps-Modeling-Bar-Charts-for-Pitch-Decks
Bar charts presenting trading multiples in pitch decks are typically formatted as grouped bars (one bar per peer company per multiple) with a dotted median line. Forward multiples (CY24E) are preferred over LTM because the market is pricing future, not historical, cash flows. The target company is differentiated by a distinct fill color to stand out from peers. Separate charts are typically presented for EV/Revenue and EV/EBITDA (the two most commonly cited enterprise value multiples). The visual convention makes it immediately apparent whether the target trades at a premium, discount, or in line with the peer group median, which forms the basis of the valuation narrative.

---

### T77. Trading-Comps-Modeling-FAQs
Common comps FAQ answers: Price/book is most useful for financial institutions (banks, insurance companies) where assets and liabilities are marked to market and book value approximates fair value. EV/EBIT solves leverage differences but not D&A accounting noise. EV/EBITDA solves both and is the most popular enterprise value multiple for most industries. EV/Revenue is used as a last resort when EBITDA is negative. Industry-specific multiples: internet companies use EV/monthly subscribers; oil & gas companies use EV/EBITDAX (adding back exploration expenses); REITs use P/FFO or P/AFFO (levered cash flow proxies). Using the wrong multiple for the wrong industry context leads to spurious conclusions about relative valuation.

---

### T78. Trading-Comps-Modeling-DCF-vs-Comps
DCF and comps are complementary, not competing, valuation methods. DCF provides intrinsic value based on the company's own projected cash flows and discount rate — rigorous but highly sensitive to assumptions. Comps provide a market-derived sanity check: if the market is pricing peers at 12x EBITDA, a DCF-derived value at 30x EBITDA should prompt a reassessment of assumptions. Both methods are implicit versions of the same underlying value equation (present value of future cash flows) — DCF makes the assumptions explicit while comps embed them implicitly in the market price of peers. Limitations of comps: truly comparable companies are hard to find; thinly traded or poorly followed stocks may not reflect fundamental value; the entire market can be wrong (bubble periods). Despite this, include the target in its own peer group — the logic is that the market is correct on average across companies even if wrong on individual names.

---

### T79. Trading-Comps-Modeling-Overview-and-Standardization
The core purpose of the comps standardization process is to make financially and operationally dissimilar companies comparable on a like-for-like basis. EV multiples solve the leverage problem; EBITDA and revenue metrics solve the D&A accounting differences problem. Lease classification differences (operating vs. capital leases) and inventory accounting differences (LIFO vs. FIFO) matter in capital-intensive and retail industries and require explicit standardization. Non-recurring items must be scrubbed consistently: a company that excludes restructuring charges from EBIT should have peers treated the same way. Business lifecycle differences (early-stage vs. mature) can be addressed via PEG for growth rate, or EV/Revenue for negative-EBITDA companies.

---

### T80. Trading-Comps-Modeling-Typical-Comps-Output
Real pitch deck example from an elite boutique advising in travel and hospitality: the output table shows company names, revenue CAGR, leverage (debt/EBITDA), and multiples for EV/Revenue, EV/EBITDA, EV/EBIT, and P/E — both LTM and forward. This format communicates to the client that the analyst understands the key value drivers (growth + margins + leverage) and how those translate to multiples. The football field applies an IBUTPO multiple range of 7–12x EBITDA to derive an implied share price range. The management case scenario is labeled separately within the football field to distinguish management's own projections from consensus estimates.

---

### T81. Trading-Comps-Modeling-Understanding-Calendarization
Without calendarization, a company with a June 30 fiscal year end would have a "CY forward" multiple based on results only 3 months from the February 23 price date, while a December 31 company's forward period is 9 months away — the June company appears cheaper solely because its "forward" year is almost complete. Calendarization fixes this by adjusting all companies' operating metrics to a common December 31 endpoint. For EXTR (FYE June 30), CY1 (calendar year 2024 estimate) = 50% × FY24 (Jul 2023–Jun 2024) estimate + 50% × FY25 (Jul 2024–Jun 2025) estimate. Applied to EXTR's EPS: 50% × $0.57 + 50% × $0.83 = $0.70 CY1 EPS. In some client situations, the analysis may be calibrated to the client's own fiscal year end instead of December 31.

---

### T82. Trading-Comps-Modeling-Unusual-or-Infrequent-Items
FASB distinguishes between "truly non-recurring" items (below the income statement: discontinued operations, extraordinary items) and "unusual or infrequent" items (embedded in the income statement: restructuring, one-time write-offs, gains/losses on asset sales, severance, litigation). The latter category must be reported within pre-tax operating line items per GAAP, even though analysts prefer to exclude them for comparability. When analysts exclude an unusual/infrequent item, they are effectively "pretending it didn't happen," which requires also reversing the associated tax shield. The Walmart example illustrates: $260M restructuring embedded in SG&A; analyst would report operating SG&A as if the $260M wasn't there, but must also add back the tax benefit that Walmart received from having that extra expense.

---

### T83. Trading-Comps-Modeling-Valuation-Overview
Enterprise value = value of operating assets − operating liabilities (left-side perspective); equivalently, EV = equity value + net debt (right-side / funding perspective). This is not a new equation but a reformulation of the basic accounting equation (assets = liabilities + equity). The hot dog stand example: $950K invested (debt + equity) → buys inventory ($500K), PP&E ($400K, with $20K invoiced), keeping $50K cash. EV = operating assets ($920K) − operating liabilities ($20K A/P) = $900K; net debt = $500K debt − $50K cash = $450K; equity value = $900K − $450K = $450K, which equals the original equity contribution. Relative valuation (comps) asks "what are similar businesses worth in the market today?" while DCF asks "what is this business worth based on its own cash flow potential?"

---

### T84. Trading-Comps-Modeling-Whats-Ahead-Spreading-Extreme-Networks
This introductory video previews the hands-on trading comps modeling exercise that follows. Students will identify a peer group for Extreme Networks, gather SEC filings for each peer, and build out the complete comps model to arrive at an implied share price range. The video reiterates that all prior conceptual videos — EV, multiples, calendarization, non-GAAP adjustments, diluted shares — have been preparation for this case study application. By the end of the full exercise, students will be able to spread a real-world comparable company analysis from scratch using only SEC filings and consensus data.

---

## Transaction Comps Modeling (T85–T129)

---

### T85. Transaction-Comps-Modeling-Case-Study-Introduction
Transaction comps value a company by analyzing the multiples paid in past acquisitions of comparable businesses. For EXTR, the transaction comp search combined: (1) a FactSet industry screen (SIC 3577), (2) acquisitions by peers (Cisco, HPE, Juniper, Brocade, EXTR itself), and (3) fairness opinions from the Foundry-Brocade S4 and the HP-3Com proxy. Arista's recently filed S-1 IPO registration also listed key landmark deals in the sector. The final four comparable transactions chosen: Extreme-Enterasys (2013), Brocade-Foundry (2008), HP-3Com (2009), and Thoma Bravo-Blue Coat Systems (2011). Cisco-Tanberg and several others were considered but excluded as insufficiently similar or lacking data.

---

### T86. Transaction-Comps-Modeling-Conclusion
Transaction comps modeling becomes more comfortable with repetition, but each deal presents unique disclosure challenges that require judgment rather than rote procedure. The key strengths of the analysis are identifying acquisition premiums and multiples that a potential acquirer would need to pay, which are inherently higher than trading comps multiples (they embed the control premium). Common pitfalls include over-reliance on FactSet/Cap IQ data without verifying against primary SEC filings, and missing obscure disclosures for private targets. The course urges practitioners to always be aware of the analysis's limitations while leveraging its insights for M&A valuation context.

---

### T87. Transaction-Comps-Modeling-Finding-Comparable-Transactions-Part-1
The three main sources for finding comparable transactions are: (1) colleagues and internal comp sets from prior deals, (2) M&A screening databases (FactSet, Cap IQ, Bloomberg, Thompson), and (3) fairness opinions embedded in S-4 filings or DEF 14A proxies of comparable company deals. Database screens use SIC codes, revenue/EBITDA size filters, deal type (merger/acquisition), and announcement date range as primary parameters; secondary parameters include industry position, leverage, and cyclicality. Screening databases are better for finding deals than for accurate financial data — always verify against primary SEC filings because database errors are common. The rule of thumb on staleness: try not to use deals older than 6-7 years, as premiums and multiples are environment-specific.

---

### T88. Transaction-Comps-Modeling-Finding-Comparable-Transactions-Part-2
Live FactSet screen for EXTR: start with EXTR's ticker to find its SIC code (3577 — Computer Peripheral Equipment NEC). Screen parameters: date range 2007 to present, deal size >$5M, deal type = acquisition/merger or majority stake, target financial filter = revenue >$50M (to eliminate private deals with no data). Applying SIC 3577 yields 37 transactions — a broad initial pool. A second screen by acquirer name (Brocade, CSCO, JNPR, HPE, EXTR itself) yields 7 more focused results. Using both screens simultaneously provides more coverage than either alone; the Brocade-Foundry deal emerges from the peer screen and will prove to be the most directly relevant transaction.

---

### T89. Transaction-Comps-Modeling-Finding-Comparable-Transactions-Part-3
Fairness opinions are among the most valuable secondary sources for transaction comps. When a public company acquires a public company, an S-4 is filed; when a public company acquires a private company, a DEF 14A proxy is filed. The investment bank advising the deal prepares a fairness opinion, which is included in these filings and contains its own comparable transactions analysis. For EXTR: Brocade's S-4 for the Foundry acquisition (filed ~August 2008) includes Merrill Lynch's fairness opinion with ~10 comparable transactions. To find the fairness opinion: go to EDGAR, search the acquirer's filings for the S-4 or proxy filed within weeks of the deal announcement date, then search the document for terms like "fairness" or "comparable transactions."

---

### T90. Transaction-Comps-Modeling-Finding-Deal-Terms
Key deal term documents: (1) announcement 8-K press release (offer price, deal structure, preliminary synergy disclosure), (2) closing press release 8-K (may contain revised terms), (3) merger agreement (detailed terms, usually filed as 8-K exhibit and embedded in proxy) — includes capitalization section with most timely share count and dilutive securities, (4) DEF 14A proxy or S-4 (shareholder vote documents containing fairness opinion with projections), (5) tender offer 14D-1 for tender offers. Historical share prices for computing premiums are available from FactSet/Bloomberg; for delisted targets, prices are often disclosed in the fairness opinion itself when database access is unavailable.

---

### T91. Transaction-Comps-Modeling-Finding-Dilutive-Shares-Data-Part-1
Transaction comps diluted shares are more complex than trading comps. When deal terms are disclosed as a per-share price (e.g., HP offering $7.90/share for 3Com), correctly calculating diluted share count is essential to computing total offer value. Basic shares: use the latest pre-deal 10-Q cover page, or the merger agreement capitalization section (often more timely, disclosing shares as of the day before closing). For dilutive securities in transaction comps: options use "outstanding" (not exerciseable) because change-of-control triggers automatic vesting. Restricted stock: include vested-and-expected-to-vest RSUs/RSAs because they too often vest automatically at deal close. This differs from trading comps convention (where unvested restricted stock is excluded).

---

### T92. Transaction-Comps-Modeling-Finding-Dilutive-Shares-Data-Part-2
The timing mismatch problem: basic shares in the merger agreement (most recent) use a different date than options data in the 10-K (stale). Best practice: use the most recent basic share count (merger agreement or 10-Q cover) and the most recent options data that includes an exercise price (usually the 10-K). In practice, analysts almost always ignore the mismatch unless there is a specific reason to believe options were massively exercised between the two dates (visible as a large increase in basic shares). GVB example: 101M basic (from merger agreement) + 3M options at $4 exercise price (from 10-K) at $8 offer price → net dilution = $12M proceeds ÷ $8 = 1.5M repurchased → 1.5M net dilution → 102.5M FDSO.

---

### T93. Transaction-Comps-Modeling-Finding-Target-Financials
For transaction comps, the relevant financial statements are those the acquirer had access to as of the announcement date. If the target's Q2 ended June 15 but results weren't announced before the July 12 deal announcement, determine whether the acquirer had access (in friendly deals, usually yes). For public targets: use the latest publicly available 10-K/10-Q adjusted for materiality timing. For private targets: search the acquirer's 8-K filings between announcement and close — material acquisitions often require the acquirer to file the target's financials. FactSet sometimes fails to find these disclosures (as happened with Enterasys); always search SEC EDGAR directly. When both parties are private, rely on press quotes and LexisNexis searches.

---

### T94. Transaction-Comps-Modeling-GAAP-to-Non-GAAP-Adjustments-Part-1
For Foundry Networks (target in Brocade deal): FY2007 earnings press release (8K) shows GAAP NI $81M → non-GAAP NI $118M. Adjustments: SBC (broken out by COGS vs. opex), stock option investigation costs (disclosed by footnote), income tax effect reversal. To apply: identify each adjustment's line-item location in the income statement, enter as a positive add-back (reversing the expense), and enter the tax effect as a negative (reversing the tax benefit). After applying all adjustments, normalized EBIT and non-GAAP NI should exactly match company-disclosed figures. EBITDA is calculated by adding D&A from the cash flow statement to normalized EBIT.

---

### T95. Transaction-Comps-Modeling-GAAP-to-Non-GAAP-Adjustments-Part-2
EBITDA for the Foundry deal: add back D&A of $11M (from press release cash flow section, or 10-K cash flow statement) to normalized EBIT → normalized EBITDA = $153M. The diluted EPS formula (non-GAAP NI ÷ diluted WA shares) should produce a figure matching the company-disclosed non-GAAP EPS of $0.76. A minor discrepancy can occur when the company uses a different share count (e.g., different anti-dilution treatment) for non-GAAP EPS than the analyst's model uses. The template includes numerator/denominator adjustment rows to allow manual reconciliation to the company-disclosed figure in these cases.

---

### T96. Transaction-Comps-Modeling-Inputting-Basic-Shares-Data
For the Brocade-Foundry deal: Foundry's latest 10-Q (filed August 5, 2008, covering Q2 ended June 30) shows 147M shares as of July 31. The merger agreement (in the proxy, section 2.3, page A-110) shows 144.9M shares as of July 18, 2008 — three days before the deal announcement. The merger agreement figure is more timely and preferable because it captures shares as of the day before the deal, before any post-announcement share activity. Best practice: always check the merger agreement capitalization section for a more precise share count, and update the model with the merger agreement data if it is more recent than the 10-Q cover page.

---

### T97. Transaction-Comps-Modeling-Inputting-Deal-Terms
Brocade-Foundry deal inputs: announcement date July 21, 2008 (per 8-K); deal completed December 19, 2008 (per closing press release). Final terms: $16.50/share all cash (original announcement was $18.50 cash + 0.0907 Brocade shares; terms were revised in the proxy). 100% acquisition. Synergies: $33M + $12M = $45M/year beginning 2010, found by Googling a news article quoting management's conference call. Unaffected share prices: 1-day prior $13.36, 1-week prior $11.61, 1-month prior $13.19 — three data points are used because rumors of the deal may inflate the price the day before, making the 1-day figure an unreliable "unaffected" price.

---

### T98. Transaction-Comps-Modeling-Inputting-Dilutive-Shares-Part-1
HP-3Com deal ($7.90/share all cash, November 11, 2009): basic shares 392M from 10-Q cover (September 25, 2009). Options: 23.8M vested-and-expected-to-vest at $5.05 average exercise price (all in-the-money vs. $7.90 offer); treasury stock method → $120M proceeds ÷ $7.90 = 15.2M repurchased → 8.6M net dilution. RSUs: 1.3M RSAs + 10.5M RSUs (from 10-Q); proxy confirms no automatic vest, but best practice includes them anyway. Net debt: $690M cash (including deposits treated as cash equivalents) − $200M gross debt = −$490M (net cash position). Transaction value = offer value − net cash = $2.769B slightly above company's stated "approximately $2.7B" (difference due to RSU inclusion).

---

### T99. Transaction-Comps-Modeling-Inputting-Dilutive-Shares-Part-2
HP-3Com merger agreement (proxy page A-19, section capitalization): 394M shares as of November 9, 2009, of which 1M are unvested restricted stock included by the company in the total. Strip the 1M: 393M clean basic shares. Options: merger agreement shows 24M but no exercise price → use 10-Q's 23.8M at $5.05. Restricted stock: 10.4M per merger agreement (11.5M from 10-Q minus 1.1M already in basic count). Lesson: always read the full paragraph when using merger agreement share counts — companies sometimes include unvested restricted shares in the total, requiring a manual strip-out. The merger agreement data is more timely but must be interpreted carefully.

---

### T100. Transaction-Comps-Modeling-Inputting-LTM-Financials
LTM = FY2007 10-K (12 months) + Q2 2008 10-Q stub (6 months ended June 30, 2008) − Q2 2007 stub (6 months ended June 30, 2007, in same 10-Q). For Foundry: the deal was announced July 21, 2008, but the Q2 10-Q was filed August 5, 2008 (two weeks post-announcement). However, since Brocade's announcement occurred the same day as Foundry's Q2 press release, Brocade almost certainly had access to Q2 data — so Q2 is used as the LTM cut-off. FY2007 10-K income statement: revenue $607M, COGS, opex, EBIT $82M (GAAP), interest income (net cash company), taxes, NI, diluted WA shares 155.5M. All expenses entered as negatives; scale adjusted to millions via Boost.

---

### T101. Transaction-Comps-Modeling-Inputting-LTM-Financials-Part-1
For 3Com (FY May 31, 2009): use the annual press release (8-K filed July 9, 2009) rather than the 10-K because the press release contains the GAAP-to-non-GAAP reconciliation. GAAP NI $114M → non-GAAP NI $176M (+EPS $0.45). Key adjustments: restructuring +$8M (opex), amortization of intangibles +$95M (opex), patent dispute, patent sale income −$85M (this is non-recurring income that must be subtracted — normalized NI is lower than GAAP in this period). SBC $29M split between COGS ($3M) and opex ($26M). D&A: $124M total from 10-K cash flow − $95M amortization already excluded = $29M add-back for EBITDA. EBITDA non-GAAP ≈ $205M; GAAP EPS $0.29 vs. non-GAAP $0.45.

---

### T102. Transaction-Comps-Modeling-Inputting-LTM-Financials-Part-2
Q1 FY2009 stub (3 months ended August 28, 2009): GAAP NI $7.4M, EPS $0.02; non-GAAP adjustments (SBC in COGS + opex, amortization, restructuring) → non-GAAP EPS $0.08. D&A from 10-Q: $21M total − $17M amortization = $4M add-back. Prior-year Q1 stub (August 29, 2008): non-GAAP EPS $0.11 after excluding large patent litigation gain. LTM results: revenue $1.2B, EBITDA $166M, EBIT $139M, LTM EPS $0.42. These LTM metrics are the denominators for all LTM multiples. The model can now compute transaction value ÷ LTM EBITDA and other LTM multiples for this comp.

---

### T103. Transaction-Comps-Modeling-Inputting-Options-Data
For Foundry: 10-Q shows 29.8M options vested-and-expected-to-vest at weighted average exercise price of $16.28 (just below the $16.50 offer price). All 29.8M are in-the-money. Treasury stock method: $485M proceeds ($29.8M × $16.28) ÷ $16.50 offer = 29.4M shares repurchased → only 0.4M net dilution. For deals where the acquirer is acquiring less than 100%: the template has a "shares required to acquire 100%" row that normalizes the offer value calculation to the full enterprise, even though only a majority stake was purchased. This avoids understating the implied transaction value.

---

### T104. Transaction-Comps-Modeling-Inputting-Projections
Transaction comps projections are found in fairness opinions embedded in the proxy/S-4. For Foundry: Merrill Lynch's fairness opinion (in the Brocade proxy) contains management case forecasts: revenue $638.6M (2008), $705M (2009). EBITDA for 2008 not explicitly provided; estimated as EBIT forecast + prior-year D&A (with a source comment explaining the estimate). EPS: $0.67 (2008). Note: only year-1 and year-2 forward multiples are needed in most transaction comps analyses — year-3 and beyond are rarely used. When data is incomplete, a reasonable estimate with a clear explanatory comment is better than leaving the field blank.

---

### T105. Transaction-Comps-Modeling-Inputting-Projections-1
For 3Com: Goldman Sachs fairness opinion (3Com proxy, page 38) provides management sensitivity case: revenue FY2010 $1.25B, FY2011 $1.4B; EBIT $113M and $141M; EPS $0.32 and $0.38 (non-GAAP). EBITDA not provided; estimated as EBIT + prior year D&A assumption with source comment. Summary of four comps completed: Enterasys has far lower revenue and EBITDA multiples than others (small company, low margin); Foundry and 3Com multiples are broadly comparable on revenue (~2.2x) but 3Com has significantly higher EBITDA multiples (reflecting higher growth premium paid by HP). Median across comps is the appropriate central tendency given the outlier variation.

---

### T106. Transaction-Comps-Modeling-Inputting-RSU-Data
Foundry RSU/restricted stock data (from 10-Q): 638K restricted stock awards + 1.379M vested-and-expected-to-vest RSUs. Proxy confirms: restricted stock stays unvested (converted to right to receive $16.50 cash held in escrow until vesting restrictions lapse) — no automatic vest at deal close. Despite this, best practice in transaction comps includes these shares in diluted count: expected-to-vest RSUs and RSAs represent a real claim on equity that the acquirer must plan for. Fully diluted shares for Foundry: 144.9M basic + 0.4M options + 2.0M restricted = ~147.3M. Offer value = 147.3M × $16.50 = $2.43B, consistent with FactSet's reported $2.4B.

---

### T107. Transaction-Comps-Modeling-Introduction-to-Transaction-Comps
Transaction comps use the same house-comparison logic as trading comps, but the "recent sale prices" are actual acquisition prices rather than market prices for publicly traded companies. Multiples in transaction comps: TV/EBITDA, TV/Revenue, TV/EBIT (enterprise value multiples), offer price/EPS, offer value/NI (equity value multiples). "Transaction value" (TV) is used interchangeably with enterprise value in the M&A context. The key distinction from trading comps: transaction multiples include an acquisition premium (often 25–50%+ above standalone market value), so transaction EV/EBITDA multiples will be structurally higher than trading comps EV/EBITDA multiples for the same company set.

---

### T108. Transaction-Comps-Modeling-Net-Debt
The template includes an "override" formula that allows the analyst to hard-code a total offer value when per-share deal terms are not disclosed (as with the Enterasys deal). The formula structure: IF(override cell has value, use override; ELSE, calculate price × diluted shares). For Foundry: gross debt $0 (no debt on balance sheet); cash $257M + $590M short-term investments + $100M long-term investments = ~$947M total cash. Net debt = −$947M. Transaction value = offer value ($2.43B) − net cash ($947M) = $1.483B. The enterprise value effectively measures only the operational business value; the cash was a separate asset acquired as part of the deal.

---

### T109. Transaction-Comps-Modeling-Review
After spreading the Foundry-Brocade deal completely, the summary section shows all calculated metrics: LTM revenue, EBITDA, EBIT, EPS; year-1 and year-2 forecasts for each; and TV/Revenue, TV/EBITDA, TV/EBIT, and P/E multiples for each period. The output tab uses an OFFSET + MATCH function to pull specific comp data by target name and metric, enabling a flexible dropdown-driven summary that can include or exclude any comp from the analysis. The OFFSET function navigates rows (matched by metric name) and columns (matched by target company name) simultaneously, pulling values from the raw input data without reformulation.

---

### T110. Transaction-Comps-Modeling-Review-and-the-Output-Tab
With all four comps spread (Foundry, Enterasys, Blue Coat, 3Com), the output tab allows flexible arrangement via a dropdown for each column position. Deal announcement dates range 2008–2013; revenues range from $300M (Enterasys) to $1.2B (3Com). Key multiples: revenue multiple ~2.2x (excluding Enterasys outlier); EBITDA multiple ~10–12x; EBIT ~14x; P/E ~21x. Enterasys is an outlier on the low end (small company, low margin, distressed profile at time of acquisition). The OFFSET/MATCH formula explained: OFFSET(start_cell, row_match, col_match) where row_match finds the metric name and col_match finds the company tag — copy/paste requires paste-values-only (Alt+E+S+V) to preserve formatting.

---

### T111. Transaction-Comps-Modeling-Review-PB-EVRev-EVEBIT-EVEBITDA-Industry-Multiples
Price/book: only useful for financial institutions where assets and liabilities are marked to market; book value approximates fair value; P/book premium implies market values future ROE above book. EV/EBIT: solves leverage; not D&A differences; best for low-capex service businesses. EV/EBITDA: solves both leverage and D&A; most popular enterprise multiple; best for similar-capital-intensity peers. EV/Revenue: last resort; implicit assumption of identical cost structures; useful for negative-EBITDA or early-stage companies. Industry-specific: internet/cable → EV/subscribers; oil & gas → EV/EBITDAX; REITs → P/FFO or P/AFFO. Underlying value drivers for all: ROIC, growth rate, WACC.

---

### T112. Transaction-Comps-Modeling-Review-PE-PEG-Ratios
P/E defined as price/EPS or equity value/NI (slight difference when share count changed during year). EPS is an accounting measure of a single period — not a direct proxy for cash flow; manipulable via accruals, depreciation choices, non-recurring items. Three fundamental PE drivers: ROE (↑→higher PE), growth rate (↑→higher PE), cost of equity (↑→lower PE). PE is most appropriate for mature companies with positive EPS and similar capital structures. PEG = PE ÷ long-term growth rate; solves for growth differences but inherits all EPS limitations; meaningless for negative-EPS companies. In the transaction comps context, P/E appears as offer price/EPS or offer value/NI, and tends to carry higher multiples than trading comps due to the acquisition premium.

---

### T113. Transaction-Comps-Modeling-Review-Shares-Outstanding-Convertible-Debt
Convertible debt uses the exact same two-test framework as convertible preferred stock. Test 1: compare market share price to conversion price (= convertible face value ÷ conversion ratio); if in-the-money, proceed. Test 2: compare diluted EPS (assuming conversion; add back after-tax interest expense as numerator adjustment since the interest is saved upon conversion) to basic EPS; if diluted < basic, it is dilutive → include in FDSO. The key difference from convertible preferred: the numerator adjustment reverses after-tax interest expense (not preferred dividends) when testing. If both tests pass, exclude the convertible principal from net debt (it converts to equity — cannot be counted as both debt and equity).

---

### T114. Transaction-Comps-Modeling-Review-Shares-Outstanding-Convertible-Exercise
Colgate 2010 exercise: 2.4M preferred shares, $65 redemption price, 8:1 conversion ratio, $16/share preferred dividend, $66.94 stock price, 87.8M basic WA shares, $2.203B net income. Test 1: conversion price = $65/8 = $8.13 < $66.94 → in-the-money. Test 2: basic EPS = ($2.203B − 2.4M×$16) ÷ 87.8M = $4.44; diluted EPS = $2.203B ÷ (87.8M + 2.4M×8) = $4.34 → diluted < basic → dilutive. Both tests pass → include 19.2M converted shares in FDSO. In-the-money test fails when redemption value ≥ ~$530/share (conversion price > market price). Anti-dilution test fails when conversion ratio ≤ 3 (dividend yield dominates dilution effect).

---

### T115. Transaction-Comps-Modeling-Review-Shares-Outstanding-Convertible-Preferred-Stock
Two sequential tests for convertible preferred: Test 1 (in-the-money) — conversion price (= liquidation/redemption value per preferred share ÷ conversion ratio) < current market share price. Test 2 (anti-dilution) — diluted EPS assuming conversion < basic EPS (where basic EPS net of preferred dividends is the baseline). High preferred dividends make conversion anti-dilutive: the dividend yield exceeds the dilution impact of new shares, so preferred shareholders rationally prefer to keep the dividends rather than convert. The three scenarios (neutral, dilutive, anti-dilutive) turn on whether the preferred dividend payout as a % of net income is less than, equal to, or greater than the dilution % from converted shares.

---

### T116. Transaction-Comps-Modeling-Review-Shares-Outstanding-Options
Basic shares: cover page of latest 10-K or 10-Q. Options: search "exerciseable" in 10-K footnotes to find the options table with tranche-by-tranche breakout of outstanding, exerciseable, and weighted average exercise price. For trading comps: use exerciseable options (vested, can be exercised today). For transaction comps: use outstanding options (change of control triggers auto-vest, making all outstanding options exerciseable). In-the-money filter: exercise price < current share price (or offer price for transaction comps). Treasury stock method: proceeds from exercise ÷ current/offer price = shares repurchased; net dilution = gross options − shares repurchased.

---

### T117. Transaction-Comps-Modeling-Review-Shares-Outstanding-Overview
The "pie slice" analogy explains why diluted shares matter: observable share price is the price per slice; multiplying by the wrong number of slices (basic instead of diluted) will understate market cap. Dilutive securities: options (treasury stock method), warrants (same as options, attached to convertible debt), convertible bonds (bond disappears upon conversion), convertible preferred (preferred disappears, shares appear). The zero-exercise-price simplification shows why all dilutive securities should conceptually be in the share count — only practical constraints (exercise prices, vesting, anti-dilution tests) determine the actual inclusion/exclusion. The market's observable share price already reflects all expected future dilution, so the analyst must use the same diluted count to reconstruct market cap correctly.

---

### T118. Transaction-Comps-Modeling-Review-Shares-Outstanding-Restricted-Stock
Vested restricted stock is already common stock — it's in the basic share count. Unvested restricted stock (RSUs, RSAs) is typically excluded in trading comps and standalone DCF analysis per practitioner convention. A theoretically sounder approach would discount unvested shares for the probability of forfeiture and apply an illiquidity discount, then include a fraction in diluted shares. However, virtually no practitioner does this in trading comps. The limitation matters when the unvested pool is large — F5 Networks example: 926K unvested RSUs excluded, which is relatively small as a % of total shares outstanding. For transaction comps, the convention flips to inclusion because change-of-control typically triggers vesting.

---

### T119. Transaction-Comps-Modeling-Review-Shares-Outstanding-Splits-Dual-Classes
Stock splits: any split between the 10-K/Q filing date and today's share price requires adjusting all historical share counts and dilutive securities by the split ratio. Missing a split will dramatically misstate market cap (post-split price × pre-split share count = large understatement). Bloomberg CACS is the fastest check; alternatively, search the company's EDGAR filings for 8-Ks or form DEF 14A amendments announcing splits. Dual-class shares (Google Class A and Class B): both classes count in the diluted share base because both have identical economic rights to earnings and dividends — only voting rights differ. Always include both classes in market cap calculation and label with a source comment.

---

### T120. Transaction-Comps-Modeling-Review-Trading-Comps-FAQs-and-Common-Misconceptions
DCF vs. comps: both needed; comps = market sanity check, DCF = intrinsic rigor. Both are actually the same underlying valuation model (DCF makes assumptions explicit; comps embed them implicitly in peer multiples). Downside of overreliance on comps: truly comparable companies are rare; thinly traded stocks may not reflect fundamental value; the market can be wrong in aggregate (bubbles). For public company valuation, include the target in its own peer group — the logic is that the market is efficient on aggregate, not individual, so the target's market price is one data point in the correct pool. Median vs. mean: median for larger groups (outlier protection); mean for small groups with no outliers. LTM vs. forward: use both; weight forward more when LTM has noise or non-recurring items.

---

### T121. Transaction-Comps-Modeling-Review-Trading-Comps-Overview
The main sources of non-comparability that must be standardized: (1) size differences — solved by using multiples (ratios) not absolute values; (2) leverage — solved by EV numerator (enterprise value is leverage-neutral); (3) D&A accounting differences (depreciation methods, useful life) — solved by using EBITDA or revenue as denominator; (4) non-recurring items — must be manually scrubbed; (5) lease classification (operating vs. capital) — matters for capital-intensive/retail sectors; (6) inventory method (LIFO vs. FIFO) — matters for companies with significant inventory; (7) business lifecycle — growth-stage companies use PEG or EV/Revenue; mature companies use EV/EBITDA or P/E. None of these adjustments are automatic — each requires analyst judgment about materiality.

---

### T122. Transaction-Comps-Modeling-Review-Valuation-Overview
Identical to T83 (Trading Comps) — this is a review module inserted for students entering the Transaction Comps course without completing Trading Comps first. EV = operating assets − operating liabilities (left-side perspective); equivalently EV = equity + net debt (right-side). Hot dog stand example: equity invested $450K + debt $500K = $950K cash invested → inventory + PP&E purchased → EV = operating assets ($920K) − operating liabilities ($20K A/P) = $900K; equity = $900K − $450K net debt = $450K. Google comparison: $75B book equity vs. $290B market cap, $245B EV — illustrates how market value can dramatically exceed book value for businesses with strong cash-generating capabilities.

---

### T123. Transaction-Comps-Modeling-Simple-Transaction-Comps-Exercise
Step sequence for transaction comps: select comparable transactions → calculate mean/median multiples → apply to target → arrive at implied transaction and equity value. Eli Lilly exercise: $50/share, 1B shares outstanding, $20B revenue, $6.5B EBITDA, $5B NI, $1B net debt. Four comparable pharma transactions provided; mean multiples: TV/Revenue 2.25x, TV/EBITDA 11x, P/E 17.5x. Transaction multiples incorporate premiums — average premium in the transaction set is also disclosed in the summary. Students must calculate: (a) implied equity value from TV/Revenue, (b) implied equity value from TV/EBITDA, (c) implied equity value from P/E; then compare each implied price to the current $50 market price.

---

### T124. Transaction-Comps-Modeling-Simple-Transaction-Comps-Solution
TV/Revenue solution: 2.25 × $20B = $45B TV − $1B net debt = $44B equity ÷ 1B shares = $44 implied → Eli Lilly appears overvalued at $50. TV/EBITDA solution: 11 × $6.5B = $71.5B − $1B = $70.5B ÷ 1B = $70.50 → undervalued. P/E solution: 17.5 × $5 EPS = $87.50 → significantly undervalued. Different multiples give completely different valuation conclusions because they embed different implicit assumptions — TV/Revenue assumes same cost structures, TV/EBITDA does not, P/E reflects leverage and growth. The analysis demonstrates why choosing the right multiple for the right context matters enormously, and why presenting multiple valuations side by side in a football field is essential.

---

### T125. Transaction-Comps-Modeling-Spreading-the-Comps
Enterasys was a private company acquired by Extreme in 2013; its financials were found only in Extreme's 8-K filed January 2014 (4 months post-closing). Key complexities: (1) Enterasys reported a $94M "affiliate debt" on its balance sheet — disclosed in footnotes as intercompany debt from its former JV parent that was eliminated upon acquisition; this must be zeroed out of net debt since it was not assumed by Extreme. (2) No GAAP-to-non-GAAP press release available (private company) → industry knowledge applied: exclude SBC ($9.4M) and amortization of intangibles ($8.0M) as is standard in this sector. (3) EBITDA double-count fix: $13.15M total D&A − $8.0M already excluded from EBIT = $5.15M add-back. (4) No forecasts available for private target → only LTM multiples calculated.

---

### T126. Transaction-Comps-Modeling-Synergies-and-Premiums
Transaction multiples are structurally higher than trading multiples because they incorporate the acquisition premium paid to gain control. Historically, premiums of 10–25% account for ~19% of deal volume; 25–50% account for ~34%; 50–75% account for ~42% (as of Q2 2014). Premiums are highly sensitive to the M&A environment — deals from 5-7 years ago may not reflect current premium levels. Strategic acquirers pay higher premiums than financial buyers (private equity) because of their ability to capture revenue and cost synergies. Announced synergies (as a % of EBITDA) should be noted and compared across the transaction comp set; higher synergies justify higher premiums.

---

### T127. Transaction-Comps-Modeling-The-Extreme-Enterasys-Deal-Review
The Extreme-Enterasys deal (September 12, 2013): $180M all-cash total consideration; no per-share terms disclosed since Enterasys was private. Three SEC documents found: (1) announcement 8-K (September 12, 2013 — "Extreme acquires Enterasys for $180M, revenues will approximately double"), (2) completion 8-K (confirming final terms), (3) financial data 8-K (filed January 2014 — Extreme disclosed Enterasys historical financials as required for a material acquisition). FactSet missed the Enterasys EBITDA data in its screen; the 8-K was found only through direct EDGAR searching. Synergies: $35M/year (midpoint of $30–40M per Raymond James Research Report). Template: use override for offer value ($180M), leave diluted shares blank.

---

### T128. Transaction-Comps-Modeling-The-HP-3Com-Deal
HP's acquisition of 3Com: announced November 11, 2009 at $7.90/share all cash; completed April 12, 2010. 3Com is a related-but-not-identical comp to EXTR — in networking equipment but with a broader product portfolio. Premium: 3Com traded at $5.40 one day prior → ~46% premium (also ~47% on a one-week and one-month basis, suggesting no major pre-announcement leakage). Both acquirer (HP) and target (3Com) were public → full filing suite available: announcement 8-K, completion 8-K, 10-K, 10-Q, proxy (with Goldman Sachs fairness opinion, management projections). This is the most disclosure-rich comp in the set and serves as the template for proper deal spreading.

---

### T129. Transaction-Comps-Modeling-The-Thoma-Bravo-Blue-Coat-Systems
Thoma Bravo's acquisition of Blue Coat Systems: announced December 9, 2011 at $25.81/share all cash; deal closed February 15, 2012 at ~$1.3B total. Acquirer is a private equity firm (financial buyer); target is public. Premium: ~50% above 1-day, 1-week, and 1-month pre-announcement prices. This is the most complex dilution calculation in the case study: Blue Coat has four types of dilutive securities — stock options, RSUs (restricted stock units), warrants (attached to convertible debt), and convertible notes. The convertible notes must pass both the in-the-money test and the anti-dilution test; the warrants use the treasury stock method with the offer price as the current price. Students are expected to work through each security independently before reviewing the instructor walkthrough.

---

*End of TC Transcripts Resummarized — 129 transcripts (T1–T84 Trading Comps, T85–T129 Transaction Comps)*
