# BUS 696 — Financial Statement Modeling: Conceptual Resummarization

> 65 transcripts. 4–8 sentence conceptual summaries. All keystroke narration removed.

---

## 1. 30000ft View

The course builds a fully integrated three-statement model for Apple from a blank Excel file, covering the income statement, balance sheet, and cash flow statement. A "general inputs" section at the top holds dynamic row headers and key assumptions that feed the rest of the model. Beyond the three core statements, supporting schedules handle complex items: PPE roll-forward, retained earnings, revolver/commercial paper, interest income and expense, EPS, and scenario analysis. The model is structured so that every financial statement mechanically connects to the others — changes on the income statement ripple into the balance sheet, which drives the cash flow statement.

---

## 2. More Robust Dividends Forecast

The dividend payout ratio is calculated as dividends divided by net income — historically around 23% for Apple — and used to forecast future dividends rather than growing them as a flat dollar amount. Share repurchases are modeled similarly, as a dollar amount tied to historical averages or a payout ratio, and serve as a second redistribution mechanism alongside dividends. Both items flow through the retained earnings roll-forward and must be explicitly forecasted because they are the primary driver of Apple's declining retained earnings balance. Sanity-checking your own dividends forecast against the JP Morgan research report confirms whether the model is in the right range.

---

## 3. Alternative Approach

The classification of non-current asset purchases — specifically whether to put them in operating or investing activities on the cash flow statement — is a judgment call with multiple defensible positions. The key requirement is that the model must balance: every asset and liability change must be reflected somewhere in the cash flow statement, and the balance sheet must still net to zero. When a classification choice is ambiguous, what matters most is the ability to articulate a rational argument for the placement rather than getting the "exact" answer. This transcript reinforces that reasonable analysts can disagree on such items as long as internal consistency is maintained.

---

## 4. Balance Sheet Forecast Concepts

The central principle of balance sheet forecasting is matching each line item to its most logical operating driver on the income statement. Accounts receivable should grow in line with revenue, because more sales implies more money owed by customers at year-end; assuming no change in collection behavior, the ratio stays constant. Inventory should grow in line with cost of goods sold for the same reason — the amount of product on hand is directly tied to how much gets sold and recognized through COGS. This linkage approach converts income statement forecasts into balance sheet forecasts automatically, so once the IS is done, much of the BS follows mechanically.

---

## 5. Balance Sheet Schedules

Simple balance sheet items (like accounts receivable or accounts payable) can be forecasted directly on the face of the balance sheet by applying a growth rate or ratio. Complex items with multiple moving parts — specifically PPE and retained earnings — require separate roll-forward schedules below the cash flow statement because you cannot simply grow them with a single driver. Each schedule follows the same structure: beginning of period + additions − subtractions = ending of period. The schedule outputs then link back into the balance sheet, maintaining model integrity without embedding complex logic directly in the BS cells.

---

## 6. Balancing the Model Exercise

When a three-statement model doesn't balance, the correct approach is systematic: go through every balance sheet line item one at a time, predict what its corresponding cash flow statement entry should be, verify the CFS actually reflects that, and cross it off. This process prevents aimless cell-by-cell searching and forces you to think about the accounting relationship for each item. The most common errors live in the working capital section (sign errors), the retained earnings roll-forward (missing prior-year balance), and equity (missing SBC linkage to common stock). Running this exercise on a deliberately broken model is one of the most important skills the course teaches.

---

## 7. Balancing the Model Solutions

Walking through the broken model reveals three classic error types. First, working capital assets on the CFS have the wrong sign — an increase in accounts receivable should be a subtraction from net income (uses cash), but was entered as positive. Second, the retained earnings roll-forward was missing its prior-year ending balance as the starting point for the current year. Third, stock-based compensation was not added to the common stock balance on the equity section, even though SBC credits common stock in the journal entry. Each error caused the balance sheet to fail to zero — fixing all three restores the balance check.

---

## 8. BS Forecasting CL and LT Debt

Accounts payable should grow with the cost of goods sold growth rate, because AP represents amounts owed to suppliers for product inputs that flow through COGS. Other current liabilities grow with revenue as a catch-all for operating obligations tied to business activity. Deferred revenue is a liability representing cash collected before revenue is earned — for Apple, primarily from iPhone software upgrade rights and gift cards — and is forecasted to grow with revenue. Commercial paper is repurposed in the model as the revolving credit line (model plug), while long-term debt is straight-lined in the base case absent a specific thesis on new borrowings.

---

## 9. BS Forecasting Current Assets

Accounts receivable is forecasted by applying the revenue growth rate to the prior year's AR balance, on the assumption that the DSO ratio stays constant. Inventory is grown by the cost of goods sold growth rate for the same reason — no change in inventory efficiency is assumed unless a separate thesis exists. Other current assets (catch-all) grow with revenue since their nature is assumed to be operationally tied to business activity. Non-current assets also grow with revenue under this simplified approach, with the exception of PPE, which gets its own schedule.

---

## 10. BS Forecasting Equity

Common stock (including additional paid-in capital) increases each year by the amount of stock-based compensation recognized, because the journal entry for SBC credits common stock/APIC. This is the offsetting entry to the SBC expense that runs through the income statement and reduces retained earnings. Other comprehensive income (AOCI) captures items that bypass the income statement — primarily FX translation and unrealized gains/losses on securities — and is unpredictable, so it is straight-lined at the last historical year's value. Forecasting these equity components correctly is essential for the balance sheet to balance.

---

## 11. CAGR

The compound annual growth rate formula is (ending value / beginning value)^(1/n) − 1, where n is the number of years in the period. In Excel, n can be made dynamic by using the COLUMNS() function to count the number of years automatically rather than hardcoding the exponent. For Apple, revenue CAGR over the historical period was approximately 3.2% while EBITDA CAGR was around 2%, implying margin compression over the same period. CAGR is a quick way to contextualize directional trends before diving into year-by-year forecast assumptions.

---

## 12. CFS Forecasting

The cash flow statement is built using the indirect method: start with net income, then make adjustments. For asset changes, the rule is subtract increases (cash outflow) and add back decreases (cash inflow) — the opposite for liabilities. Depreciation and amortization are added back because they are non-cash expenses embedded in net income. Stock-based compensation is also added back as a non-cash item. The CFS is not a new forecast; it is purely a mechanical derivation of year-over-year balance sheet changes, which is why no historical CFS inputs are needed in the model.

---

## 13. CFS Introduction

The cash flow statement is a reconciliation — not a new forecast — of year-over-year changes in the balance sheet. Because all the historical information is already captured in the balance sheet and income statement, there is no need to separately enter historical CFS data in the model. When a three-statement model doesn't balance, the error almost always lives in the CFS linkages rather than the IS or BS. Understanding why this is true requires deeply internalizing the relationship between every BS line item and its corresponding CFS treatment.

---

## 14. Cash From Operations

Conceptual exercises drive home the directional rules for CFS adjustments. When a company builds up inventory (asset increases), cash goes out — subtract from net income. When a company recognizes accrued wages (liability increases), cash hasn't left yet — add to net income. Gift cards sold but not redeemed create deferred revenue (liability up) — add to net income. When deferred revenue reverses as revenue is recognized, the liability decreases — subtract from net income. These exercises build the intuition that the sign of every working capital adjustment on the CFS is driven by whether a balance sheet item is an asset or a liability.

---

## 15. Circularity

Interest income and interest expense create a circular reference in a three-statement model because they depend on average cash and debt balances, which themselves depend on net income, which depends on interest. The simplest fix is to use only the beginning-of-period balance instead of the average for the interest calculation, eliminating the circularity at the cost of slight inaccuracy. The more correct approach is a circuit breaker: an IF statement that checks whether a triggering cell equals zero (indicating a broken circular chain) and returns zero instead of triggering an infinite loop. To enable true iterative calculation instead, go to Alt+T+O → Formulas → enable iterative calculations; both the interest income and interest expense circularities must be addressed together.

---

## 16. Depreciation Waterfall Intro

The simple approach to forecasting depreciation is to model DNA as a percentage of CapEx, with that ratio trending toward 1.0 as the company matures — reflecting the convergence of current investment and prior depreciation. The more sophisticated approach is a depreciation waterfall schedule, which tracks each cohort of capital expenditures separately and applies the appropriate useful life to each vintage. This transcript introduces the waterfall as an upgrade to be built in Part 2 of the course, after the core three-statement model is complete. The simple step-function approach is used first because it is sufficient for establishing a functional model.

---

## 17. EPS

Earnings per share requires forecasting both basic and diluted weighted average shares outstanding. The basic EPS denominator uses weighted average shares: average of beginning-of-period (BOP) and end-of-period (EOP) shares. EOP shares = prior year shares + new SBC shares issued (SBC dollars ÷ stock price) − shares retired through buybacks (buyback dollars ÷ stock price). Diluted shares = basic shares + a straight-lined historical gap between basic and diluted (representing in-the-money options and restricted stock). A constant P/E ratio assumption converts EPS to an implied share price forecast.

---

## 18. Welcome

This course is built by Wall Street Prep and uses Apple as a live case study, building the full financial model from scratch in Excel without a pre-filled template. The scope covers the income statement, balance sheet, cash flow statement, and supporting schedules including PPE, retained earnings, revolver/circularity, error-proofing, and scenario analysis. The course is designed for finance professionals who need hands-on modeling skills, not just conceptual understanding. By the end, students will have a fully functional, integrated model that reflects real-world sell-side analyst practices.

---

## 19. Forecasting BS Intro

Working capital items on the balance sheet do not need to be forecasted from scratch — they naturally piggyback on the income statement forecasts already built. Because accounts receivable grows with revenue, inventory with COGS, and payables with COGS, once the IS is complete the BS largely follows via ratios. This section transitions the model from relying on JP Morgan for everything to building mechanical linkages that derive the balance sheet from the income statement. The goal is to move away from hardcoded assumptions toward a model where changing one driver (e.g., revenue growth) ripples through the entire model correctly.

---

## 20. Forecasting CFI and CFF

Cash from investing (CFI) captures CapEx as a negative (cash out) and any asset sales as positive; in the base model, only CapEx is forecasted. Cash from financing (CFF) captures long-term debt changes (use a formula = current year LT debt − prior year LT debt, not a hardcoded zero) plus buybacks and dividends. The ending cash balance is calculated as prior year cash balance plus the net change in cash — not a direct reference to the net change line — because summing the sections directly is a common source of errors. This transcript highlights that a correct sign convention and formula structure prevents many of the most common model mistakes.

---

## 21. Forecasting EPS (Detailed)

End-of-period shares for each forecast year = prior year EOP + (SBC dollars / stock price) − (buyback dollars / stock price). The weighted average shares for the EPS denominator uses the average of BOP and EOP for the year. Diluted shares = basic weighted average + historical dilution gap (difference between diluted and basic in prior years, straight-lined forward). The stock price assumption is driven by applying a constant historical P/E ratio to the forecasted EPS, creating a circular dependency that must be managed carefully. All share counts are in millions; all dollar amounts in thousands (matching the model scale).

---

## 22. Forecasting Interest Expense

Commercial paper interest expense = weighted average interest rate (from Apple's footnote, ~2.18%) × average commercial paper balance (BOP + EOP) / 2. Long-term debt interest is harder to isolate because Apple doesn't separately report LT debt interest expense — it must be backed into: total interest expense minus the CP component, divided by the average LT debt balance, yields an implied LT rate (~3.2%). That implied rate is then straight-lined forward for the forecast period. Modeling interest expense this way avoids hardcoding a number and keeps the model dynamic as debt balances change.

---

## 23. Forecasting Interest Income

Interest income is modeled as the weighted average interest rate earned on cash × average cash balance. Apple's interest income rate of 2.16% comes from its footnote to the "other income and expense" section of the 10-K. Applying this rate to the average (BOP + EOP) / 2 cash balance yields the interest income forecast; as cash declines in the model due to buybacks and dividends exceeding free cash flow, interest income should decline — which serves as a good sanity check. With interest income and expense now forecasted, net income is fully determined and should be checked against consensus estimates.

---

## 24. Forecasting Income Statement

The income statement forecast uses JP Morgan research as the primary source for the first three years: revenue −4%, +6%, +6.9% (declining then recovering). Gross profit margin is forecasted at 37.8%–38.2% and straight-lined thereafter. R&D and SGA are forecasted as a percentage of revenue, also relying on JP Morgan for the initial years then straight-lined. Interest income and expense are left blank initially because they depend on schedules not yet built; everything else (revenue through pre-tax income and taxes) is complete by the end of this transcript.

---

## 25. Gathering Documents (PIB)

The Public Information Book (PIB) is the analyst's starting research set: 10-K (annual report with financials, MD&A, footnotes), 8-K press releases (quarterly earnings with tables), conference call transcripts, equity research (JP Morgan for this model), and consensus estimates from Capital IQ. SEC EDGAR is the primary source for 10-K and 8-K filings, navigated via the search bar with a 10-K filter and date range. PDF shortcuts: Ctrl+Shift+N jumps to a specific page number (the PDF page, not the document page); Ctrl+F finds text. For private companies, management-provided financials replace public filings.

---

## 26. Inputting Total D&A

Total depreciation and amortization has two components: DNA specifically from PPE (found in the PPE footnote of the 10-K) and non-PPE DNA (primarily intangible asset amortization, estimated at ~0.6% of revenue). The PPE-specific DNA comes from the footnote because the cash flow statement lumps all DNA together without breaking it out by asset type. Total DNA = PPE DNA + non-PPE DNA, and this total feeds both the income statement EBITDA calculation and the cash flow statement add-back. Separating the two components is essential for the PPE roll-forward schedule, which requires only the PPE portion.

---

## 27. Inventory Roll Forward

Two approaches exist for forecasting inventory, toggled with an IF statement based on a user input. Approach 1: keep the average inventory / COGS ratio constant (equivalent to growing with COGS); the EOP balance is backed into as: (ratio × COGS × 2) − BOP. Approach 2: specify a target inventory turnover (COGS / average inventory); EOP = (COGS / turnover × 2) − BOP. In both cases, the formula solves for EOP given that the average of BOP and EOP should equal the target ratio applied to COGS. The toggle allows the modeler to either maintain historical efficiency or explicitly model improving/worsening inventory management.

---

## 28. Locating Relevant Filings Part 1

Apple's 10-K is found on SEC EDGAR by filtering for 10-K filings after September 2018 (Apple's fiscal year ends late September). The 10-K is the primary source for historical financials, footnotes, and MD&A. Apple's fiscal year-end date is typically September 28–30, so document dates and page numbers in the PDF reader do not align — the PDF page number differs from the document page number. Ctrl+Shift+N in the PDF reader jumps to a specific reader page; Ctrl+F searches for text. The 8-K press release provides the same data in a more condensed format and is often used for quarterly results.

---

## 29. Locating Relevant Filings Part 2

JP Morgan's research report provides explicit forecasts for revenue, gross margin, R&D%, SGA%, CapEx, and EPS used to populate the income statement forecast section. Canaccord Genuity's report is used for the price×volume analysis because it provides a product-level unit and ASP breakdown that JP Morgan does not. Capital IQ's consensus data shows ~41 analysts contributing near-term forecasts (2019–2020), dropping to fewer than 10 for years 3+; the consensus $257B revenue forecast for 2019 is close to JP Morgan's $254B. When commingling research from different firms, confirm that their top-line revenue estimates are broadly consistent before relying on one firm's product-level breakdown.

---

## 30. Model Structure

Financial models should use fewer, longer worksheets rather than many short ones, because cross-sheet linking (linking between tabs) is a major source of errors in practice. Within a worksheet, "elevator drops" (Ctrl+↓ through blank column gaps) navigate quickly between sections. Models should be "light disposable" rather than black-box — transparent formulas, consistent color coding (blue = hardcoded inputs, black = formulas), and no hidden rows. Annual models are appropriate for strategic analysis; quarterly models are used when near-term earnings estimates or liquidity timing matter.

---

## 31. Modeling Best Practices

Core rules: (1) Never re-enter the same input twice — always reference the source. (2) Never embed numbers directly inside formulas — all assumptions must live in labeled input cells. (3) Label balances clearly (BOP vs EOP). (4) Calculate subtotals with formulas, never hardcode them. (5) Keep formulas short — one operation per cell is better than nested mega-formulas. (6) No daisy-chain links (A→B→C→D across sheets); keep source references short. (7) Use grouping (Shift+Alt+→) to collapse rows, never hide. (8) Use center-across-selection for headers instead of merge cells — merge cells break Excel functions. These practices are what distinguish professional-grade models from error-prone student work.

---

## 32. Modeling Historical Data

The model starts from a blank Excel template, not a pre-filled file — inputting data yourself builds understanding of the numbers. Date cells use the EOMONTH function to ensure period-end dates (e.g., September 30 = EOM for an Apple fiscal year). A balance checker cell uses ROUND(assets − liabilities − equity, 3) = 0 to verify the balance sheet balances at all times. Blue formatting signals hardcoded historical inputs, distinguishing them from formulas. Using the alt+mouse drag technique in the PDF reader allows selecting individual columns from the 10-K without grabbing adjacent data.

---

## 33. Modeling PPE

The PPE roll-forward follows: BOP net PPE + CapEx − DNA (PPE only) = EOP net PPE. Historical CapEx comes from the cash flow statement investing section; PPE-specific DNA comes from the footnote (not the aggregate CFS DNA line). The DNA/CapEx ratio is trended toward 1.0 by stepping it up 2% per year — reflecting the assumption that as Apple matures, its depreciation will converge to match current spending. Once the PPE EOP balance is calculated, it links back to the balance sheet; CapEx also links to CFI as a negative cash outflow with a sign flip.

---

## 34. Modeling Retained Earnings

Apple's retained earnings are unusual because treasury stock repurchases are embedded within the RE balance rather than broken out as a separate contra-equity account, as most companies do. The roll-forward is: BOP RE + net income − dividends − share repurchases = EOP RE. Because Apple returns more to shareholders (dividends + buybacks) than it earns in net income, RE is forecasted to become increasingly negative over the forecast period — which is mathematically fine and reflects Apple's deliberate capital return strategy. Apple borrows cheaply (~2.5%) to fund some of these distributions while sitting on enormous cash balances, exploiting the arbitrage between its borrowing rate and the return earned on its cash.

---

## 35. Modeling Roadmap Part 1

The core three-statement model is built in six steps: (1) input historical IS and BS data from the 10-K; (2) forecast the income statement using JP Morgan research; (3) forecast the balance sheet by linking items to IS drivers; (4) derive the cash flow statement from year-over-year BS changes; (5) handle the circular reference created by interest income/expense; (6) build scenario and sensitivity analysis around key assumptions. The income statement must be forecasted first because its outputs (revenue, COGS, SGA) drive the balance sheet items. The CFS is derived, not separately forecasted, because it is a pure reconciliation of BS changes.

---

## 36. Modeling Roadmap Part 2

Part 2 adds five enhancements to the core model: (1) error-proofing via a balance checker and a systematic process for identifying why a model doesn't balance; (2) a price×volume revenue build that breaks aggregate revenue into product-level unit × ASP components; (3) working capital schedules giving the user explicit control over DSO, DIO, and DPO assumptions; (4) a depreciation waterfall for more precise PPE DNA modeling; (5) a model update exercise using Apple's actual 2019 10-K results to compare forecast vs. actual. The update exercise is particularly important because updating an existing model is what analysts do most often in practice.

---

## 37. Modeling the Revolver

The revolver is a model plug that automatically handles cash shortfalls by drawing on the credit line or pays down the revolver when excess cash exists. Cash available to pay down the revolver = BOP cash − minimum cash balance (2% of revenue as a convention, or $50B for Apple) + CFO + CFI + non-revolver CFF items. A MIN function prevents the revolver from going negative: MIN(current revolver balance, cash available) controls the paydown amount. Critical mistake to avoid: do not grab the total net change in cash (which already includes the revolver line) when calculating available cash — this creates an unintentional circular reference. A discretionary draw line can override the automatic paydown to model a company's preference for maintaining debt.

---

## 38. Other Assets and Liabilities

"Other" buckets on the balance sheet aggregate multiple smaller items whose individual nature may not be disclosed. The forecasting decision tree: (1) check footnotes for a breakout; (2) if items are operationally tied to the business (tax payables, deferred costs), grow with revenue; (3) if items are financial in nature (FX-linked instruments, investments in affiliates) or if the nature is unclear, straight-line the last historical year's value. The straight-line vs. revenue-growth distinction conveys a philosophical stance: "I think this is tied to operations" vs. "I don't know and won't guess." For Apple, other non-current liabilities ($45B) include a deferred tax component and unidentified items that are straight-lined.

---

## 39. Other Non-Current Assets

Other non-current assets include intangibles and goodwill (embedded in this bucket for Apple post-2015 reporting changes). A roll-forward schedule is built: BOP + additions − non-PPE amortization = EOP. The amortization was already forecasted (non-PPE DNA = ~0.6% of revenue), so the additions line is a solve-for-X: additions = EOP − BOP + amortization. On the cash flow statement, only the additions line should flow through (as a cash outflow), not the full year-over-year change in the balance — because the full change already nets out the amortization that was separately reported as a CFS add-back. Using the full balance change would double-count the amortization and prevent the model from balancing.

---

## 40. PPE Roll Forward Concept

The roll-forward is the standard forecasting framework for any balance sheet line with more than one driver: BOP + additions (CapEx) − subtractions (DNA) = EOP. Historical CapEx is found on the cash flow statement under investing activities; it may be labeled "purchases of property, plant and equipment" rather than "capital expenditures." For growing companies, CapEx exceeds depreciation (ratio >1) because new purchases outpace historical depreciation; for mature or declining businesses, the ratio converges toward 1.0 as companies invest only to replace existing assets. This conceptual framework applies to PPE, retained earnings, and any other complex balance sheet item.

---

## 41. Price × Volume Exercise 1

The price×volume build breaks Apple's $265B historical revenue into five product categories: iPhone, iPad, Mac, Services, and Other Products. For the three hardware categories, the average selling price (ASP) is backed into as revenue ÷ unit count. Historical unit volumes come directly from Apple's 10-K MD&A; ASP is an implied calculation because Apple does not separately disclose it. The exercise builds a table with: historical revenue by product, unit counts, implied ASPs, unit growth rates, ASP growth rates, and revenue growth rates by product — creating the foundation for a bottom-up revenue forecast.

---

## 42. Price × Volume Exercise 1 Solution

Data entry from the PDF uses the alt+mouse drag trick to select individual columns without capturing adjacent content. Revenue figures are in millions and unit counts are in thousands, so the scale adjustment for ASP calculation is: revenue (millions) ÷ units (thousands) × 1,000 = ASP in dollars. Apple's implied iPhone ASP in 2018 was ~$765. Always paste special → values (Alt+E+S+V) to avoid overwriting pre-formatted cells in the template. Subtotals for total revenue use Alt+= (auto-sum) rather than hardcoded values so that changes to individual products flow through automatically.

---

## 43. Price × Volume Exercise 2

JP Morgan's research does not provide a product-level price×volume breakdown for Apple, so Canaccord Genuity's report is used instead — commingling research from two firms is acceptable when total revenue forecasts are broadly consistent. Canaccord provides forecasted unit counts and ASPs for iPhone, iPad, and Mac for the first two years, with historical data matching Apple's actual disclosures (subject to minor reclassifications). The near-term iPhone unit decline (from 217M to 173.7M) is the primary driver of Apple's forecasted revenue decline — directly visible once the build is constructed. The ASP-adjusted-for-deferrals metric in Canaccord's report reflects stripping out prior-period deferred revenue cycling through current-period iPhone revenue recognition.

---

## 44. Price × Volume Exercise 2 Solution

Canaccord's historical iPhone revenue of $164.9B differs slightly from Apple's reported $166.7B due to a reclassification Apple made between the press release and the 10-K filing, moving ~$2.5B from iPhone to Services. The practical fix is to accept the minor discrepancy and use Canaccord's ASP forecast ($792 and $725 for years 1–2) straight-lined thereafter. Services and Other Products revenues are forecasted using Canaccord's explicit estimates for two years, then straight-lined at those implied growth rates. The resulting model shows -4.7% revenue decline in year 1 vs. JP Morgan's -4% — close enough to confirm both represent a consistent "street case."

---

## 45. PV Build Exercise 3

With the price×volume build complete, the challenge is to replace the simple revenue growth scenario assumptions with more granular iPhone unit and iPhone ASP scenarios. Instead of a single "revenue growth rate" input with best/base/weak cases, the scenario table should include separate rows for iPhone units (best/base/weak) and iPhone ASP (best/base/weak). The revenue build then reads those scenario values and calculates the implied revenue, which flows into the income statement's revenue growth rate. The mechanical challenge is wiring the scenario dropdown through the revenue build worksheet and back to the IS without breaking the existing offset-match functions.

---

## 46. PV Build Exercise 3 Solution

The solution adds two new rows to the scenario table (iPhone ASP and iPhone units) using the same offset-match function already in place — the headings must match exactly for the match function to work. The revenue build is updated to pull its iPhone unit and ASP assumptions from the "active scenario" row (not the raw scenario table) so that toggling the scenario dropdown changes the units and ASPs feeding into the revenue calculation. The revenue build then calculates implied growth rates, which feed the IS's revenue line — replacing the old hardcoded JP Morgan growth rate. The data flow is: dropdown selection → scenario table → revenue build → implied revenue growth rate → income statement.

---

## 47. Retained Earnings Roll Forward Concepts

Retained earnings is a complex balance sheet item requiring an explicit roll-forward because it has three moving parts: net income increases it, dividends reduce it, and share repurchases reduce it (for Apple). For most companies, buybacks reduce a separate treasury stock contra-equity account; Apple consolidates everything into retained earnings. If the sum of dividends and repurchases exceeds net income — as is the case for Apple — retained earnings declines and eventually goes negative. This is not fundamentally problematic; it simply reflects aggressive capital return to shareholders funded by prior accumulated earnings and ongoing free cash flow.

---

## 48. Revenue Build

The revenue build is motivated by a need to understand *why* revenue is expected to grow or decline, not just by how much. JP Morgan forecasts a revenue decline, and the product-level analysis reveals this is driven almost entirely by a major iPhone unit volume decline — not by services or other products, which are growing fast. Apple discloses product-level revenue and unit counts in the MD&A section of the 10-K; geographic breakdowns are also available but less analytically useful for this exercise. For private companies, segment-level data is typically provided only when the company engages with an advisor for a specific transaction.

---

## 49. Review the Cash Flow Statement

The cash flow statement reconciles accrual-based net income to actual cash generation, addressing the fundamental limitation of GAAP income accounting: high reported profits can coexist with deteriorating cash, and low reported profits can accompany strong cash generation (e.g., a capital-intensive business in its investment phase). The indirect method — used by virtually all companies — starts with net income and adjusts for: (1) non-cash expenses (add back DNA, SBC, write-downs); (2) working capital changes (asset increases subtract, liability increases add). Three sections: CFO (core operations), CFI (CapEx, acquisitions, asset sales), CFF (debt issuance/repayment, equity issuance, buybacks, dividends). Both the IS and CFS must be analyzed together — using either alone creates blind spots.

---

## 50. Roll Forward Concept

The roll-forward (also called a "base" or "waterfall") is the universal template for forecasting any balance sheet item with multiple drivers: beginning of period (= prior year EOP) + additions − subtractions = end of period. Simple items (AR, inventory) can be forecasted directly on the BS face using a single growth rate. Complex items (PPE, retained earnings, other non-current assets) require a dedicated roll-forward schedule because they have multiple independently-forecasted components. The discipline of explicitly identifying what makes each line go up and what makes it go down prevents the analytical error of treating a complex item as though it moves monolithically.

---

## 51. Sanity Checking Model vs. Consensus

After completing the income statement forecast, EBITDA is compared against JP Morgan's estimate (~$73.3B) and the Capital IQ consensus (~$74.7B for 2019) — the model's ~$72.9B is close, confirming the forecast is in the right ballpark. Consensus data from Capital IQ shows ~41 analysts for the near-term forecast period, dropping to <10 for years 3+, so the far-out consensus is less reliable. At this stage, the model still has three open items: interest income/expense (need schedules), commercial paper (needs revolver calculation), and other non-current assets (needs its own schedule). Completing those items will finalize net income and allow the final consensus comparison.

---

## 52. Scenario Analysis Exercise

Scenario analysis allows the user to toggle the entire model between a best case, base case, and weak case by changing a single dropdown cell. Four key income statement drivers are included in the scenario table: revenue growth, gross profit margin, R&D as % of sales, and SGA as % of sales. The base case uses JP Morgan's estimates; the best/weak cases apply a ±2.5% variance to revenue growth and ±1% to margin assumptions. The Excel challenge is creating a formula that can be copied uniformly across all four assumption rows while pulling the correct case value for each row — this requires a combination of OFFSET and MATCH functions.

---

## 53. Scenario Analysis Solution

The OFFSET-MATCH formula solution: use OFFSET with a fixed anchor cell (reference point above the scenario table), with the number of rows specified by MATCH(selected_case, case_header_range, 0) for the case dimension, and MATCH(assumption_name, assumption_header_range, 0) − 1 for the row dimension. The second MATCH automatically increments by 4 rows for each new assumption category, so the formula is truly universal — one formula that can be copied down and across without modification. After building the scenario output, the final step is relinking the four income statement input cells to point to the scenario output row rather than the original hardcoded assumptions. Stress-test the formula by switching cases and verifying that net income changes directionally as expected.

---

## 54. Sensitivity Analysis with Data Tables

A two-variable data table sensitizes one output (e.g., Year 1 net income) to two inputs simultaneously (e.g., revenue growth and gross profit margin). The output variable is referenced into the top-left corner of the table; the row inputs (revenue growth scenarios) are hard-coded in the first column; the column inputs (margin scenarios) are hard-coded in the first row. The shortcut is Alt+D+T; Excel prompts for which cell in the model each set of inputs should temporarily replace. Data table inputs must be hardcoded — not linked — and the table is set to "automatic except data tables" in workbook calculation settings to avoid grinding Excel to a halt. Press F9 to recalculate after building the table.

---

## 55. Step 1: Inputting Historicals

The exercise loads Apple's 2018 10-K and populates three years of historical data for the income statement and balance sheet (2016–2018). For the income statement, all expenses are entered as negatives; Apple's "other income and expense net" line must be decomposed using the footnote to separate interest income (~$5.6B), interest expense (~$3.2B), and other items. EBITDA requires adding back DNA and SBC, both found on the cash flow statement's operating section. Additionally, the general inputs section needs historical data scavenged from the 10-K: CapEx (from CFS investing), PPE-specific DNA (footnote page 48), dividends and repurchases (statement of shareholders' equity), commercial paper interest rate (debt footnote, 2.18%), and interest income rate (other income footnote, 2.16%).

---

## 56. Step 1 Solution: Balance Sheet

Apple's balance sheet aggregates cash + short-term marketable securities + long-term marketable securities into a single "cash" line (~$237B total in 2018) because all three are highly liquid. Apple has no treasury stock line on its equity section — share repurchases are embedded in retained earnings, as confirmed by the statement of shareholders' equity. AOCI (accumulated other comprehensive income) represents the cumulative impact of items that bypass the income statement: FX translation, unrealized gains/losses on hedges and securities. Deferred revenue ($7–8B) covers iPhone software upgrade rights (amortized over 2 years per unit sold) and gift card balances. The balance check = ROUND(assets − liabilities − equity, 3) should equal zero after data entry.

---

## 57. Step 1 Solution: Historical CF and Other

Capital expenditures historically ran $12.4–13.3B annually and are entered as positive in the general inputs (sign flipped from CFS). PPE-specific DNA of $8.2–9.3B comes from the footnote (not the CFS aggregate DNA line), which also clarifies that Apple uses straight-line depreciation for PPE. Dividends and repurchases are taken from the statement of shareholders' equity (preferred for consistency) rather than the CFS. The interest rate on commercial paper (2.18% in 2018, 1.2% in 2017) comes from the debt footnote; the interest income rate on cash (2.16%) comes from the other income/expense footnote. The observation that Apple borrows at ~2.18% while earning ~2.16% on its cash explains why a cash-rich company would still choose to carry significant debt.

---

## 58. Step 1 Solution: Income Statement

The latest share count (~4.7B shares) comes from the cover page of the 10-K. The alt+mouse PDF drag technique selects income statement columns cleanly; alt+E+S (paste special) → values-only preserves template formatting. All expenses are formatted as negatives using paste-special multiply by −1. The "other income expense net" line is decomposed via the footnote into interest income (+$5.6B), interest expense (−$3.2B), and other expense (−$0.3B). The effective tax rate fell from ~28% in 2017 to ~18% in 2018 due to the Tax Cuts and Jobs Act (35% → 21% statutory rate), which affects the historical tax rate and the forward-looking forecast rate. EBITDA add-backs (DNA + SBC) come from the cash flow statement's operating section.

---

## 59. Step 2: Forecasting the Income Statement

Revenue is grown by the JP Morgan forecast rate; gross profit is calculated as gross profit margin × revenue (making cost of sales the residual plug). R&D and SGA are each calculated as a percentage of revenue using JP Morgan's margin assumptions. Taxes = effective tax rate × pre-tax income (never as a percentage of revenue — a common and material mistake). SBC is grown proportionally with revenue as the best available proxy for future equity compensation. "Other income/expense" items (besides interest) are straight-lined from the most recent historical year since their composition is unknown and erratic. DNA is left blank as a placeholder until the PPE schedule is built. Final sanity check: compare operating income to JP Morgan's estimate (~$61.6B vs. model's ~$61.7B) and Capital IQ consensus (~$62–65B range) to confirm model alignment.

---

## 60. The Revolver as a Model Plug

Every three-statement model needs a revolving credit line that automatically handles cash shortfalls and surpluses — not just for Apple but for any company. For Apple's case, the model forecasts approximately $50B in free cash flow but $86B in buybacks and dividends, meaning the company draws down its $237B cash cushion each year. The revolver plug calculates: how much cash is available (BOP cash − minimum cash balance + all other cash flows except the revolver itself), and if positive, pays down the revolver; if negative, draws on it. The minimum cash assumption ($50B for Apple, or ~2% of revenue for a typical company) prevents the model from depleting all liquidity. This section is the most technically challenging part of the model and the most common source of circular references if the formula grabs the total net change in cash rather than the individual components.

---

## 61. Updating an Existing Model

When Apple's 2019 10-K was released, actual results showed only a −2% revenue decline vs. the −4% JP Morgan had forecasted — a meaningful beat that illustrates how quickly sell-side forecasts can be wrong. The update process: insert a new column for fiscal year 2019, copy the prior-year column's formulas (especially EOMONTH date formulas), override the historical inputs with actual 2019 data from the new 10-K, and update forecast assumptions for years 2020 onward using the latest consensus. The share count and share price must be updated ($293.65 at December 31, 2019 closing price). Model updates on a completed model should take ~25–30 minutes, because all the architecture is already in place.

---

## 62. Working Capital and Liquidity Analysis

Inventory is added to the working capital schedules alongside accounts receivable, using the same two-approach toggle. Approach 1 keeps the average inventory/COGS ratio constant (the default); Approach 2 allows the user to specify an inventory turnover target (COGS ÷ average inventory). In both cases, EOP inventory = (target average inventory × 2) − BOP inventory. Apple turns inventory ~37 times per year, reflecting an extremely efficient supply chain; modeling a further improvement reduces the inventory balance in the forecast. Working capital efficiency metrics (DSO, DIO, DPO) can be tracked in a separate ratio section to monitor model assumptions against industry benchmarks.

---

## 63. Working Capital Concept Checker Exercise

A conceptual exercise presents the most common working capital items and asks students to identify each item's income statement driver before seeing the solution. Deferred revenue is explained via two Apple examples: gift cards (cash received, no revenue recognized until redeemed) and iPhone software upgrade rights (a portion of the purchase price is deferred as a liability and amortized over ~2 years as the software service is delivered). Accrued expenses represent wages or bonuses earned by employees but not yet paid — the expense hits the IS when earned, not when cash leaves. Prepaid expenses are the reverse: cash paid upfront for future services that hit the IS as the service is consumed. Taxes payable accumulates tax liabilities recognized but not yet remitted to the government.

---

## 64. Working Capital Concept Checker Solution

The framework for forecasting working capital items is: tie each item to its closest income statement driver. AR → revenue growth. Inventory → COGS growth (because inventory cycles through the IS via COGS). Prepaid expenses → SGA growth (most prepaid items ultimately hit SGA) or revenue if SGA-specific growth is unavailable. Accounts payable → COGS growth (since AP primarily reflects supplier obligations for product inputs) or revenue. Deferred revenue → revenue growth (as more products are sold, more software upgrade rights are deferred). Accrued expenses → SGA or revenue. Taxes payable → tax expense growth or revenue as lowest common denominator. When in doubt, revenue growth is the acceptable fallback for any WC item.

---

## 65. Working Capital Schedules

After completing the core three-statement model, the working capital section is enhanced to give the user explicit control over DSO (days sales outstanding for AR) and inventory turnover. Previously, AR was simply grown at the revenue growth rate and inventory at the COGS growth rate — which is adequate for a basic model but doesn't let analysts model changing collection efficiency or supply chain improvements. The new schedules provide two approaches for each item (default growth-rate linkage or explicit turnover target), toggled by a user input. This enhancement is classified as "nice to have" for scenarios where the analyst has a specific thesis about working capital management changes, rather than a standard requirement for every model.

---
