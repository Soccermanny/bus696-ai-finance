# DCF Transcripts — Conceptual Summaries
*Chapman FSM — BUS 696 DCF Course | All 63 Videos | Generated May 12, 2026*

---

## 1. Adding Midyear Toggle

The transcript walks through adding a binary toggle (0 or 1) that switches the DCF between end-of-period and midyear discounting conventions. The toggle is implemented as an Excel IF statement so that both approaches co-exist in the model without hardcoding either. A dropdown list is added to the inputs section to make the toggle user-friendly. The key insight is that midyear discounting assumes cash flows arrive at the midpoint of each period rather than the end, which raises the present value of stage-one cash flows. The toggle is a best-practice feature for professional models that need to present both conventions to clients or reviewers.

---

## 2. Model Review So Far

This video recaps the full unlevered free cash flow (UFCF) buildup constructed to that point in the course, tracing the path from EBIT → NOPAT (after-tax operating profit) → add back D&A → adjust for changes in net working capital → subtract CapEx. Using Apple as the running example, stage-one forecasted UFCFs range from roughly $61B to $75B over the five-year explicit period. Terminal value is previewed under both the perpetuity growth and exit multiple methods. WACC is temporarily hardcoded at 9% to allow the model to run before the full WACC build-out, yielding an implied enterprise value of approximately $1.1 trillion — a number that serves as the first sanity check against Apple's observable market value.

---

## 3. Book Value vs. Market Value

The transcript clarifies a conceptual distinction that trips up students: book value of equity (what accountants record on the balance sheet) is almost never equal to market value of equity (what the stock market says the company is worth). Using Google as the example, book equity was roughly $75B while the market cap was approximately $290B — nearly a 4× difference driven entirely by intangible value and market expectations. Enterprise value is also reframed: it equals the value of operating assets net of operating liabilities, and it deliberately excludes cash and financial investments because those are non-operating items. Getting this distinction right is foundational before building the net debt bridge from enterprise value to equity value.

---

## 4. Calculating LTM Data

LTM (Last Twelve Months) data normalizes financial results to a rolling twelve-month window, which matters when a company's fiscal year-end does not align with the valuation date. The formula combines the fraction of the current fiscal year that has elapsed with the complementary fraction of the prior fiscal year: LTM = (elapsed fraction × current year) + (remaining fraction × prior year). The transcript uses Apple as an example and then applies a sanity check: forward-year multiples based on projected earnings should always be lower than LTM multiples, since the market pays today's price for tomorrow's growth. If the forward multiple is higher, something is wrong with the forecast or the multiple calculation.

---

## 5. Calculating Implied Growth Rate

Rather than simply accepting the perpetuity growth rate as an input, analysts can back-solve for the growth rate implied by a given terminal value — a powerful sanity check. The algebra rearranges the perpetuity formula (TV = FCF × (1+g) / (WACC − g)) to solve for g: g = (WACC − FCF/TV) / (1 + FCF/TV). For the exit multiple approach, the implied growth rate embedded in the chosen EV/EBITDA multiple can be cross-checked by computing what long-run growth assumption would reproduce that multiple given the company's forecasted WACC and cash flow profile. If the implied growth rate is implausibly high (e.g., above nominal GDP growth), it signals either an aggressive DCF assumption or an overpriced peer set.

---

## 6. Calculating Valuation Multiples

With a completed DCF, the model's implied valuation multiples can be compared to observable market multiples as a final sanity check. Using Apple, the DCF-derived per-share value of roughly $270 closely matched the then-current market price, validating the model's assumptions. EV/Revenue and EV/EBITDA multiples are computed off year-one forecasted figures and compared to peer trading comps. The transcript emphasizes that this reverse-engineering exercise is one of the most powerful uses of a DCF: rather than using it only to produce a standalone value, analysts use the model to reveal what growth, margin, and reinvestment assumptions the current market price is implicitly endorsing.

---

## 7. Cash Flow vs. Relative Valuation

This conceptual video contrasts the two main approaches to equity valuation. DCF is intrinsic or cash-flow-based: value equals the present value of what the business will generate in perpetuity, independent of what the market happens to be paying today. Comparable company analysis (comps) is relative or market-based: value is inferred from what peers trade for today, expressed as a multiple of some financial metric. The transcript explains that for truly identical companies with identical growth, margins, and risk, both methods should theoretically produce the same answer. When they diverge, the divergence is informative — either the DCF assumptions are mispricing the company or the peer set is not truly comparable.

---

## 8. Connecting DCF to 3-Statement Model

This video demonstrates how to wire the standalone DCF into an integrated three-statement financial statement model (FSM). The key linkages are: EBIT pulled from the income statement, D&A from the income statement or cash flow statement, changes in net working capital from the balance sheet, and CapEx from the investing section of the cash flow statement. A critical modeling note: unlevered taxes in the DCF are computed as tax rate × EBIT — not the actual taxes paid — so that the debt tax shield is excluded from free cash flow and captured exclusively in WACC. Scenario switching is handled through the FSM's case selector, automatically flowing different revenue and margin assumptions into the DCF without manual overrides.

---

## 9. Convertible Debt — Tesla Example

Using a real Tesla filing as the case study, the transcript teaches how to locate convertible debt information. The principal amount is disclosed in footnotes to the financial statements, not on the face of the balance sheet where it may be netted or presented at carrying value. The conversion ratio specifies how many common shares are issued per $1,000 face value "lot" of notes. The in-the-money test compares the conversion value (shares × current price) to the principal; if conversion value exceeds principal, rational investors will convert. This if-converted logic feeds both the diluted share count (add shares) and the net debt calculation (remove the convertible from debt).

---

## 10. Convertible Preferred — Terminology

The transcript introduces the vocabulary needed to analyze convertible preferred stock, which behaves similarly to convertible debt but has a few distinct terms. Redemption value (also called liquidation value) is the par or stated value per preferred share — the amount the company would pay to redeem it. The conversion ratio specifies how many common shares each preferred share converts into. The conversion price is derived as: redemption value per preferred share divided by the conversion ratio. Knowing the conversion price allows an analyst to perform the same in-the-money test used for convertible debt, determining whether preferred shareholders have an economic incentive to convert.

---

## 11. Convertible Securities — Share Impact

This video consolidates the mechanics of how convertible securities affect the diluted share count. Under the if-converted method, whenever the current stock price exceeds the conversion price, the analyst assumes full conversion: the convertible security is eliminated from the net debt calculation, and the shares that would be issued upon conversion are added to the diluted share count. This is applied consistently to convertible bonds and convertible preferred stock alike. The transcript walks through the sign conventions carefully, emphasizing that removing a convertible from net debt raises equity value (fewer liabilities), while adding the conversion shares lowers the per-share value — the two effects partially offset each other.

---

## 12. Date and General Model Inputs

The model inputs section handles the housekeeping variables that control timing and mechanics throughout the DCF. Fiscal year-end dates are entered and the EOMONTH Excel function is used to cleanly derive the last day of each fiscal period. The stub year fraction — representing the portion of the current fiscal year remaining at the valuation date — is calculated as (fiscal year-end − valuation date) / 365 and used to scale the first year's forecasted cash flow. The transcript debates hardcoding the valuation date versus using the NOW() function; hardcoding is strongly preferred for a "locked" analysis so results do not shift when a model is reopened. WACC is temporarily hardcoded at 9% as a placeholder until the full WACC section is built.

---

## 13. DCF Implementation

This video provides a higher-level implementation overview of the full two-stage DCF model structure. The formula mechanics are reviewed: stage-one cash flows are discounted individually, and the terminal value is discounted as a lump sum from the end of the explicit forecast period back to the valuation date. The importance of always presenting DCF output as a range — not a single point estimate — is emphasized because WACC and terminal growth rate are inherently uncertain. The three most important sensitivity levers are identified as: (1) the discount rate (WACC), (2) revenue growth and operating margins, and (3) the terminal value assumption. Any credible DCF presentation should show how the valuation moves across reasonable ranges for each.

---

## 14. DCF Model Sections Preview

A template walkthrough introduces the standard architecture of a professional DCF model. The major sections in order are: (1) model inputs and assumptions, (2) unlevered free cash flow forecast, (3) discounting of stage-one cash flows, (4) terminal value calculation under both methods, (5) enterprise value aggregation, (6) net debt bridge to equity value, (7) per-share value calculation using diluted share count, and (8) sensitivity/scenario output tables. This section-by-section preview serves as a roadmap for the entire course and helps students understand how each individual skill connects to the final output before they begin building.

---

## 15. Integrated vs. Standalone DCF

The course distinguishes between two flavors of DCF model. A standalone or "one-pager" DCF embeds its own simplified income statement and cash flow assumptions directly in the model without linking to a separate three-statement FSM; it is typical in pitch books, sell-side research, and situations where full historical financials are unavailable (e.g., private companies). An integrated DCF links directly to a three-statement model and pulls all drivers — EBIT, D&A, NWC, CapEx — from living model rows, enabling full scenario analysis and ensuring internal consistency. The transcript recommends building the one-pager first to master core DCF mechanics before adding the complexity of FSM integration.

---

## 16. DCF Overview

This foundational video introduces the core concept: the value of any asset equals the present value of all the cash flows it will generate over its life, discounted at a rate that reflects the risk of those cash flows. A hot dog stand thought experiment anchors the theory: $10,500 in year-one cash flow growing at 5% per year, discounted at 10%, yields roughly $43K in stage-one present value plus a $210K terminal value — a total intrinsic value of approximately $253K. The two-stage framework is introduced: an explicit forecast period (typically 5–10 years) where cash flows are modeled in detail, followed by a terminal value that captures the infinite tail of cash flows beyond the forecast horizon.

---

## 17. DCF vs. Comps — Part 1

Using the hot dog stand as a tangible example, the transcript shows a comparable company analysis alongside the DCF. Three comparable hot dog stands with observable transaction prices average a 26× cash flow multiple, implying a value of roughly $273K for our stand. The DCF, by contrast, produces $323K — about 18% higher. The transcript unpacks why these values differ: either the comparable businesses are genuinely different (lower growth, higher risk), the DCF contains overly aggressive assumptions, or the market has mispriced the comparable transactions. This exercise teaches students to treat the convergence or divergence of DCF and comps as a diagnostic signal, not just noise.

---

## 18. DCF vs. Comps — Part 2

This video catalogs the formal advantages and disadvantages of the DCF versus comparable company analysis. DCF advantages include: it is academically grounded in first principles, it produces an intrinsic value independent of market sentiment, and it can value individual business segments or divisions where no direct comps exist. Disadvantages include: there is no universal consensus on implementation details (WACC inputs, terminal value method, stub year treatment), the output is extremely sensitive to small changes in discount rate and growth rate, and the model requires detailed financial forecasts that may not be available for early-stage or opaque companies. Both approaches are valid and should be used together in practice.

---

## 19. Discounting UFCFs

The transcript covers the Excel mechanics of discounting stage-one cash flows using actual calendar dates rather than integer year numbers. The discount exponent for each period is computed as (period cash flow date − valuation date) / 365, yielding a fractional year that handles stub periods and leap years correctly. The valuation date cell reference is anchored with dollar signs so it does not shift when formulas are copied across columns. Each cash flow is then discounted as: UFCF / (1 + WACC)^exponent. The approach is more precise than assuming whole-number periods and is the standard method in professional investment banking models.

---

## 20. Enterprise Value vs. Equity Value

This video is a careful conceptual treatment of what enterprise value actually represents. EV equals the value of a company's operating assets net of its operating liabilities — it deliberately excludes cash, short-term investments, and other non-operating assets because those are separate claims. Equity value equals EV minus net debt (where net debt = gross financial debt − cash and cash equivalents). The hot dog stand exercise puts numbers to this: EV = $900K, net debt = $50K (debt of $500K minus cash of $450K), so equity value = $450K. Students often confuse EV with total firm value; the transcript hammers that EV is specifically the value accruing to all capital providers (debt + equity) for the operating business only.

---

## 21. Estimating Unlevered Taxes

NOPAT (Net Operating Profit After Tax) is the after-tax earnings available to all capital providers before financing effects. It is computed as EBIT × (1 − effective tax rate), deliberately ignoring the actual taxes the company pays in its GAAP income statement. The effective tax rate is sourced from the company's disclosures — either stated explicitly in the tax footnote or calculated as tax provision / pre-tax income from the income statement. The critical reason for using EBIT-based taxes rather than actual taxes is to strip out the interest tax shield from the free cash flow calculation; that shield is captured separately in WACC by tax-adjusting the after-tax cost of debt. Mixing the shield into both places double-counts it.

---

## 22. Exit Multiple Concepts

The exit multiple method calculates terminal value as: terminal-year EBITDA × an exit multiple derived from comparable company trading multiples or precedent transaction multiples. The appeal is simplicity and intuitive alignment with how private equity buyers think about terminal value — because PE funds typically exit within a defined investment horizon, using a market-based exit multiple mirrors their actual hold-period economics. The major weakness identified in the transcript: since terminal value often represents 70–80% of total enterprise value, anchoring it to a market-derived multiple causes the majority of a supposedly "intrinsic" DCF to actually depend on market pricing — the very thing the DCF is meant to be independent of.

---

## 23. Finding Beta on Bloomberg

The transcript is a hands-on Bloomberg tutorial for locating a company's historical beta. Navigating to Apple (AAPL US Equity → DES screen) displays the Bloomberg-calculated beta. The video discusses the choice of lookback period and data frequency: a 2-year weekly calculation is the Bloomberg default and produces a raw beta of 1.10 for Apple, while the adjusted (Vasicek) beta pulls the raw estimate toward 1.0, yielding 1.17. The transcript explains why Bloomberg's adjusted beta is preferred in practice — raw betas have high estimation error, and a mean-reversion adjustment reduces noise. The choice of lookback period meaningfully affects the result, so analysts often check multiple windows for robustness.

---

## 24. Finding Cost of Debt on Bloomberg

Using Bloomberg's SRCH (bond search) function, the transcript demonstrates how to screen for a specific company's publicly traded bonds. For Apple, the filter criteria are: issuer = AAPL, seniority = senior unsecured, face value ≥ $250M, maturity = 8–11 years, currency = USD. A 2.2% coupon bond is identified and opened in ALLQ (all quotes) to find the bid-side spread of 102 basis points over the benchmark rate. Navigating to the YA (yield analysis) screen provides the actual yield to maturity of 1.75%, which becomes the pre-tax cost of debt. This YTM-based cost of debt is superior to simply using the coupon rate because it reflects current market pricing of the company's credit risk.

---

## 25. Forecasting UFCFs

Rather than hardcoding absolute dollar values for revenue and EBITDA, professional DCF models back into growth rates and margins implied by consensus estimates — this keeps the forecast transparent and easy to stress-test. The transcript uses Capital IQ consensus revenue and EBITDA estimates for Apple and calculates the implied year-over-year growth rates (approximately 8%, 9%, 6%, 6%, 6%) and EBITDA margin assumptions. These are then cross-checked against sell-side research reports to validate plausibility. Anchoring to consensus estimates also makes the model defensible in a client or interview setting; analysts can explain the basis for each assumption rather than having to justify self-derived numbers.

---

## 26. Forecasting Working Capital

Net working capital (NWC) is projected as a percentage of revenue based on the historical relationship rather than forecasting individual balance sheet line items independently. For Apple, the historical NWC/revenue ratio is approximately −10.5% (negative because Apple collects cash from customers before paying suppliers — a float business). The transcript flags a common modeling error: analysts sometimes forecast changes in NWC by applying a growth rate directly to prior-year NWC changes, which incorrectly compounds small fluctuations. The correct approach is to project the NWC balance as a percentage of revenue each year, then compute the annual change as (prior-year NWC balance − current-year NWC balance), respecting the sign convention that an increase in an asset is a cash outflow.

---

## 27. From EV to Equity Value

The net debt bridge converts enterprise value to equity value per share. The standard formula is: equity value = EV + cash and cash equivalents − gross debt (+ any other adjustments). Apple is an unusual case because it holds approximately $205B in cash and marketable securities against only $108B in gross debt, producing a negative net debt position — meaning cash exceeds debt. The transcript notes that Capital IQ's reported net debt figure misses Apple's non-current marketable securities (roughly $99B) because those line items are not flagged as current, requiring a manual adjustment. After this correction, the DCF-derived equity value of approximately $1.2 trillion aligns closely with Apple's market cap at the time of the recording.

---

## 28. From Historical NOPAT to UFCF

The transcript walks through computing historical UFCFs from financial statement data as the empirical baseline for the forecast. EBIT (= operating income) is pulled directly from the income statement. D&A is sourced from the cash flow statement's operating section, where it appears as a non-cash add-back to net income. Capital expenditures are found in the investing activities section of the cash flow statement. Changes in net working capital are computed from the balance sheet by identifying the relevant operating current asset and current liability line items and computing year-over-year changes with careful attention to sign conventions. Historical UFCFs provide the pattern of margins and reinvestment intensity that anchors the forward-looking assumptions.

---

## 29. If-Converted Method

This video works through a numerical example of the if-converted method for a hypothetical convertible bond. A $10M face value convertible is issued in $1,000 lots; each lot converts into 4 common shares, yielding a conversion price of $250 per share ($1,000 / 4). If the current share price is $300, conversion is in the money, so the analyst assumes full conversion: 10,000 lots × 4 shares = 40,000 new common shares are added to the diluted count, and the $10M principal is removed from net debt (since bondholders would receive equity rather than cash repayment). If the current price were below $250, conversion would be out of the money and neither adjustment would be made. The same logic applies to convertible preferred stock using its own conversion ratio.

---

## 30. Industry Beta — Concepts

A single company's historical beta is a noisy estimate due to a relatively small number of observations and the fact that beta changes over time as capital structure shifts. The industry beta methodology solves this by using the beta of the entire peer group. The process has three steps: (1) collect the levered (observed) beta for each comparable company; (2) de-lever each beta to remove the effect of that peer's specific capital structure, producing an asset beta or unlevered beta; (3) average the unlevered betas across the peer group; and (4) re-lever the average using the target company's own capital structure (or intended capital structure if modeling a transaction). This produces a more stable, representative measure of operating risk stripped of financing noise.

---

## 31. Industry Beta — Exercise

The transcript applies the industry beta methodology to a real peer group of drugstore chains: Walgreens, CVS, and Rite Aid. Each company's observed levered beta is de-levered using the standard Hamada formula, adjusting for each firm's unique debt/equity ratio and tax rate. Rite Aid's highly leveraged balance sheet creates the largest difference between its levered and unlevered beta — a concrete illustration of why de-levering is necessary. The average unlevered beta across the three peers is approximately 1.0. This is then re-levered at the target private drugstore's capital structure (assumed to be more leveraged than publicly traded peers), producing a re-levered beta of approximately 1.26 to use in the CAPM cost of equity calculation.

---

## 32. Inputting Historical UFCFs

This practical video covers the mechanics of pulling historical financial data from public filings and entering it into the DCF model's historical UFCF section. EBIT is sourced from the income statement (labeled as "operating income" in most filings). D&A is taken from the operating activities section of the cash flow statement, where it appears as a reconciling add-back. CapEx is found in the investing activities section, typically labeled "purchases of property, plant, and equipment." Three to five years of historical data are populated to establish the baseline trajectory of margins and reinvestment intensity. This historical context disciplines the forward-looking assumptions and provides a visible sanity check if forecasted ratios deviate sharply from trend.

---

## 33. Introduction

The opening video frames the entire DCF course using a simple hot dog stand thought experiment. An owner acquires a business with $950K in total assets financed by $500K of debt and $450K of equity, introducing the fundamental accounting identity (assets = liabilities + equity) and the concept of book value. The instructor explains that the course will progressively build every component of a professional DCF model, from conceptual foundations through Bloomberg data gathering to full Excel implementation. Students are introduced to the idea that book value of a business (what accountants say it is worth) bears little relationship to intrinsic value (what the business is actually worth based on its future cash flows) — a tension that motivates the entire course.

---

## 34. Midyear Adjustment — Impact on Terminal Value

The midyear convention does not apply only to stage-one cash flows — it also affects how the terminal value is discounted. When the midyear toggle is enabled, the terminal value is automatically discounted from the midpoint of the final explicit forecast year rather than from its end-of-year date, because the terminal value conceptually represents cash flows beginning in the middle of that last forecasted period (consistent with the midyear timing applied to all stage-one flows). The transcript shows that this adjustment is handled automatically in the Excel model once the midyear exponent formula is set up correctly. The implied terminal growth rate sanity check — back-solving for g from the computed terminal value — continues to work under either convention.

---

## 35. Model Output — Sensitivity Tables

Sensitivity tables are the primary output mechanism for communicating DCF valuation ranges. The transcript covers Excel's built-in two-variable data table feature: one input is placed in the row header (e.g., WACC), a second in the column header (e.g., long-term growth rate for perpetuity approach or exit multiple), and the output cell references the equity value per share or enterprise value. A critical Excel rule: the input cells referenced in the data table must be hardcoded numbers, not formulas linked back to the main model, or the table will not populate correctly. F9 forces a manual recalculate if automatic recalculation is disabled. For client presentations, the instructor recommends narrowing the input ranges to a defensible band centered on the base-case assumptions rather than wide theoretical extremes.

---

## 36. Model Presentation — Football Field Chart

The football field chart is the standard visual for presenting valuation ranges in investment banking deliverables. It is built in Excel as a stacked bar chart where the lower "base" series has no fill and no border, making it invisible and effectively floating the visible valuation range bar at the correct vertical position. The chart typically shows DCF ranges under both the perpetuity growth and exit multiple approaches, alongside the 52-week trading range and (for M&A advisory) a precedent transaction range based on EV/EBITDA purchase multiples. For sell-side advisory engagements, a separate table of implied EV/EBITDA acquisition multiples at various offer prices is also included to help the board assess fairness.

---

## 37. Modeling Basic Shares and RSUs

The diluted share count starts with the basic weighted-average share count disclosed on the front cover of the most recent 10-Q, which reflects shares actually outstanding. Unvested restricted stock units (RSUs) are then added because, unlike stock options, RSUs require no cash exercise — they will vest and become outstanding shares simply through continued employment. For Apple, approximately 4.3B basic shares plus 96M unvested RSUs produces the diluted share count used in the per-share value calculation. The transcript flags a common data source error: Capital IQ's reported diluted share count often excludes unvested RSUs, causing an analyst who uses it uncritically to slightly overstate per-share value.

---

## 38. Modeling Convertible Debt

Using a hypothetical Apple convertible debt position of $14.5B, the transcript models two conversion ratio scenarios. With 3 shares per $1,000 lot, the implied conversion price is $333 — above the current share price of $273, so conversion is out of the money and no dilution occurs. With 4 shares per $1,000 lot, the conversion price falls to $250, which is below $273, so conversion is in the money: approximately 58 million additional dilutive shares are added to the diluted count, and the $14.5B convertible is removed from the net debt calculation. The transcript reinforces the mechanics: in-the-money → assume conversion → add shares, remove from debt. Out-of-the-money → no adjustment to either shares or debt.

---

## 39. Modeling Convertible Preferred Stock

The treatment of convertible preferred stock is structurally identical to convertible debt, with terminology substitutions. A hypothetical $34.6B preferred position issued to 45.5M preferred shares is analyzed. Each preferred share converts into 4.3 common shares (the conversion ratio), implying a conversion price of approximately $177 per common share ($34.6B / 45.5M / 4.3). Since the current stock price of $273 exceeds the $177 conversion price, conversion is in the money: 195M dilutive common shares (45.5M × 4.3) are added to the diluted count, and the $34.6B preferred is removed from the net debt bridge. The parallel to convertible debt is made explicit so students apply the same mental model to both instruments.

---

## 40. Modeling Net Debt

The transcript demonstrates sourcing the components of net debt from Apple's 10-Q balance sheet. Gross debt consists of $10B commercial paper plus $93B of long-term debt, totaling approximately $108B. The filing is then searched for "convertible" and "non-controlling interest" to identify any additional financial claims — none found in Apple's actual capital structure. Non-operating assets that reduce net debt include $39B of cash and equivalents, $67B of short-term marketable securities, and $99B of non-current marketable securities — a total of approximately $205B. The non-current marketable securities line item is one that Capital IQ frequently omits, and the transcript stresses the importance of going directly to the filing rather than relying on data providers for this item.

---

## 41. Modeling Stock Options

The treasury stock method is used to compute the net dilutive effect of outstanding stock options. A hypothetical set of Apple options totaling 205M shares is broken into tranches by exercise price. The in-the-money test is applied tranche by tranche: exercise price < current share price of $273. For in-the-money tranches only, the proceeds the company would receive upon exercise (exercise price × number of options) are used to hypothetically repurchase shares at the current market price (the "treasury stock" buyback). The net dilution equals gross in-the-money options minus shares repurchased: approximately 132M gross in-the-money options → $22B proceeds → buyback of 81M shares at $273 → net dilution of approximately 50M shares.

---

## 42. Modeling the Midyear Convention

End-of-year discounting assumes all cash flow occurs on the last day of each period, which systematically overdiscounts cash flows that in reality arrive throughout the year. The midyear convention corrects this by treating cash flows as if they arrive at the midpoint of each period. In Excel, the cash flow date for year N is set as the average of the period start date and end date (or equivalently, the midpoint of the fiscal year). The discount exponent for each period then uses this midpoint date rather than the year-end date. The result is a higher present value for stage-one cash flows compared to the year-end convention, reflecting that some of the year's cash arrives earlier in the year. Most professional DCF models default to the midyear convention.

---

## 43. Modeling WACC — Capital Weights

WACC is computed using market-value weights rather than book-value weights, because book values of equity can be wildly different from market values. For Apple, equity weight = market cap / (market cap + market value of debt). Because Apple holds more cash than debt (negative net debt), the equity weight mathematically exceeds 100% and the debt weight becomes slightly negative, which is numerically correct. The transcript explains the economic intuition: when a company has excess cash, its observed beta understates the true riskiness of its operating cash flows because cash dilutes the beta. The negative debt weight corrects for this effect in the WACC. An optional override allows analysts to use target or normalized capital structure weights instead of the current market-value weights.

---

## 44. Modeling WACC — Cost of Equity and Debt

The complete WACC inputs are assembled for Apple. From Bloomberg, the pre-tax cost of debt is the YTM of 1.75%; after-tax cost of debt = 1.75% × (1 − 26% tax rate) ≈ 1.3%. The risk-free rate is the 10-year US Treasury yield of 1.23%. Beta is Bloomberg's adjusted (Vasicek) beta of 1.09. The equity risk premium from Duff & Phelps is 5.5%. Plugging into CAPM: cost of equity = 1.23% + 1.09 × 5.5% = 7.3%. Combining with Apple's approximately 100% equity weight produces a WACC of approximately 7.3% — slightly below the 9% placeholder used earlier and consistent with Apple's low leverage and relatively modest systematic risk. The model is updated and the revised equity value recalculated.

---

## 45. Negative Net Debt

This video provides the conceptual and mathematical justification for negative WACC weights when a company has more cash than debt. The equity weight = equity / (equity + debt − cash); if cash > debt, the denominator is equity minus a positive number, so the equity weight exceeds 1. The debt weight is (debt − cash) / total capital, which is negative when net debt is negative. Counterintuitively, this is not a modeling error — it is economically correct. A company flush with cash that earns a low return on that cash has "diluted" betas, meaning its observed stock beta understates the riskiness of its underlying operations. The negative debt weight adjusts the WACC upward to reflect this, producing a higher discount rate than a naive positive-weights calculation would suggest.

---

## 46. Normalizing Terminal Year Free Cash Flows

The terminal value perpetuity formula assumes the last forecasted year's cash flow is representative of a sustainable, steady-state level of earnings and reinvestment. For a high-growth company like Apple, this may not be true at the end of a five-year explicit period: CapEx may still significantly exceed D&A, one-time working capital swings may distort the NWC change, and margins may still be transitioning. The transcript outlines normalization adjustments: gradually converge the CapEx/D&A ratio toward 1.0 over the forecast horizon so that maintenance reinvestment equals the depreciation approximation for steady state, eliminate unusual or one-time NWC changes in the terminal year, and ensure that the terminal-year margin reflects a long-run sustainable level rather than a peak or trough.

---

## 47. Shares Outstanding — Overview

The diluted share count represents all claims on the company's equity value, not just currently outstanding shares. Using a pizza pie analogy, each type of security represents a slice: basic shares are the starting point, and potentially dilutive instruments — stock options, warrants, unvested RSUs, convertible bonds, and convertible preferred — add more slices. The worked example uses $500M in equity value with 100M basic shares plus 25M dilutive options plus 75M dilutive convertible preferred shares = 200M fully diluted shares → $2.50 per share, not the $5.00 one would compute on basic shares alone. This fully diluted share count is what should always be used in the denominator of the per-share value calculation in a DCF.

---

## 48. Stock Options in Share Count

When computing diluted shares from stock options, two key decisions must be made. First, use outstanding options (all options granted, whether or not they can currently be exercised) rather than exercisable options; outstanding is the more conservative and complete view. Second, only include in-the-money options in the dilution calculation — options with an exercise price above the current share price have no economic incentive for exercise and are excluded. Company filings present option data in two formats: a detailed tranche-by-tranche table (exercise price and quantity per tranche) or a single-line aggregate disclosure. The treasury stock method is then applied to each in-the-money tranche: gross dilution offset by a hypothetical buyback funded by the option exercise proceeds.

---

## 49. Stock Splits and Dual-Class Shares

Two corporate action adjustments are needed before finalizing the diluted share count. For stock splits: all historical share counts in SEC filings filed before the split date reflect pre-split numbers, so every pre-split figure must be multiplied by the split factor. Analysts check Bloomberg's CACS function or recent news to identify any corporate actions between the most recent filing date and the current valuation date. For dual-class share structures (e.g., Google with Class A and Class B shares): both classes represent an economic claim on the same underlying equity and should be added together for purposes of the diluted share count, even though they carry different voting rights. Ignoring Class B shares would understate the share count and overstate per-share value.

---

## 50. Stub Year Fraction

The stub year is the partial first year of the explicit forecast period — the portion of the current fiscal year that remains at the valuation date. If a company's fiscal year ends December 31 and the valuation date is September 30, the stub fraction is approximately 0.25 (3 months remaining). The first year's UFCF in the model is multiplied by this stub fraction to avoid over-counting cash flow for the months that have already passed. This stub fraction adjustment is separate from the discount exponent adjustment made for midyear or end-of-period timing: the stub fraction scales the magnitude of the first-period cash flow, while the discounting convention determines the timing at which that scaled cash flow is discounted.

---

## 51. Terminal Value — Exit Multiple Approach

Under the exit multiple method, terminal value is calculated as: terminal-year EBITDA × an exit multiple selected from the trading comps or transaction comps for the company's peer group. For Apple, comparable public companies include Microsoft, Samsung, HP, Dell, Google, and Amazon; their EV/EBITDA trading multiples average approximately 15.4×. Multiplying Apple's forecasted terminal-year EBITDA by this multiple produces a terminal value that is then discounted back to the valuation date using WACC. The exit multiple and perpetuity growth approaches are typically presented side by side; when they produce similar values, it provides comfort that neither method is an outlier and that the assumptions are internally consistent.

---

## 52. Terminal Value — Perpetuity Method

The perpetuity growth method calculates terminal value as the present value of a perpetuity growing at a constant long-term rate. The formula is: TV = FCF_n × (1 + g) / (WACC − g), where FCF_n is the normalized free cash flow in the final explicit forecast year and g is the assumed long-term growth rate (typically set near long-run nominal GDP growth, e.g., 2–3%). For Apple, $75B in terminal-year UFCF growing at 3% discounted at 9% produces a terminal value at end of stage one of approximately $1.3 trillion. Discounting this back five years to the valuation date at 9% WACC produces a present value of approximately $1.1 trillion — which accounts for the majority of Apple's total enterprise value and illustrates why the long-term growth rate assumption is so consequential.

---

## 53. The Two-Stage DCF Approach

The two-stage DCF is demonstrated end-to-end on the hot dog stand. In stage one, each of five years of explicitly forecasted cash flows is discounted individually to time zero. In stage two, the perpetuity formula is applied at the end of year five to produce a terminal value representing the present value, as of year five, of all cash flows from year six onward. This year-five terminal value is then discounted back an additional five periods to time zero. Summing the present values of all stage-one cash flows and the discounted terminal value yields a total intrinsic value of $323,547 for the hot dog stand. The exercise makes tangible the mechanics that underlie every professional DCF regardless of the company's size or complexity.

---

## 54. Terminal Value as % of Total and Implied Multiple

For most mature companies, terminal value represents a strikingly large proportion of total enterprise value — Apple's terminal value accounts for approximately 77.6% of the DCF-derived EV. This concentration is a frequently cited criticism of the DCF: if most of the value is in the terminal value, the analysis is highly sensitive to long-term growth and discount rate assumptions that are genuinely unknowable. As a sanity check, the implied exit multiple at end of stage one is computed as TV / terminal-year EBITDA = approximately 12.7× for Apple. If this implied multiple is wildly inconsistent with where comparable companies trade today, it signals that either the long-term growth rate or the discount rate assumption needs revisiting.

---

## 55. UFCF Forecasting Mechanics

The transcript covers the specific mechanics of forecasting each line item of the UFCF build-up. EBITDA margin is forecasted directly as a percentage of revenue (declining from approximately 29% to 28% over the explicit period), and EBITDA is computed as revenue × margin. EBIT is derived by subtracting D&A, which is itself backed out as EBITDA − EBIT rather than forecasted independently. CapEx is modeled as a percentage of revenue, declining from approximately 4% to 3.6% over five years as the business matures. The effective tax rate of approximately 16% is sourced from Capital IQ consensus estimates. Each assumption is explicitly shown as a driver row so that sensitivity analysis can flex any individual assumption cleanly.

---

## 56. Understanding Net Debt

This video provides a conceptual grounding for the net debt calculation used in the EV-to-equity bridge. Book value of debt is nearly always an acceptable proxy for market value (unlike equity, debt book values rarely diverge dramatically unless rates have moved substantially). All debt tranches from the 10-Q must be included: commercial paper, revolving credit, term loans, senior notes, and subordinated notes. Any convertible securities that are assumed converted are excluded from net debt (shares are issued instead). Non-controlling interest represents a third-party financial claim against the consolidated business and must be subtracted from enterprise value alongside net debt. Interest income is deliberately excluded from UFCF — because the corresponding cash balance is already captured in the net debt bridge, double-counting would occur if interest income were also included in cash flows.

---

## 57. Unlevered vs. Levered DCF — Concept Check

A brief quiz-style video tests the conceptual distinctions between the two DCF frameworks. In the unlevered DCF: the cash flow is UFCF (before financing), the discount rate is WACC (blended cost of all capital), and the output is enterprise value (value to all capital providers). In the levered DCF: the cash flow is LFCF (after interest and debt repayment), the discount rate is cost of equity only, and the output is equity value directly. UFCF is always numerically larger than LFCF because LFCF subtracts debt service. Both approaches are theoretically equivalent: if applied consistently with matching cash flows, discount rates, and capital structure assumptions, they should produce the same equity value.

---

## 58. Unlevered vs. Levered DCF — Approach

The unlevered DCF is the standard approach for most industries because it cleanly separates operating performance from financing decisions, making it easier to compare across companies with different capital structures and to model changing leverage over time. UFCF = EBIT(1 − t) + D&A ± ΔNWC − CapEx. The levered approach uses levered free cash flow = cash from operations − CapEx − scheduled debt repayments, discounted at the cost of equity. Banks and other financial institutions are the primary exception to the unlevered default: for banks, interest income is the core revenue line and interest expense is the primary operating cost, so separating financing from operations is impossible; the levered approach is used instead, discounting dividendable cash flows to equity holders at the cost of equity.

---

## 59. Value Drivers

This conceptually rich video derives the fundamental drivers of DCF value from first principles. Free cash flow equals operating profit × (1 − reinvestment rate), where reinvestment rate is the fraction of operating profit reinvested back into the business (CapEx + NWC investment − D&A) / operating profit. Growth equals reinvestment rate × ROIC (return on invested capital). Substituting into the perpetuity formula reveals that enterprise value = operating profit × (1 − g/ROIC) / (WACC − g). This decomposition shows that the EV/EBIT multiple is driven by exactly three things: WACC, reinvestment rate, and ROIC. Crucially, if ROIC equals WACC, reinvestment creates no value — growth only creates value when a company earns above its cost of capital. This insight explains why DCF-implied multiples may differ from comps-derived multiples: peers with different ROIC, reinvestment, and risk profiles will rationally trade at different multiples.

---

## 60. WACC Concepts — Part 1

This video provides a comprehensive conceptual treatment of the weighted average cost of capital formula. WACC combines the after-tax cost of debt and the cost of equity, weighted by each source's proportion of total capital at market value. The after-tax adjustment to the cost of debt reflects the interest tax shield: if the pre-tax cost of debt is 10% and the marginal tax rate is 40%, the effective after-tax cost is only 6%, because each dollar of interest reduces taxable income and thus saves 40 cents in taxes. A standard DCF assumes a constant WACC throughout — implying a stable capital structure — because changing WACC each period would require re-levering/unlevering cash flows annually, complicating implementation significantly. For equity weights, public companies use market cap; private companies may need to iterate (with Excel's circular reference iteration enabled) or use comparable peer equity values. Debt weights typically use book value of debt as a proxy for market value.

---

## 61. WACC Concepts — Part 2

The cost of equity is the most contested component of WACC because, unlike the cost of debt, it is not directly observable in the market. The Capital Asset Pricing Model (CAPM) is the most widely used framework despite significant academic criticism. CAPM distinguishes between unsystematic (company-specific, diversifiable) risk — which investors receive no return premium for bearing — and systematic (market) risk, which cannot be diversified away and therefore commands a return premium. The cost of equity formula is: cost of equity = risk-free rate + beta × equity risk premium. Beta measures a company's sensitivity to market-wide movements: a beta of 2 means the stock moves twice as fast as the overall market in both directions. Other models (Fama-French, Arbitrage Pricing Theory) exist but are less common in practice.

---

## 62. WACC Concepts — Part 3

The risk-free rate in CAPM should theoretically match the duration of the cash flows being discounted; since DCFs extend to perpetuity, the 10-year US Treasury yield is the standard proxy in the United States. Equivalent 10-year government bond yields are used for European (Germany) and Asian (Japan) companies. The equity risk premium (ERP) represents the expected excess return of investing in equities over a risk-free instrument. The prevailing approach uses the historical spread between S&P 500 returns and 10-year Treasury yields over approximately 80 years, which suggests an ERP in the 5%–8% range; IBBOTSON (Morningstar) is the leading data source. Additional premiums are added for small companies (below ~$4B market cap) and for companies operating in higher-risk countries, neither of which is beta-adjusted — they are added as flat increments to the CAPM cost of equity.

---

## 63. WACC Concepts — Part 4

Beta interpretation is the focus of this final WACC video. A beta of 1.0 means the stock historically moves in line with the overall market; a beta of 2.0 means it moves twice as much. Zero-beta assets (US Treasuries, cash) have no relationship to market fluctuations. Negative-beta assets (gold, certain insurance products) are flight-to-safety instruments that tend to rise when equity markets fall. Consumer staples companies have low betas because demand for essential products is largely insensitive to economic conditions. Luxury and highly discretionary businesses carry high betas because their revenues are acutely tied to consumer confidence and aggregate economic conditions. Beta is the only company-specific variable in CAPM — the equity risk premium, small cap premium, and country risk premium are all market- or category-level adjustments that are independent of the individual company being analyzed. The video closes with a Home Depot WACC exercise: 6.8% cost of equity (2% risk-free + 0.8 beta × 6% ERP), 90.8% equity weight, producing a WACC very close to the cost of equity given Home Depot's low leverage.

---

*End of summaries — 63 transcripts covered.*
