# BUS 696 M&A Modeling — Resummarized Transcripts

Each transcript condensed to 4–8 sentences covering the key conceptual takeaways.

---

### 1. MA-Modeling-Apple-Acquires-Disney-General-Assumptions

Case study setup: a junior banker pitching Apple's hypothetical acquisition of Disney as of September 8, 2014. The model is organized into sections: transaction assumptions, sources & uses of funds, purchase price allocation (PPA) with write-ups, goodwill, pro forma balance sheet, credit statistics, accretion/dilution analysis, sensitivity tables, and contribution analysis. Key inputs are Apple's current share price ($98.76), Disney's current share price ($90.40), and an assumed 25% acquisition premium that yields an offer price of approximately $113 per Disney share. The model is designed to let analysts sensitize key assumptions—offer premium, % stock vs. cash—and observe the downstream impact on accretion/dilution and credit metrics.

---

### 2. MA-Modeling-Asset-Sales-338-Election-vs-Stock-Sales-DTAs-DTLs

Acquisitions can be structured as either stock sales or asset sales (or 338(h)(10) elections), each generating very different tax outcomes and two separate "sets of books" (GAAP vs. tax). In a stock sale, only the book basis of target assets is written up to fair market value; the tax basis remains at the original (historical) cost, creating a divergence that gives rise to deferred tax liabilities (DTLs) for the acquiring company. In an asset sale or 338(h)(10) election, both book and tax bases are stepped up to fair market value, so the acquirer gets real incremental depreciation and goodwill amortization deductions that generate actual tax savings—no deferred taxes arise because the bases converge. Exceptions are land (non-depreciable, so no timing difference) and goodwill (permanent rather than temporary difference under certain structures). Acquirers strongly prefer asset sales for these tax benefits, though target shareholders often resist because they trigger immediate capital gains recognition rather than the deferred treatment available in a stock sale.

---

### 3. MA-Modeling-Balance-Sheet-Historicals

This video covers entering Apple's and Disney's Q3 2014 balance sheets into the merger model. Cash is aggregated as the sum of cash & equivalents, short-term marketable securities, and long-term marketable securities for both companies. Equity is entered as a single lump sum rather than being broken into individual components at this stage. A balance check formula confirms that assets = liabilities + equity (should equal zero deviation), and once balanced, the model moves on to the PPA section.

---

### 4. MA-Modeling-Buy-Side-Process

The buy-side M&A process unfolds across four broad stages: (1) preliminary assessment, where the acquirer internally evaluates strategic fit and does initial valuation; (2) contacting targets, which involves signing NDAs and conducting preliminary valuation discussions; (3) due diligence, beginning with a letter of intent (LOI) and covering both business and financial due diligence; and (4) definitive agreement and closing, which includes arranging financing, filing with the SEC, and obtaining shareholder and regulatory approvals. The full timeline from initial assessment to closing typically runs 16–36 weeks depending on deal complexity. Investment bankers at each stage advise on structure, perform analysis, and interface with counterparties. Junior bankers provide the grunt-work analysis that underlies each stage's presentations and negotiations.

---

### 5. MA-Modeling-Calendarization

The calendarization schedule aligns the acquirer's and target's forecasts to a common fiscal year using a YEARFRAC-based weighting formula. The calendarized EPS for the target equals a blend of the prior-period, current-period, and next-period EPS weighted by how much of each falls within the acquirer's fiscal year. In the Apple/Disney case, both companies share a late-September fiscal year-end, so no calendarization adjustment is strictly needed—but the formula is built dynamically so it applies correctly when target and acquirer have different year-ends. For years beyond sell-side analyst coverage (typically 2–3 years out), a long-term growth rate from FactSet (14.5%) is used to extend EPS estimates.

---

### 6. MA-Modeling-Cash-vs-GAAP-EPS

Accretion/dilution models include a toggle between GAAP EPS and Cash EPS using an IF statement that zeros out non-cash adjustment line items (primarily incremental D&A from PPA write-ups and financing fee amortization) when "Cash EPS" is selected. FactSet provides both GAAP and non-GAAP EPS estimates for both Apple and Disney, so standalone EPS inputs can also be switched to match the selected basis. The real driver of accretion/dilution is the incremental impact of the deal's adjustments on the combined EPS, not the standalone measurement basis itself. Dynamic row headers in the model update automatically to display "Pro Forma GAAP EPS" or "Pro Forma Cash EPS" as the toggle changes, ensuring visual clarity in the output table.

---

### 7. MA-Modeling-Conceptually-Understanding-Contribution-Analysis

Contribution analysis quantifies each company's percentage contribution of revenue, EBITDA, net income, and enterprise value to the combined entity and identifies whether each dollar of financial performance is valued equivalently in the market. In the Apple/Disney case, Apple contributes ~80% of revenues and EBITDA but only ~68% of enterprise value, revealing that Disney's each dollar of earnings is valued at a higher multiple than Apple's. This multiple discrepancy explains structurally why any stock deal between the two will be dilutive—Apple's stock is "cheaper" relative to earnings than Disney's. Contribution analysis extends to an implied target valuation by applying the acquirer's market multiple to the target's financial metrics: assume the combined EV equals the acquirer's EV divided by the acquirer's revenue contribution percentage, then subtract the acquirer's standalone EV to arrive at an implied target EV, net debt, and per-share value. This provides an acquirer-centric benchmark that strips away growth premium differences and asks "what would this target be worth if each dollar of revenue were valued the same as ours?"

---

### 8. MA-Modeling-Conclusion

The course concludes after covering the M&A process and timeline, current trends, key M&A concepts, and a full modeling exercise using the Apple/Disney case study. Topics spanned the accretion/dilution model, PPA/goodwill/write-ups, deferred taxes, sources & uses, pro forma balance sheet, credit statistics, contribution analysis, and exchange ratio structures. Students seeking deeper coverage of granular deal mechanics are directed to the advanced transaction modeling course. The modeling skills built here underlie the pitch books, fairness opinions, and offering memoranda that investment bankers produce in real deals.

---

### 9. MA-Modeling-Credit-Statistics

After completing the pro forma balance sheet, the model calculates key credit statistics on both a standalone and pro forma basis. Net debt = total gross debt – cash & equivalents (note: Apple has massive cash balances making its standalone net debt negative; the pro forma figure reflects new borrowings). Equity value for standalone companies = share price × diluted shares outstanding; for the pro forma acquirer, it equals Apple's share price × (pre-deal shares + newly issued deal shares), assuming the share price is unchanged (a conventional simplification). Enterprise value = equity value + net debt; EV/EBITDA, gross debt/EBITDA, and debt/equity ratios are then calculated. Switching between all-stock and all-cash deal structures shows that enterprise value is nearly invariant to deal structure (it reflects the value of operations), while equity value and leverage ratios shift dramatically—illustrating why deal structure must be sensitized carefully.

---

### 10. MA-Modeling-Deal-Assumptions

Deal-level assumptions include: 60% stock / 40% cash deal structure, stock sale tax structure, 0.1% deal fees on offer value (~$200M), and $100M in estimated pre-tax synergies (a placeholder with no real basis). Acquirer shares issued = (offer value × % stock) ÷ Apple share price; nominal exchange ratio = offer price ÷ Apple price (1.14x); actual exchange ratio adjusts for partial stock consideration. Calendar-year EBITDA forecasts for Apple and Disney are built by summing the appropriate four quarterly FactSet figures, with dynamic row headers identifying the period. Tax rates are sourced from each company's most recent 10-Q income tax footnote: Disney 34.8%, Apple 26.1% (last 9 months ended). A critical nuance: Apple's 10-K shows 4M options at $140 strike pre-split, but the 10-Q corrects this to 13M options at $22.39 post-7-for-1 split—always verify corporate actions (Bloomberg CAC command or Google search) before relying on 10-K option data.

---

### 11. MA-Modeling-Deferred-Tax-Exercise

This exercise isolates the mechanics of DTL creation in a stock sale by working through a simple two-year example with a $400M book PPE write-up (tax basis stays at $300M), straight-line depreciation, 40% tax rate, and $500M cash revenues. Book depreciation = $200/year; tax depreciation = $150/year, creating a $50/year difference in pre-tax profit → $20/year gap in actual taxes paid ($140) vs. GAAP taxes ($120) → $40 total DTL recognized on deal day to acknowledge the cumulative future tax excess. The balance sheet at deal day must reflect this: PP&E debit of $100M (write-up), offset by DTL credit of $40M and equity credit of $60M (not the full $100M). Over the two-year asset life, the DTL reverses to zero as the temporary difference unwinds, confirming it is a timing difference rather than a permanent one.

---

### 12. MA-Modeling-Diluted-Shares

Diluted share counts for both Disney (target) and Apple (acquirer) are computed from SEC filings, but with an important distinction: for the target, outstanding options (not just exerciseable ones) are used because a change-of-control accelerates vesting. The treasury stock method applies: option proceeds ÷ offer price = shares bought back; net dilution = gross option dilution – buyback. Unvested RSUs of the target also vest upon change of control → included in Disney's diluted count (adds ~21M shares). For Apple (no change of control), only exerciseable options count and unvested RSUs are excluded. Disney has no convertibles or preferred stock (confirmed by searching 10-K and 10-Q for "convertible"). Final Disney diluted share count ≈ 1.76 billion → offer value = 1.76B × $113 ≈ $199 billion.

---

### 13. MA-Modeling-Financing-Assumptions

The transaction financing section breaks the 40% cash consideration into two sources: 80% financed with new debt ($63B) and 20% from Apple's existing cash ($15.9B), validated against Apple's $164B total liquid assets (~10% utilization). If Disney's debt is refinanced rather than assumed, an additional $16.1B must be raised, bringing total new borrowing to ~$79B. Financing fees = 0.5% × total acquisition financing → ~$400M, capitalized and amortized over the 5-year loan term at $80M/year (distinct accounting treatment from deal fees, which are expensed as incurred). New debt carries a 4% interest rate; Apple's cash earns 1%, so the opportunity cost of using cash is lower than the debt cost—relevant to accretion/dilution. All amounts flow into dynamic row headers using DATE and CONCATENATE formulas tied to the deal date.

---

### 14. MA-Modeling-Fixed-vs-Floating-Exchange-Ratios-Part-1

The exchange ratio defines how many acquirer shares must be issued per target share: nominal ratio = offer price ÷ acquirer price (1.14 Apple shares per Disney share); actual ratio adjusts for partial stock consideration. In a fixed exchange ratio deal, the ratio is locked at announcement regardless of subsequent price movements—if Apple's share price falls before closing, Disney shareholders receive less value (they bear acquirer price risk). In a floating exchange ratio (fixed value) deal, the ratio adjusts so Disney always receives $113 worth of Apple stock per share; Apple must issue more shares if its price declines. Acquirers generally prefer fixed ratios (their dilution is defined); targets prefer floating ratios (their value is locked). Most large public deals use fixed exchange ratios.

---

### 15. MA-Modeling-Fixed-vs-Floating-Exchange-Ratios-Part-2

This video presents fixed vs. floating exchange ratios graphically using line charts. A fixed exchange ratio produces a diagonal value line—the deal value for the target rises and falls linearly with the acquirer's share price. A floating exchange ratio (fixed value) produces a flat horizontal line—$199B regardless of Apple's price, because the ratio floats to compensate. Plain vanilla floating exchange ratios with no collar are almost never used in practice; acquirers request protections (collars) at some point. The charts visually crystallize the risk-sharing tradeoff between the two structures.

---

### 16. MA-Modeling-Fixed-vs-Floating-Exchange-Ratios-Part-3

Collars add a floor and cap to limit each party's exposure. In a fixed exchange ratio deal with a collar: below $90/share (Apple), the value becomes fixed at $181B floor (ratio floats up, protecting Disney); above $106, the value is capped at $205B (ratio floats down, limiting Apple's excess cost); below $82, target walk-away rights trigger. In a floating exchange ratio deal with a collar: Apple locks its dilution below a floor price; Disney shares in upside if Apple's price rises above a cap; walk-away rights apply on the low end for Disney. The typical structure is a symmetric floor/cap around the announcement price with bilateral walk-away rights at extreme levels. Simpson Thatcher publishes a reference document listing recent deals' specific exchange ratio structures.

---

### 17. MA-Modeling-Income-Statement-Adjustments-in-MA

Four categories of adjustments must be made before combining net incomes in an accretion/dilution model: (1) acquisition financing—incremental interest expense if new debt is raised, or lost interest income if acquirer cash is used (both reduce combined net income); (2) synergies—pre-tax cost savings or revenue upside added back positively; (3) fees—deal advisory/legal/accounting fees expensed immediately; financing (underwriting) fees capitalized and amortized over the loan term; (4) accounting adjustments—incremental D&A from PPA write-ups to FMV (non-cash, reduces GAAP EPS; excluded from Cash EPS) and goodwill (no amortization under current GAAP, only impairment). These adjustments explain why even a 100% stock deal is not simply net income addition—the accounting and financing effects matter. The standard accretion/dilution presentation is a two-variable sensitivity data table (offer price × % stock) showing the resulting EPS impact.

---

### 18. MA-Modeling-Intermediate-AccretionDilution-Cocktail-Exercise

Google acquires LinkedIn exercise: $260 offer price, $200 pre-deal price (30% premium), $31.2B offer value, 50/50 stock/cash, 40% tax rate for both, $100M synergies, $120M deal fees, $100M financing fees, 5-year term at 5%, $200M write-up with 10-year life. Sources & uses: $15.6B stock + $15.8B new debt = $31.4B uses. Numerator adjustments: combined pre-tax income ($23.8B), less incremental interest expense ($789M), less incremental D&A ($20M), less financing fee amortization ($20M), less deal fees ($120M), plus synergies ($100M) = $23.0B adjusted pre-tax → after-tax ~$13.8B. Denominator: Google's 700M shares + 26M new shares = 726M. Pro forma EPS = $19.03 vs. standalone $20 → 5% dilutive. Insight: the higher LinkedIn's P/E relative to Google's, the more dilutive the stock component; the higher the borrowing rate, the more dilutive the cash component.

---

### 19. MA-Modeling-MA-Accounting-Overview

FASB's acquisition method requires that all target assets and liabilities be rewritten to fair market value on deal day—something that cannot be done in normal course of business (conservation and historical cost principles). Purchase price allocation (PPA) proceeds in three steps: (1) eliminate pre-existing target goodwill to isolate tangible book value; (2) write up depreciable assets and intangibles to FMV; (3) allocate any remaining excess of purchase price over FMV of net assets to new goodwill. Common write-up candidates: PPE (especially land and buildings), intangible assets (brands, patents, customer relationships), and LIFO inventory (old layers often undervalued in rising-price environments). Goodwill under current GAAP is not amortized—it sits on the balance sheet until impaired, so it does not flow through the income statement absent an impairment charge. Practitioners often skip write-up estimates (FMV appraisals come late in the deal process) and approximate goodwill as purchase price minus book value of equity.

---

### 20. MA-Modeling-Modeling-Pro-Forma-Adjustments

Building the combined pro forma balance sheet requires recording deal adjustments in two columns (debits/credits) before summing. Cash decreases by cash paid to Disney shareholders ($15.9B) plus fees paid in cash ($1.3B). Accounts receivable and inventory are simply summed; PPE gets the $4.6B write-up and intangibles get the $7B write-up. Goodwill: eliminate Disney's pre-existing goodwill ($27B), add newly created goodwill (~$170B) → net addition of ~$143B. Disney's DTA ($480M) is written off (NOL assumption). New DTLs (from stock sale) are recognized. Disney's debt tranches are zeroed out (refinanced) and new long-term borrowing (~$79B) is added. Disney's entire equity is eliminated; deal fees (~$1B) reduce Apple's retained earnings; financing fees ($399M) create a new intangible asset; Apple's newly issued stock ($119B) increases equity. Balance check: the adjustment column should net to zero.

---

### 21. MA-Modeling-Modeling-the-Contribution-Analysis

Pull calendarized revenue, EBITDA, and net income from the main model (excluding all deal-related adjustments) and compute each company's % share of the combined total for each metric. Compute standalone multiples (EV/Revenue, EV/EBITDA, P/E) for Apple, Disney, and the pro forma combined entity. Apple's lower multiples versus Disney's confirm why the deal is dilutive under stock consideration—each dollar of Apple earnings is worth less to the market. For the implied target valuation, divide Apple's standalone EV by Apple's revenue contribution % to derive the implied combined EV; subtract Apple's EV to get Disney's implied EV; subtract Disney's net debt to get implied equity value; divide by Disney's shares for implied share price. Contribution analysis charts: stacked horizontal bar for % contributions; clustered column for multiple comparisons; clustered column for implied vs. offer share price comparison.

---

### 22. MA-Modeling-Modifying-the-Calendarization-Schedule

Add EBITDA and revenue forecasts from FactSet (Disney and Apple) to the calendarization tab in the same structure as the EPS forecasts. For targets with different fiscal year-ends, include a prior-period and next-period forecast row so the YEARFRAC weighting formula can blend across fiscal boundaries. For the last year where analysts do not provide EBITDA/revenue growth explicitly, extrapolate by applying the most recent observed period-over-period growth rate (no long-term growth rate is available for EBITDA/revenue from FactSet, unlike EPS). This is an approximation but the standard industry workaround when analyst coverage runs out.

---

### 23. MA-Modeling-Overview

An M&A model (also called an accretion/dilution model) shows how combining two companies affects the acquirer's EPS, credit profile, and valuation. A deal is accretive when combined pro forma EPS > standalone acquirer EPS for the same period; dilutive when the reverse holds. Public acquirers care significantly about EPS optics—press releases often tout accretion to reassure investors and support the acquirer's post-announcement share price. Acquirers can pay with cash (using existing reserves or new debt), stock, or a combination; historically ~58% of deals by dollar volume are cash-only, ~28% mixed, ~10% pure stock. The choice of consideration carries major legal, tax, and accounting implications explored throughout the course. The model is built around the Apple/Disney case study, where Apple (lower P/E) acquires Disney (higher P/E), creating structural dilution if significant stock is used.

---

### 24. MA-Modeling-PPA-Goodwill-Write-Ups

Disney's balance sheet pre-deal: PPE $23B, intangibles $7B, goodwill $27B, total equity ~$45B. For modeling purposes, PPE write-up = 20% of $23B = $4.6B; assigned 20-year useful life → $230M/year incremental depreciation. Intangibles write-up = 100% of $7B = $7B; assigned 15-year useful life (aligned with IRS 15-year amortization rule for intangibles) → $466M/year incremental amortization. DTLs from the write-ups = total write-up × acquirer's (Apple's) tax rate (not Disney's), because the DTL reflects Apple's future tax burden. Disney's pre-existing DTA of $480M is assumed to be NOL-related and written off (both asset and stock sales eliminate or limit NOL benefit for the acquirer). New goodwill = $199B offer value – $28.5B FMV of net assets ≈ $170B, making goodwill the largest pro forma asset by far.

---

### 25. MA-Modeling-Pre-Deal-DTAs-DTLs-NOLs-and-Summary

Pre-existing DTLs in an asset sale: since both book and tax bases converge to FMV, the basis differences that created those DTLs disappear → existing DTLs are eliminated (a credit to equity). In a stock sale: the original basis differences persist, so existing target DTLs carry over and are magnified by new DTLs from write-ups. Pre-existing DTAs from NOLs in an asset sale: the acquirer cannot use the target's NOLs → write them off in the model. In a stock sale: NOLs transfer to the acquirer but are subject to an annual IRC §382 cap = purchase price × IRS long-term tax-exempt rate (~3%), so large NOL balances must be substantially written down. Non-NOL DTAs (e.g., revenue recognition timing differences) generally carry over in both structures. Summary rule: asset sale → existing DTLs gone, new DTLs none, NOLs unusable; stock sale → existing DTLs persist, new DTLs created, NOLs limited by §382.

---

### 26. MA-Modeling-Rule-of-Thumb-for-DTLs

The general rule for calculating the DTL created in an M&A stock sale is: DTL = (fair value book basis − tax basis) × tax rate. This can be verified at each period end by multiplying the remaining basis difference by the tax rate, which should equal the DTL balance on the balance sheet. At deal day, the full write-up amount × tax rate creates the initial DTL. Over the asset's depreciable life, the basis difference narrows as the book basis depreciates faster (higher) than the tax basis (lower/unchanged), and the DTL balance unwinds correspondingly each year. By the end of the asset's useful life, both bases reach zero and the DTL is fully reversed to zero, confirming it is a temporary timing difference.

---

### 27. MA-Modeling-Sample-Pitchbooks-Fairness-Opinions-and-OMs

This video introduces the key M&A work-product documents that investment bankers produce. Sample pitch books included are from real deals (J.Crew/TPG, Genentech/Roche), providing a view into what buy-side and sell-side presentation decks look like. The offering memorandum (OM), also called a confidential information memorandum (CIM), is a 50+ page document prepared by sell-side bankers presenting the seller's financials, investment highlights, and business overview to potential buyers (example: American Casino OM). Fairness opinions are independent documents presented to the board of directors attesting that the deal price is financially fair to shareholders; they incorporate comps, DCF, and contribution analysis. The underlying technical modeling covered in the course is what populates these documents.

---

### 28. MA-Modeling-Sell-Side-Process

The sell-side M&A process mirrors the buy-side across four stages but is typically shorter and more definitive once initiated: (1) strategy and target-buyer identification, culminating in the CIM/OM; (2) contacting potential buyers, distributing the OM under NDA, and gauging interest; (3) receiving letters of intent and managing due diligence requests through the data room; (4) negotiating final terms and executing the definitive agreement. The data room is a junior banker responsibility—uploading financial information in response to buyer requests, often with selective redaction to protect sensitive client details. Information flows progressively as buyer interest deepens: broad brush first, then more granular financials as NDAs are signed and LOIs received. Junior bankers on sell-side engagements prefer them over buy-side because the process is more likely to actually close once the seller commits.

---

### 29. MA-Modeling-Sensitivity-Analysis-Part-1

Sensitivity (data table) analysis for accretion/dilution uses Excel's two-variable data table (Alt+D+T) with offer price as the row input and % stock as the column input. For the Apple/Disney model, offer prices range $95–$125 ($10 increments) and % stock from 100% to 0% (25% decrements); hit F9 to manually recalculate (Excel is set to automatic except for data tables to avoid performance slowdowns). Key finding: lower offer price → less dilution; less stock → less dilution (because Apple's P/E < Disney's P/E, making stock the expensive currency). Five data tables planned: Year 1, Year 2, Year 3 accretion/dilution; financing mix (% debt vs. cash for cash consideration) vs. accretion; and offer price vs. % stock vs. leverage ratio (Debt/EBITDA).

---

### 30. MA-Modeling-Sensitivity-Analysis-Part-2

Completing the remaining four data tables. Year 2 and Year 3 dilution are structured identically to Year 1 but reference later period EPS outputs; both show the deal becoming less dilutive over time as synergies and growth offset deal costs. The financing mix table (% cash vs. debt for cash consideration) shows negligible impact when the deal is 100% stock (no debt raised at all) but meaningful impact at 0% stock: using Apple's excess cash (1% opportunity cost foregone) is more accretive than borrowing at 4%. The leverage table (Debt/EBITDA) shows that higher offer prices and more debt both increase leverage—relevant if there are credit rating or covenant considerations post-deal. At this point the core accretion/dilution and credit model is complete; the next sections cover contribution analysis and exchange ratio sensitivity.

---

### 31. MA-Modeling-Simple-AccretionDilution-Exercise

Conceptual walkthrough of the simplest accretion/dilution case: 100% stock deal, no premium, acquirer at $25/share (P/E = 10), target at $60/share (P/E = 12). Exchange ratio = $60 ÷ $25 = 2.4 acquirer shares per target share. New shares issued = 2.4 × 1,000 target shares = 2,400; total pro forma shares = 6,400. Combined net income = $10,000 + $5,000 = $15,000. Pro forma EPS = $15,000 ÷ 6,400 = $2.34 vs. standalone $2.50 → 16¢ dilutive (6.4%). Key interview insight: in a 100% stock deal, the deal is dilutive when target P/E > acquirer P/E and accretive when target P/E < acquirer P/E—because a lower-P/E acquirer must issue fewer of its "cheaper" shares per dollar of target earnings. This rule-of-thumb applies only in pure stock deals; cash deals depend on the cost of financing.

---

### 32. MA-Modeling-Sources-Uses-of-Funds

The sources and uses (S&U) schedule is built by first establishing uses, then sizing sources to match. Uses: (1) equity value to target = offer price × diluted target shares (the % stock slice is stock, the % cash slice is cash); (2) refinancing of target debt if selected (IF statement based on assumed vs. refinanced toggle); (3) deal advisory fees; (4) financing fees (separated because accounting treatment differs). Sources: (1) value of new acquirer stock issued; (2) new debt raised; (3) acquirer's existing cash used. New debt = total uses − stock issued − acquirer cash used. A balance check formula confirms sources = uses; if the model is out of balance, a flag ("NO") triggers. The S&U schedule feeds directly into the pro forma balance sheet cash and debt adjustments.

---

### 33. MA-Modeling-The-Current-MA-Environment

Contextual overview (circa 2014): global M&A volumes reached their highest level since 2007, up ~50% YoY in H1 2014, with EBITDA deal multiples approaching 11–12× and 42% of deals carrying 25–50% premiums. The US accounted for ~50% of global volume; bulge-bracket banks (Goldman, Morgan Stanley, JPMorgan, BofA, Citi) dominate league tables. Consumer non-cyclicals and financial services comprise ~50% of deal count. PE buyers represented 17% of 2014 deal volume (down from 25% in 2013) as strategic buyers returned to the market amid strong equity markets and CEO confidence. Video includes CNBC interview clips with Citi's M&A head contrasting 2014's buoyant deal environment with the cautious early-2012 environment. Deal-size distribution: ~97% of deals by count are small/mid-market; a handful of mega-deals dominate by dollar volume.

---

### 34. MA-Modeling-The-Role-of-the-Banker-in-MA

The M&A ecosystem involves buyers (business development, management/board, shareholders), sellers (same groups), and supporting players (investment bankers, accountants, tax advisors, lawyers, regulators for antitrust/disclosure). Investment banks advise on optimal deal structure and terms, facilitate capital access, identify counterparties, and negotiate on behalf of clients. The three main IB engagement types are buy-side advisory, sell-side advisory, and fairness opinions (independent valuations delivered to boards and included in shareholder proxy materials). Senior bankers cultivate relationships and pitch M&A ideas to management teams; junior bankers build the models and PowerPoint pitch books that underlie those pitches. A real sell-side pitch book (Catalyst Partners pitching Autonomy to Oracle) is reviewed to illustrate pitch book structure: executive summary, comps analysis, annotated stock chart, financial performance overview, and inside ownership trends.

---

### 35. MA-Modeling-What-do-Buyers-and-Sellers-Care-About-Most

Target considerations center on: offer price (dominant), tax structure (targets strongly prefer stock sales because shareholders defer capital gains recognition; asset sales trigger immediate taxable gain), restrictions on newly issued acquirer stock, board/management representation in the combined company, compensation terms for management, earn-out provisions (common in smaller deals to bridge valuation gaps), and fixed vs. floating exchange ratio. Acquirer considerations include all the above plus: accounting implications of PPA write-ups (higher D&A reduces GAAP EPS; larger goodwill on balance sheet; must communicate changes to investors) and accretion/dilution to both GAAP EPS and Cash EPS (critical for investor optics, press releases, and post-announcement share price). The merger model built in this course directly quantifies the acquirer-side considerations: deal structure effects on EPS, balance sheet leverage, and credit statistics. Both sides also weigh synergy realization risk, financing availability, and regulatory/antitrust approval probability as deal-level factors.

---
