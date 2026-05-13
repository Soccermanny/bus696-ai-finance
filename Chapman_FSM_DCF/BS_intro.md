Balance Sheet Forecasting Guide
How to Build Balance Sheet Projection in Excel
Imagine that we are tasked with building a 3-statement model for Apple. Based on analyst research and management guidance, we have completed the company’s income statement projections, including revenues, operating expenses, interest expense and taxes – all the way down to the company’s net income.

Typically, the main balance sheet section of a model will either have its own dedicated worksheet or it will be part of a larger worksheet containing other financial statements and schedules. Before we dive into individual line items, here are some balance sheet best practices.

At least two years of historical data: It is recommended that at least two years of historical results are inputted into the model to help provide some context to forecasts. Data is organized in columns ascending from left to right.
Reclassify GAAP to suit your needs: Companies present their balance sheet in ways that are not always optimized for analysis. For example, companies may lump line items with different drivers together. In these cases, the line items need to be separated and forecasting approaches should be tailored to the nature of the items. Conversely, GAAP requires that certain line items be broken out into current and long-term components (deferred taxes and deferred revenue are common examples). However, for forecasting purposes, they can be combined because they are forecast using the same drivers.
Use supporting schedules: All forecasting needs to be done in supporting schedules — either in the same worksheet or in dedicated separate worksheets. This is where the forecasting and calculations should take place. The consolidated balance sheet simply pulls the finished product — the forecasts — to present a complete picture.
balance sheet forecast

Working Capital Forecast
We start the balance sheet forecast by forecasting working capital items.

Broadly speaking, working capital items are driven by the company’s revenue and operating forecasts.

Conceptually, working capital is a measure of a company’s short-term financial health.

The common working capital items include:

Accounts Receivable (A/R)

Grow with sales (net revenues).
Using an IF statement, model should enable users to override with days sales outstanding (DSO) projection, where days sales outstanding (DSO) = (AR / Credit Sales) x days in period.
Inventories

Grow with cost of goods sold (COGS).
Override with inventory turnover (Inventory turnover = COGS / Average inventory).
Prepaid expenses

If prepaid expenses comprise expenses predominantly classified as SG&A, grow with SG&A. If you aren’t sure, grow with revenue.
Other Current Assets

Grow with revenues (presumably these are tied to operations and grow as the business grows).
If there’s reason to believe they are not tied to operations, straight-line the projections.
Accounts payable

If the payables are generated predominantly for inventory, grow with COGS. If you aren’t sure, grow with revenue.
Override with payables payment period assumption.
Accrued Expenses

If the accrued expenses are largely for expenses that will be classified as SG&A, grow with SG&A. If you aren’t sure, grow with revenue.
Deferred revenue

Refers to sales that cannot be recognized as revenue yet. Examples include gift cards and software for which upfront payment implies rights to future upgrades.
Grow with the revenue growth rate.
Taxes Payable

Grow with the growth rate in tax expense on the income statement.
Other current liabilities

Grow with revenues.
If there’s reason to believe they are not tied to operations, straight-line the projections.
PP&E and Intangible Assets
The largest component of most company’s long term assets are fixed assets (property plant and equipment), intangible assets, and increasingly, capitalized software development costs.

These line items are also driven largely by the company’s operations. In other words, the more revenue, the more capital spending and purchases of intangibles we expect to see.

Unlike working capital, PP&E and intangible assets are depreciated or amortized (with a few notable exceptions like land and goodwill). This creates a layer of complexity in the forecasting, as illustrated below:

PP&E Roll-Forward
PP&E (BOP) + capital expenditures ‑ depreciation‑ asset sales = PP&E (EOP)

Line Item	How to Forecast
PP&E (BOP)	
Reference from last period’s EOP
Capital expenditures	
Use equity research or management guidance when available. In the absence of guidance, assume purchases in line with historical trends as a % of sales.
Depreciation	
Approach 1: Forecast as a % of capital expenditures using historical depreciation as a guide.
Approach 2: Depreciation waterfall analysis (useful when companies provide sufficient detail).
Asset sales	
Most companies do not regularly offload assets as a matter of course, so barring specific guidance, assume no assets sales.
That said, some industries (like REITs) require recurring asset sale forecasts.
Intangible Asset Roll-Forward
intangible assets (BOP) + purchases – amortization = intangible assets (EOP)

Line Item	How to Forecast
Intangible Assets (BOP)	Reference from last period’s EOP
Purchases	
Approach 1: Use equity research or management guidance when available.
Approach 2: In the absence of guidance, look at historical purchases (disclosed in the cash flow statement). If historical purchases are significant, grow as a % of sales. If historical trends are lumpy or undisclosed, assume no new purchases.
Amortization	Companies typically disclose future amortization expense for the current intangible assets in 10K footnote. Of course, if forecasting new purchases, this will have incremental impact on future amortization. In this case, apply the historical ratio of amortization/purchases.
Goodwill
Goodwill is usually straight-lined in a 3-statement financial model. In other words, if goodwill on the latest balance sheet is $400m, it stays at $400m indefinitely. (For more on goodwill, read our quick primer on how goodwill is created.) That’s because to do anything else would imply either:

Future goodwill impairment
or
Future acquisitions where the company pays in excess of the fair market value of the assets acquired.
It is difficult to reliably forecast such things. One exception to this is when modeling private companies that amortize goodwill.

Deferred tax assets and liabilities
Deferred taxes are a complex topic and, as you see below, are either grown with revenue or straight-lined in the absence of a detailed analysis.

Deferred tax assets	
Approach 1: Since most DTAs are tied to operations (revenue recognition timing differences and NOLs) grow with revenue.
Approach 2: Straight-lining is also acceptable in the absence of sufficient disclosures to fully understand the nature of the deferred taxes.
Deferred tax liabilities	
Approach 1: Since DTLs are often tied to a discrepancy between book and tax depreciation methods, DTLs will grow with operations over the long run. As a result, a common approach when the full nature of the DTLs isn’t known is to grow with revenue, just like DTAs.
Approach 2: Straight-lining is also acceptable in the absence of enough disclosures to fully understand the nature of the DTLs
Note that DTAs and DTLs can be classified in the financial statements as both current and non-current.

Other non-current assets and liabilities
You’ll often encounter catch-all line items on the balance sheet simply labeled “other.” Sometimes the company will provide disclosures in the footnotes about what’s included, but other times it won’t. If you don’t have good detail on what these line items are, straight-line them as opposed to growing with revenue. That’s because unlike current assets and liabilities, there’s a likelihood these items could be unrelated to operations such as investment assets, pension assets and liabilities, etc.

Long term debt
Below we see Apple’s 2016 debt balances. We observe that Apple has both short-term commercial paper and long-term debt (including a portion that’s due this year):

How to Build Balance Sheet Projection in Excel image 1

 

Let’s focus on long term debt for now and get back to the commercial paper later. Companies will usually provide a footnote disclosure of future maturities of long-term debt. In Apple’s 2016 10K, you can see a typical debt maturity disclosure which identifies all the upcoming maturities of long-term debt (including the $3.5 billion current portion of long term debt that is due in 2017):

How to Build Balance Sheet Projection in Excel image 2

So we know these notes will be coming due – after all, Apple is contractually required to pay them down. This might lead you to believe that forecasting debt is just a matter of reducing the current debt balances by these scheduled maturities. But a financial statement model is supposed to represent what we think will actually happen. And what will most likely actually happen is that Apple will continue to borrow and offset future maturities with additional borrowings.

That’s because most companies replace (or “refinance”) maturing debt with new debt. Companies do this to maintain a stable capital structure. This means that even when the footnotes disclose that debt will be paid down, it is more appropriate to assume that debt stays at current levels or grows to reflect a fixed capital structure. Mechanically we do this by either:

Holding the company’s long term debt balance constant (or)
Growing long term debt at the growth in the company’s net income (arguably a better approach because it ties debt to equity growth by using net income as a proxy for equity growth).
Shareholders equity
We have now identified the forecasting techniques for all assets and liabilities except for cash and the revolver. We now turn to forecasting the line items in the statement of shareholders’ equity. The four big line items in that section are:

Common Stock and APIC
Treasury Stock
Retained Earnings
Other Comprehensive Income
Common stock and APIC
Companies issue new common stock in one of two ways:

New stock issuance (IPO or secondary offerings)
Companies do this to raise capital, typically to fund growth. For example, if a company wants to raise $100m via an equity offering, they get $100m in cash (debit cash) with a corresponding $100m increase in common stock and APIC (credit).
Why do companies issue stock and how does it compare to raising money by borrowing from a bank? In some ways it’s like borrowing, but rather than paying interest, the share issuance dilutes existing equity owners.
How do we forecast future issuances? Since companies don’t issue stock (via IPO or secondary offering) on a regular basis, most of the time, no forecast of stock issuance from this is necessary (i.e. we assume no new share issuance unless there is specific justification).
Stock-based compensation
Companies issue stock-based compensation to incentivize employees with stock in addition to cash salary. Companies primarily issue stock options and restricted stock to employees.

Accounting for stock-based compensation
Although no cash exchanges hands when companies issue their employees options or restricted stock, companies must recognize an expense for this (which they estimate using an options pricing model). For example, if Apple gave an employee 1,000 stock options at $150 exercise price, and which vest equally over the next 2 years, Apple might estimate that this has a present value of $5,000 ($5 per option). This has the effect of debiting retained earnings (since stock-based compensation expense is accounted for as an operating expense), while the offsetting credit is common stock and APIC. Below you can see that Apple’s common stock and APIC account is increased by the $2.863b in stock-based compensation expense:
How to Build Balance Sheet Projection in Excel image 3

How do we forecast stock-based compensation expense?
The most common way to forecast stock-based compensation is to straight-line historical ratio of SBC to revenue or operating expense. Since stock-based compensation expense increases capital stock, whatever we forecast must increase common stock. Since it also reduces retained earnings but has no cash impact, we also need to add it back to net income in the cash flow statement (see below).
Treasury stock
Some companies buy back their own shares when they have excess cash. For example, if a company buys back $100 million of its own shares, treasury stock (a contra account) declines (is debited) by $100 million, with a corresponding decline (credit) to cash.

Conceptually, a share buyback is essentially a dividend to remaining shareholders paid in the form of additional ownership of the company. In our example, the $100 million that the company wants to return to shareholders can actually be achieved one of two ways: via a cash dividend or equivalently via a $100m buyback. The per share increase to each shareholder (all else equal) should amount to exactly $100 million in aggregate value. One benefit with the share repurchase approach is that unlike a cash dividend, tax can usually be deferred paid by shareholders on the buyback.

From a modeling perspective, barring some management guidance or thesis on future buybacks, if a company has engaged in recurring buybacks historically (the amount of buybacks can be found on the historical cash flow statement), straight-lining the amount into the forecast period is usually reasonable.

Forecasting shares outstanding and EPS
Share issuance and buybacks that we forecast on the balance sheet directly impacts the shares forecast, which is important for forecasting earnings per share.

Retained earnings
Retained earnings is the link between the balance sheet and the income statement. In a 3-statement model, the net income will be referenced from the income statement. Meanwhile, barring a specific thesis on dividends, dividends will be forecast as a percentage of net income based on historical trends (keep the historical dividend payout ratio constant).

The retained earnings roll-forward
retained earnings (BOP) + net income – dividends (common and preferred) = retained earnings (EOP)

Line Item	How to Forecast
Net income	From income statement forecast
Dividends (Common and Preferred)	Forecast as a % of net income based on historical trends.
Other comprehensive income (OCI)
Under GAAP, there are many financial activities whose gains and losses don’t impact net income: Gains and losses on foreign currency translations, derivatives, etc. Instead, they are classified as “other comprehensive income” (OCI) and are accumulated in a balance sheet line item distinct from retained earnings. You can see this in Apple’s balance sheet (observe that the line “accumulated other comprehensive income” declined by $1,427m during the year from an accumulated balance of $1,082 to a negative $354m):

How to Build Balance Sheet Projection in Excel image 4

And in a separate schedule in the 10K you can see a full breakout of $1,427m in year-over-year changes in OCI (much like the income statement is a breakout of the year over year changes in retained earnings):

other comprehensive income

Forecasting OCI
Forecasting OCI is fairly straightforward. Because the gains and losses that flow into this line item are difficult to predict, the safest bet is to assume no change year-over-year going forward (in other words, straight-line the last historical OCI balance on the balance sheet):

The other comprehensive income roll-forward:
OCI (BOP) +/- OCI generated during the year = OCI (EOP)

Line item (see formula above)	How to forecast
OCI generated during the year	Assume no OCI gains and losses in the forecast (i.e. straight-line historical OCI balance).
Forecasting Cash and Short Term Debt (Revolving Credit Line)
Last but not least, we turn to the forecasting of short term debt and cash. Forecasting short term debt (in Apple’s case commercial paper) requires an entirely different approach than any of the line items we’ve looked at so far. It is a key forecast in an integrated 3-statement financial model, and we can only quantify the amount of short term funding required after we forecast the cash flow statement. That’s because cash and short term debt (the revolver) serve as a plug in most 3-statement financial models – if after everything else is accounted for, the model is forecasting a cash deficit, the revolver will grow to fund the deficit. Conversely, if the model is showing a cash surplus, the cash balance will simply grow.

Balancing the Model
Finally, any balance sheet forecast isn’t complete if the balance sheet does not balance. While a company’s reported balance sheet will always show assets equaling liabilities plus equity, when forecasting the balance sheet, any number of mistakes can lead to the model getting out of balance. In fact, the strength of a 3-statement model is that the three statements are interlinked. However, these inter-linkages also increase the potential for error. Some of the most common reasons the balance sheet doesn’t balance include:

Signs (+/-) are switched: For example, if your capital expenditures is inputted in the balance sheet as a negative (or in the cash flow statement as a positive), your model will be out of balance.
Mislinks: For example, if your model accidentally references dividends instead of stock-based compensation into the common stock schedule, your model will be out of balance.
Cash flow statement errors: Getting a model to balance is usually more about getting the cash flow statement correct than it is about getting the balance sheet correct. For example, if you forecast that “other long term assets” on the balance sheet grow at the same rate as revenues but forget to include the cash impact of this change on the cash flow statement, your model will not balance.
How to Balance the 3-Statement Model
Print out the full model.
Beginning with the accounts receivable line on the B/S, calculate the cash impact of each line of the B/S with a calculator.
Once you’ve made the calculation, verify that this cash impact is correctly expressed on the cash flow statement.
Once verified on the CFS, cross off both the balance sheet and cash flow statement line items with a pencil.
Proceed to the next line and continue until you get to the last line of the balance sheet.
While this can be a time consuming process, the good news is that if you follow the above steps correctly, you will locate the error and your model will balance.