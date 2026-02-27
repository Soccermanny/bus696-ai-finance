# BUS 696: Generative AI in Finance

**Chapman University | Argyros College of Business and Economics | Spring 2026**

Artificial intelligence is transforming finance — how markets process information, how trades are executed in microseconds, how risk is measured and managed, how ordinary people access financial advice once reserved for the wealthy. This course uses the tools and frameworks of machine learning and data science to understand how finance is changing.

This is a hands-on, applied course. We read research papers, but we also run code — working with Jupyter notebooks to see how neural networks learn patterns in market data, how reinforcement learning agents discover trading strategies, and how backtesting reveals whether a model actually works or is just overfitting to noise.

## What You'll Learn

- **Finance fundamentals** through the lens of AI: returns, risk, portfolios, CAPM, options, and the Efficient Market Hypothesis
- **Machine learning for markets**: feature engineering, logistic regression, random forests, XGBoost, and why most ML trading strategies fail out-of-sample
- **Neural networks and deep learning** applied to financial prediction
- **Reinforcement learning**: agents that learn trading strategies from experience
- **Backtesting done right**: avoiding look-ahead bias, survivorship bias, overfitting, and the data snooping trap
- **Risk management** with AI: VaR, expected shortfall, and model risk
- **Alternative data**: satellite imagery, sentiment analysis, and the search for alpha
- **Crypto and DeFi**: blockchain fundamentals, Bitcoin, and decentralized finance

## Course Schedule

### Part I: Introduction to Finance

| Week | Date | Topic | Materials |
|------|------|-------|-----------|
| 1 | Feb 11 | **What is AI in Finance?** From human traders to algorithms to AI. Finance basics: stocks, bonds, returns. | [Slides](slides/BUS_696_Introduction-to-AI-in-Finance.pdf) / [Lab](labs/Week_01_Lab_Finance_Fundamentals.ipynb) |
| 2 | Feb 18 | **Finance Theory for AI Applications.** Risk-return tradeoff, CAPM, beta, options payoffs, portfolio diversification. | [Slides](slides/BUS_696_class02_Finance-Theory-for-AI-Applications.pdf) / [Lab](labs/Week_02_Lab_Finance_Theory.ipynb) |
| 3 | Feb 25 | **Data-Driven Finance: Can AI Beat the Market?** EMH, feature engineering, ML model horse race, overfitting, walk-forward validation. | [Slides](slides/BUS_696_class03_Data-Driven-Finance-Can-AI-Beat-the-Market_with_figures_v2.pdf) / [Lab](labs/Week_03_Lab_Data_Driven_Finance.ipynb) / [Practice Lab](labs/Week_03_Practice_Lab_Market_Analysis.ipynb) |

### Part II: AI Techniques in Finance

| Week | Date | Topic |
|------|------|-------|
| 4 | Mar 4 | **AI-First Finance.** Supervised vs. unsupervised learning. Neural networks: a conceptual overview. |
| 5 | Mar 11 | **Dense Neural Networks for Market Prediction.** Deep learning, feature engineering, and the overfitting problem. |
| 6 | Mar 18 | **Reinforcement Learning in Trading.** Agents, actions, rewards. Learning to trade from experience. |
| — | Mar 25 | *Spring Break* |
| 7 | Apr 1 | **Backtesting.** Look-ahead bias, survivorship bias, transaction costs, and market impact. |
| 8 | Apr 8 | **Risk Management with AI.** VaR, expected shortfall, model risk. |

### Part III: Applications and the Future

| Week | Date | Topic |
|------|------|-------|
| 9 | Apr 15 | **Algorithmic Trading and Market Making.** HFT, liquidity provision, the Flash Crash. |
| 10 | Apr 22 | **Bitcoin, Crypto, and DeFi.** Blockchain fundamentals, crypto volatility, decentralized finance. |
| 11 | Apr 29 | **AI Agents, Robo-Advisors, and Consumer Finance.** The future of financial advice. |
| 12 | May 6 | **Alternative Data and Market Intelligence.** Satellite imagery, sentiment analysis, news analytics. |
| 13 | May 13 | **Final Presentations.** |

## Labs

The labs are interactive Jupyter notebooks that let you work with real market data using Python. No prior programming experience is assumed — the notebooks are designed to be run cell-by-cell with explanations at every step.

| Lab | What You'll Do |
|-----|---------------|
| [Week 1: Finance Fundamentals](labs/Week_01_Lab_Finance_Fundamentals.ipynb) | Download stock data, build candlestick charts, calculate returns, compute moving averages and the Sharpe ratio |
| [Week 2: Finance Theory](labs/Week_02_Lab_Finance_Theory.ipynb) | Test whether returns are normally distributed, run CAPM regressions to estimate beta, plot option payoffs, visualize portfolio diversification |
| [Week 3: Data-Driven Finance](labs/Week_03_Lab_Data_Driven_Finance.ipynb) | Test the random walk hypothesis, engineer features (SMA, RSI, momentum), train ML models (logistic regression, Lasso, random forest, XGBoost), see overfitting in action |
| [Week 3: Market Analysis Practice](labs/Week_03_Practice_Lab_Market_Analysis.ipynb) | Conduct an event study on the Trump tariff reversal, trade the news with puts and calls, compare RSI and PE ratios across tech vs. consumer staples |

## Setup

Labs require Python 3 with the following packages:

```bash
pip install yfinance pandas numpy matplotlib plotly scipy statsmodels scikit-learn xgboost
```

## Textbook

Yves Hilpisch, [*Artificial Intelligence in Finance*](https://www.oreilly.com/library/view/artificial-intelligence-in/9781492055426/) (O'Reilly, 2020)

## Use of AI Tools

Students are encouraged to use AI tools (ChatGPT, Claude, Copilot) as learning aids — for brainstorming, debugging, and improving clarity of explanation. However, you must understand and be able to explain any work you submit.

## Instructor

**Jonathan Hersh, Ph.D.**
Associate Professor, Economics and Management Science
Argyros College of Business and Economics, Chapman University
[jonathanhersh.com](https://jonathanhersh.com)

## Syllabus

The full syllabus is available [here](syllabus/BUS_696_GenAI_Finance_Syllabus.pdf).
