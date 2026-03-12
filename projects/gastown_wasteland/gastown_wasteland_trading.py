"""
Gas Town & The Wasteland — A Working Multi-Agent Trading System
===============================================================
BUS 696: Generative AI in Finance | Professor Jonathan Hersh

This script demonstrates Steve Yegge's Gas Town / Wasteland concepts
using a simple multi-agent trading system built with tools from this class.

GAS TOWN CONCEPTS:
  - Town     = this trading system workspace
  - Rigs     = components (regime detection, momentum, mean-reversion, volatility)
  - Polecats = worker "agents" — each one builds a trading signal independently
  - Mayor    = the integrator that combines signals using trust-weighted averaging

WASTELAND CONCEPTS:
  - Wanted Board  = the list of signals we need built
  - Stamps        = reputation earned by each agent based on out-of-sample accuracy
  - Trust Levels  = how much weight each agent gets (earned, not assigned)
  - Kelly Link    = trust allocation mirrors Kelly sizing: don't over-trust!

Run: python gastown_wasteland_trading.py
Requires: numpy, pandas, matplotlib, yfinance, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Course color palette
NAVY = '#1E2761'
CORAL = '#F96167'
TEAL = '#028090'
GOLD = '#F9A825'
GRAY = '#6c757d'

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================
# PART 1: DATA (shared resource — all Polecats use the same data)
# ============================================================

def download_data():
    """Download SPY data and engineer features."""
    print("=" * 60)
    print("TOWN: 'Multi-Agent Trading System'")
    print("=" * 60)
    print("\nDownloading SPY data...")

    spy = yf.download('SPY', start='2010-01-01', progress=False)
    spy.columns = spy.columns.get_level_values(0)
    spy['Return'] = spy['Close'].pct_change()
    spy['Log_Return'] = np.log(spy['Close'] / spy['Close'].shift(1))
    spy = spy.dropna()

    print(f"  {len(spy)} trading days: {spy.index[0].date()} to {spy.index[-1].date()}")
    return spy


def engineer_features(spy):
    """Build the feature set that all agents share."""
    df = pd.DataFrame(index=spy.index)

    # Lagged returns (what happened recently?)
    for lag in range(1, 6):
        df[f'ret_lag{lag}'] = spy['Log_Return'].shift(lag)

    # Moving averages (trend signals)
    df['sma_5'] = spy['Close'].rolling(5).mean() / spy['Close'] - 1
    df['sma_20'] = spy['Close'].rolling(20).mean() / spy['Close'] - 1
    df['sma_60'] = spy['Close'].rolling(60).mean() / spy['Close'] - 1

    # Volatility (risk signals)
    df['vol_10'] = spy['Log_Return'].rolling(10).std()
    df['vol_60'] = spy['Log_Return'].rolling(60).std()
    df['vol_ratio'] = df['vol_10'] / df['vol_60']  # vol regime indicator

    # Momentum
    df['mom_20'] = spy['Close'].pct_change(20)
    df['mom_60'] = spy['Close'].pct_change(60)

    # RSI (mean-reversion signal)
    delta = spy['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Target: will tomorrow's return be positive?
    df['target'] = (spy['Log_Return'].shift(-1) > 0).astype(int)
    df['next_return'] = spy['Log_Return'].shift(-1)

    df = df.dropna()
    return df


# ============================================================
# PART 2: THE POLECATS — Independent Worker Agents
# ============================================================
# Each Polecat builds one trading signal. In a real Gas Town,
# these would be separate Claude Code instances running in parallel.
# Here we simulate them as functions that each train a different model.

class Polecat:
    """A worker agent that builds and maintains a trading signal."""

    def __init__(self, name, feature_subset, model):
        self.name = name
        self.feature_subset = feature_subset
        self.model = model
        self.scaler = StandardScaler()
        self.stamps = []         # Wasteland reputation history
        self.trust_level = 1     # L1=registered, L2=contributor, L3=maintainer

    def train(self, X_train, y_train):
        """Train this agent's model on its feature subset."""
        X = self.scaler.fit_transform(X_train[self.feature_subset])
        self.model.fit(X, y_train)

    def predict(self, X_test):
        """Generate trading signal: probability that market goes up."""
        X = self.scaler.transform(X_test[self.feature_subset])
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        """Evaluate and earn stamps (Wasteland reputation)."""
        X = self.scaler.transform(X_test[self.feature_subset])
        preds = self.model.predict(X)
        acc = accuracy_score(y_test, preds)

        # Award stamps based on quality (1-5 scale)
        if acc >= 0.55:
            quality = 5
        elif acc >= 0.53:
            quality = 4
        elif acc >= 0.51:
            quality = 3
        elif acc >= 0.49:
            quality = 2
        else:
            quality = 1

        self.stamps.append(quality)

        # Trust level promotion (Wasteland L1 -> L2 -> L3)
        avg_quality = np.mean(self.stamps[-5:])  # last 5 evaluations
        if len(self.stamps) >= 5 and avg_quality >= 4.0:
            self.trust_level = 3  # maintainer
        elif len(self.stamps) >= 3 and avg_quality >= 3.0:
            self.trust_level = 2  # contributor

        return acc, quality

    def __repr__(self):
        avg = np.mean(self.stamps) if self.stamps else 0
        return f"Polecat('{self.name}', L{self.trust_level}, stamps={len(self.stamps)}, avg_quality={avg:.1f})"


def create_polecats():
    """
    WANTED BOARD: Create the worker agents.
    Each one claims a different task on the board.
    """
    polecats = [
        # Polecat 1: Momentum specialist — uses trend features
        Polecat(
            name="Momentum Agent",
            feature_subset=['ret_lag1', 'ret_lag2', 'ret_lag3', 'sma_5', 'sma_20', 'mom_20', 'mom_60'],
            model=LogisticRegression(max_iter=1000, random_state=42)
        ),

        # Polecat 2: Mean-reversion specialist — uses RSI and short-term features
        Polecat(
            name="Mean-Reversion Agent",
            feature_subset=['rsi', 'ret_lag1', 'ret_lag2', 'sma_5', 'vol_10'],
            model=LogisticRegression(max_iter=1000, random_state=42)
        ),

        # Polecat 3: Volatility regime specialist — uses vol features
        Polecat(
            name="Volatility Regime Agent",
            feature_subset=['vol_10', 'vol_60', 'vol_ratio', 'ret_lag1', 'ret_lag2', 'sma_60'],
            model=RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        ),

        # Polecat 4: Kitchen-sink agent — uses everything
        Polecat(
            name="Full-Feature Agent",
            feature_subset=['ret_lag1', 'ret_lag2', 'ret_lag3', 'ret_lag4', 'ret_lag5',
                            'sma_5', 'sma_20', 'sma_60', 'vol_10', 'vol_60',
                            'vol_ratio', 'mom_20', 'mom_60', 'rsi'],
            model=RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        ),
    ]

    return polecats


# ============================================================
# PART 3: THE MAYOR — Supervisor Agent
# ============================================================
# The Mayor combines Polecat signals using trust-weighted averaging.
# This mirrors Kelly: agents with more proven edge get more weight.

class Mayor:
    """
    The supervisor agent that integrates Polecat signals.
    Uses trust-weighted averaging — the Wasteland equivalent
    of Kelly position sizing.
    """

    def __init__(self, polecats):
        self.polecats = polecats

    def get_trust_weights(self):
        """
        Convert Wasteland trust/stamps into portfolio weights.
        This IS the Kelly parallel: don't over-weight an unproven agent.
        """
        weights = {}
        for p in self.polecats:
            if not p.stamps:
                # New agent, no track record — minimal trust (like L1 in Wasteland)
                weights[p.name] = 0.1
            else:
                # Weight = average stamp quality * trust level multiplier
                avg_quality = np.mean(p.stamps[-5:])  # recent performance
                trust_mult = {1: 0.5, 2: 1.0, 3: 1.5}[p.trust_level]
                weights[p.name] = avg_quality * trust_mult

        # Normalize to sum to 1
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        return weights

    def combine_signals(self, signals_dict):
        """
        Mayor combines all Polecat signals into one trading decision.
        Weighted average of predicted probabilities.
        """
        weights = self.get_trust_weights()
        combined = np.zeros(len(next(iter(signals_dict.values()))))

        for name, signal in signals_dict.items():
            combined += weights[name] * signal

        return combined

    def print_trust_report(self):
        """Display the Wasteland trust network status."""
        weights = self.get_trust_weights()
        print("\n  WASTELAND TRUST NETWORK:")
        print("  " + "-" * 55)
        for p in self.polecats:
            level_name = {1: 'Registered', 2: 'Contributor', 3: 'Maintainer'}[p.trust_level]
            avg_q = np.mean(p.stamps[-5:]) if p.stamps else 0
            print(f"  L{p.trust_level} {level_name:12s} | {p.name:25s} | "
                  f"weight={weights[p.name]:.1%} | avg_stamp={avg_q:.1f}")


# ============================================================
# PART 4: WALK-FORWARD BACKTEST (the integration rig)
# ============================================================
# This is Rig 4 from the Gas Town architecture: wire everything
# together and test with proper walk-forward validation.

def run_backtest(df, polecats, train_years=3):
    """
    Walk-forward backtest: retrain annually with expanding window.
    This is the discipline that prevents overfitting — Lopez de Prado approved.
    """
    mayor = Mayor(polecats)
    feature_cols = [c for c in df.columns if c not in ['target', 'next_return']]
    years = sorted(df.index.year.unique())

    all_results = []

    print("\n" + "=" * 60)
    print("RIG 4: Walk-Forward Integration Backtest")
    print("=" * 60)

    for i, test_year in enumerate(years[train_years:]):
        train_start = years[0]
        train_end = test_year - 1

        train = df[df.index.year.isin(range(train_start, train_end + 1))]
        test = df[df.index.year == test_year]

        if len(test) == 0:
            continue

        # Each Polecat trains independently (parallel in Gas Town)
        print(f"\n  Year {test_year}: Training on {train_start}-{train_end} "
              f"({len(train)} days), testing on {test_year} ({len(test)} days)")

        signals = {}
        for p in polecats:
            p.train(train[feature_cols], train['target'])
            prob = p.predict(test[feature_cols])
            signals[p.name] = prob

            # Evaluate and award stamps
            acc, quality = p.evaluate(test[feature_cols], test['target'])
            print(f"    Polecat '{p.name}': acc={acc:.1%}, stamp={quality}/5, L{p.trust_level}")

        # Mayor combines signals using trust weights
        combined_prob = mayor.combine_signals(signals)
        combined_signal = np.where(combined_prob > 0.5, 1, -1)

        # Calculate returns
        test_results = pd.DataFrame({
            'date': test.index,
            'signal': combined_signal,
            'next_return': test['next_return'].values,
            'strategy_return': combined_signal * test['next_return'].values,
            'year': test_year,
        }).set_index('date')

        all_results.append(test_results)

        # Show trust network after this year
        mayor.print_trust_report()

    results = pd.concat(all_results)
    return results, mayor


# ============================================================
# PART 5: PERFORMANCE DASHBOARD
# ============================================================

def plot_results(results):
    """4-panel dashboard: cumulative returns, drawdown, agent weights, annual bars."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: Cumulative returns
    ax = axes[0, 0]
    cum_strat = (1 + results['strategy_return']).cumprod()
    cum_market = (1 + results['next_return']).cumprod()
    ax.plot(cum_strat.index, cum_strat, color=TEAL, linewidth=2, label='Gas Town System')
    ax.plot(cum_market.index, cum_market, color=GRAY, linewidth=1.5, label='Buy & Hold SPY')
    ax.set_title('Cumulative Returns: Gas Town Multi-Agent vs Buy & Hold',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylabel('Growth of $1')

    # Panel 2: Drawdown
    ax = axes[0, 1]
    rolling_max = cum_strat.cummax()
    drawdown = (cum_strat - rolling_max) / rolling_max * 100
    ax.fill_between(drawdown.index, drawdown, 0, color=CORAL, alpha=0.5)
    ax.set_title('Strategy Drawdown', fontweight='bold', fontsize=11)
    ax.set_ylabel('Drawdown (%)')

    # Panel 3: Annual returns comparison
    ax = axes[1, 0]
    annual = results.groupby('year').agg(
        strategy=('strategy_return', lambda x: (1 + x).prod() - 1),
        market=('next_return', lambda x: (1 + x).prod() - 1)
    )
    x = np.arange(len(annual))
    w = 0.35
    ax.bar(x - w/2, annual['strategy'] * 100, w, color=TEAL, label='Gas Town System')
    ax.bar(x + w/2, annual['market'] * 100, w, color=GRAY, label='Buy & Hold')
    ax.set_xticks(x)
    ax.set_xticklabels(annual.index, rotation=45)
    ax.set_ylabel('Annual Return (%)')
    ax.set_title('Annual Returns by Year', fontweight='bold', fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(0, color='black', linewidth=0.5)

    # Panel 4: Summary stats table
    ax = axes[1, 1]
    ax.axis('off')
    strat_ret = results['strategy_return']
    mkt_ret = results['next_return']

    stats = {
        'Total Return': f"{(cum_strat.iloc[-1] - 1):.1%} vs {(cum_market.iloc[-1] - 1):.1%}",
        'Ann. Return': f"{strat_ret.mean() * 252:.1%} vs {mkt_ret.mean() * 252:.1%}",
        'Ann. Volatility': f"{strat_ret.std() * np.sqrt(252):.1%} vs {mkt_ret.std() * np.sqrt(252):.1%}",
        'Sharpe Ratio': f"{strat_ret.mean() / strat_ret.std() * np.sqrt(252):.2f} vs "
                        f"{mkt_ret.mean() / mkt_ret.std() * np.sqrt(252):.2f}",
        'Max Drawdown': f"{drawdown.min():.1f}%",
        'Hit Rate': f"{(strat_ret > 0).mean():.1%}",
    }

    table_text = "METRIC                    GAS TOWN vs MARKET\n" + "-" * 50 + "\n"
    for k, v in stats.items():
        table_text += f"{k:25s} {v}\n"

    ax.text(0.05, 0.95, table_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle("Gas Town Multi-Agent Trading System — Performance Dashboard",
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('gastown_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nDashboard saved to gastown_results.png")


# ============================================================
# PART 6: THE KELLY-WASTELAND PARALLEL
# ============================================================

def demonstrate_kelly_trust_parallel(polecats):
    """
    Show the conceptual parallel between Kelly sizing and Wasteland trust.
    Both answer: "How much should I commit given uncertainty?"
    """
    print("\n" + "=" * 60)
    print("THE KELLY-WASTELAND PARALLEL")
    print("=" * 60)
    print("""
    KELLY CRITERION               WASTELAND TRUST
    ─────────────────              ─────────────────
    How much to BET?        ↔     How much to TRUST?
    Edge / Variance         ↔     Avg Stamps / Variability
    Full Kelly = aggressive ↔     L3 Maintainer = full trust
    Half Kelly = safe       ↔     L2 Contributor = partial trust
    Overbet → RUIN          ↔     Over-trust → SYSTEM FAILURE
    LTCM: "We're geniuses, ↔     "This agent passed 2 tests,
     lever up!"                     give it L3!"
    """)

    print("  AGENT TRUST AS KELLY FRACTIONS:")
    print("  " + "-" * 55)
    for p in polecats:
        if p.stamps:
            avg_q = np.mean(p.stamps)
            std_q = np.std(p.stamps) if len(p.stamps) > 1 else 1.0
            # Kelly-style: edge / variance
            kelly_trust = avg_q / (std_q + 1.0)  # +1 to avoid division by zero
            print(f"  {p.name:25s} | avg_stamp={avg_q:.2f} | "
                  f"stamp_vol={std_q:.2f} | kelly_trust={kelly_trust:.2f}")
        else:
            print(f"  {p.name:25s} | no track record yet")

    print("""
  KEY INSIGHT: Just like Kelly says "don't overbet a small edge,"
  the Wasteland says "don't over-trust an unproven agent."

  A Polecat with 2 high-quality stamps might look great, but the
  sample size is too small — just like a backtest with 10 trades.
  Trust, like position size, should scale with EVIDENCE.
    """)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Step 1: Download data (shared resource)
    spy = download_data()
    df = engineer_features(spy)
    print(f"  {len(df)} samples with {len([c for c in df.columns if c not in ['target','next_return']])} features\n")

    # Step 2: Create the Polecats (worker agents)
    print("WANTED BOARD — Claiming tasks:")
    polecats = create_polecats()
    for p in polecats:
        print(f"  ✓ '{p.name}' claimed its task")

    # Step 3: Run walk-forward backtest (Mayor integrates)
    results, mayor = run_backtest(df, polecats, train_years=3)

    # Step 4: Show final trust state
    print("\n" + "=" * 60)
    print("FINAL WASTELAND STATUS")
    print("=" * 60)
    for p in polecats:
        print(f"  {p}")

    # Step 5: Performance dashboard
    plot_results(results)

    # Step 6: Kelly-Wasteland parallel
    demonstrate_kelly_trust_parallel(polecats)

    print("\n" + "=" * 60)
    print("WHAT THIS DEMONSTRATES")
    print("=" * 60)
    print("""
  GAS TOWN in action:
    - 4 Polecats worked independently on different signal types
    - The Mayor combined them using trust-weighted averaging
    - Walk-forward validation prevented overfitting (Lopez de Prado)

  WASTELAND in action:
    - Each Polecat earned stamps based on out-of-sample accuracy
    - Trust levels promoted agents from L1 → L2 → L3 over time
    - The Mayor allocated weight like Kelly allocates capital:
      more evidence → more trust → more weight

  The same principles that prevent hedge fund blow-ups (Kelly)
  also prevent AI system failures (Wasteland trust).
    """)
