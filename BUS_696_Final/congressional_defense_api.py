"""
Congressional Defense API Module
=================================
Tracks defense-related legislation and correlates with defense stock performance.

Purpose:
  1. Pull NDAA bills, appropriations, and amendments from Congress.gov
  2. Extract dates when defense spending laws pass
  3. Correlate law passage dates with defense stock returns
  4. Identify which laws drive strongest stock reactions
  
Author: Manuel Lara | Chapman University | BUS 696
Date: May 6, 2026
"""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────────────────
# PART 1: CONGRESS.GOV API CLIENT
# ────────────────────────────────────────────────────────────────────────────

class CongressionalDefenseAPI:
    """
    Fetch defense legislation from Congress.gov API.
    
    API: https://api.congress.gov/v3/
    Documentation: https://github.com/LibraryOfCongress/api.congress.gov
    """
    
    BASE_URL = 'https://api.congress.gov/v3'
    
    # Search terms for defense-related bills
    DEFENSE_KEYWORDS = [
        'NDAA',                          # National Defense Authorization Act
        'defense authorization',
        'defense appropriation',
        'military spending',
        'armed forces',
        'Department of Defense',
        'F-35',                          # Key weapon system
        'hypersonic',                    # Emerging tech
        'cyber defense',
        'space force',
        'military construction',
        'military pay',
        'military equipment',
        'defense budget',
    ]
    
    # Defense contractors to track
    DEFENSE_CONTRACTORS = {
        'LMT': 'Lockheed Martin',
        'RTX': 'Raytheon Technologies',
        'BA': 'Boeing',
        'NOC': 'Northrop Grumman',
        'GD': 'General Dynamics',
        'TXT': 'Textron',
        'HII': 'Huntington Ingalls',
        'CACI': 'CACI International',
    }
    
    def __init__(self, api_key: str = None):
        """
        Initialize Congress API client.
        
        Args:
            api_key: Optional API key (Congress.gov doesn't require one, but good practice)
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.bills_cache = {}
        
    def search_bills(self, query: str, congress: int = None, limit: int = 100) -> List[Dict]:
        """
        Search for bills matching query.
        
        Args:
            query: Search term (e.g., "NDAA", "defense appropriation")
            congress: Congress number (current is 118, previous 117, 116, etc.)
            limit: Max results to return
            
        Returns:
            List of bill dictionaries with metadata
            
        Example:
            >>> api = CongressionalDefenseAPI()
            >>> bills = api.search_bills('NDAA defense authorization', congress=118)
            >>> len(bills)  # Number of matching bills
        """
        
        if congress is None:
            # Get current congress (as of May 2026, Congress 119 or 120)
            congress = 119
        
        # Build query parameters
        params = {
            'q': query,
            'limit': limit,
            'format': 'json'
        }
        
        # Add API key if available
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = self.session.get(
                f'{self.BASE_URL}/bill/{congress}',
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            bills = data.get('bills', [])
            
            print(f"✓ Found {len(bills)} bills matching '{query}'")
            return bills
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching bills: {e}")
            return []
    
    def get_ndaa_history(self, start_year: int = 2000, end_year: int = 2026) -> pd.DataFrame:
        """
        Fetch historical NDAA bills and passage dates (Federal Laws Only).
        
        Args:
            start_year: Start year for NDAA search
            end_year: End year for NDAA search
            
        Returns:
            DataFrame with NDAA bills and dates
            
        Columns:
            - congress: Congress number
            - decade: 10-year increment (2000-2009, 2010-2019, 2020-2029)
            - bill_number: Bill ID (e.g., S.4711)
            - introduced_date: Date bill introduced
            - passed_date: Date bill passed (or None if pending)
            - enacted_date: Date signed into law (FEDERAL ONLY)
            - title: Full bill title
            - url: Bill URL on Congress.gov
            - is_federal: Boolean (always True for NDAA - these are federal)
            
        NOTE: NDAA = National Defense Authorization Act (FEDERAL LAW)
              Congress.gov only tracks federal legislation
        """
        
        ndaa_bills = []
        
        print("\n" + "="*70)
        print(f"Fetching NDAA History ({start_year}-{end_year})")
        print("Federal Laws Only - NDAA (National Defense Authorization Act)")
        print("="*70)
        
        # Search for NDAA bills in each congress
        # 118th Congress: 2023-2024 (FY2024-2025)
        # 117th Congress: 2021-2022 (FY2022-2023)
        # 116th Congress: 2019-2020 (FY2020-2021)
        # etc.
        
        for congress in range(106, 120):  # 106th (2000) to 119th (2026)
            bills = self.search_bills('NDAA national defense authorization', congress=congress, limit=50)
            
            for bill in bills:
                try:
                    bill_number = bill.get('number', 'N/A')
                    title = bill.get('title', '')
                    url = bill.get('url', '')
                    
                    # Get detailed bill info (includes dates)
                    details = self._get_bill_details(congress, bill_number)
                    
                    if details:
                        # Calculate decade for 10-year grouping
                        enacted_date = pd.to_datetime(details.get('enacted_date'), errors='coerce')
                        if pd.notna(enacted_date):
                            decade_start = (int(enacted_date.year) // 10) * 10
                            decade = f"{decade_start}-{decade_start + 9}"
                        else:
                            decade = "Unknown"
                        
                        ndaa_bills.append({
                            'congress': congress,
                            'decade': decade,
                            'bill_number': bill_number,
                            'title': title,
                            'introduced_date': details.get('introduced_date'),
                            'passed_date': details.get('passed_date'),
                            'enacted_date': details.get('enacted_date'),
                            'status': details.get('status'),
                            'url': url,
                            'is_federal': True,  # NDAA is always federal
                            'law_type': 'Federal - NDAA (National Defense Authorization Act)',
                        })
                        
                        print(f"  • {congress}: {bill_number} [{decade}] - {title[:45]}...")
                
                except Exception as e:
                    print(f"  ⚠️  Error processing bill: {e}")
                    continue
        
        df = pd.DataFrame(ndaa_bills)
        
        # Convert date strings to datetime
        for col in ['introduced_date', 'passed_date', 'enacted_date']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        print(f"\n✓ Retrieved {len(df)} Federal NDAA bills")
        return df
    
    def get_ndaa_by_decade(self, ndaa_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group NDAA bills by 10-year increments.
        
        Args:
            ndaa_df: DataFrame of NDAA bills from get_ndaa_history()
            
        Returns:
            Dictionary {decade: DataFrame of bills for that decade}
            
        Example:
            >>> decades = api.get_ndaa_by_decade(ndaa_df)
            >>> decades['2010-2019']  # All NDAAs from 2010-2019
            >>> decades['2020-2029']  # All NDAAs from 2020-2029
        """
        
        decades_dict = {}
        
        # Group by decade
        for decade in ndaa_df['decade'].unique():
            if decade != 'Unknown':
                decade_data = ndaa_df[ndaa_df['decade'] == decade].copy()
                decade_data = decade_data.sort_values('enacted_date')
                decades_dict[decade] = decade_data
        
        return decades_dict
    
    def print_ndaa_by_decade(self, ndaa_df: pd.DataFrame) -> None:
        """
        Pretty-print NDAA bills grouped by 10-year increments.
        
        Args:
            ndaa_df: DataFrame of NDAA bills
        """
        
        decades = self.get_ndaa_by_decade(ndaa_df)
        
        print("\n" + "="*80)
        print("FEDERAL NDAA BILLS GROUPED BY 10-YEAR INCREMENTS")
        print("="*80)
        
        for decade in sorted(decades.keys()):
            df = decades[decade]
            print(f"\n📅 DECADE: {decade}")
            print(f"   Total NDAA bills: {len(df)}")
            print(f"   Federal law type: NDAA (National Defense Authorization Act)")
            print("   " + "-"*76)
            
            for idx, row in df.iterrows():
                enacted = row['enacted_date'].strftime('%Y-%m-%d') if pd.notna(row['enacted_date']) else 'N/A'
                print(f"   • {row['bill_number']:8s} | Enacted: {enacted} | {row['title'][:55]}")
        
        print("\n" + "="*80)
    
    def _get_bill_details(self, congress: int, bill_number: str) -> Dict:
        """
        Get detailed information about a specific bill.
        
        Args:
            congress: Congress number
            bill_number: Bill number (e.g., "4711" or "s4711")
            
        Returns:
            Dictionary with bill details
        """
        
        try:
            # Remove 's' or 'h' prefix if present
            bill_num = bill_number.replace('s', '').replace('h', '').replace('S', '').replace('H', '')
            
            response = self.session.get(
                f'{self.BASE_URL}/bill/{congress}/s/{bill_num}',
                params={'format': 'json'},
                timeout=10
            )
            
            if response.status_code != 200:
                # Try House bill
                response = self.session.get(
                    f'{self.BASE_URL}/bill/{congress}/h/{bill_num}',
                    params={'format': 'json'},
                    timeout=10
                )
            
            if response.status_code == 200:
                bill = response.json().get('bill', {})
                
                # Parse dates from action history
                actions = bill.get('actions', [])
                
                return {
                    'status': bill.get('latestAction', {}).get('text', 'Introduced'),
                    'introduced_date': bill.get('introducedDate'),
                    'passed_date': self._extract_passed_date(actions),
                    'enacted_date': self._extract_enacted_date(actions),
                }
            
            return None
            
        except Exception as e:
            print(f"Error getting bill details: {e}")
            return None
    
    @staticmethod
    def _extract_passed_date(actions: List[Dict]) -> str:
        """Extract date when bill passed from action history."""
        for action in actions:
            if 'passed' in action.get('text', '').lower():
                return action.get('actionDate')
        return None
    
    @staticmethod
    def _extract_enacted_date(actions: List[Dict]) -> str:
        """Extract date when bill was enacted (signed) from action history."""
        for action in actions:
            text = action.get('text', '').lower()
            if 'signed' in text or 'enacted' in text:
                return action.get('actionDate')
        return None
    
    def search_defense_appropriations(self, year: int = 2024, congress: int = None) -> List[Dict]:
        """
        Search for defense appropriation bills for a given year.
        
        Args:
            year: Fiscal year
            congress: Congress number (defaults to current)
            
        Returns:
            List of appropriation bills
        """
        
        if congress is None:
            congress = 119
        
        bills = self.search_bills(
            'defense appropriation military spending', 
            congress=congress,
            limit=50
        )
        
        return bills
    
    def get_bill_amendments(self, congress: int, bill_number: str) -> List[Dict]:
        """
        Get amendments to a bill (e.g., NDAA).
        
        Amendments can indicate specific contractors receiving funding.
        Example: Amendment to add $5B to F-35 program → LMT positive
        
        Args:
            congress: Congress number
            bill_number: Bill number
            
        Returns:
            List of amendments
        """
        
        try:
            # Note: Congress.gov API may have limited amendment support
            # Alternative: Parse HTML from Congress.gov website
            
            response = self.session.get(
                f'{self.BASE_URL}/bill/{congress}/amendments',
                params={'format': 'json'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                amendments = data.get('amendments', [])
                
                print(f"✓ Found {len(amendments)} amendments")
                return amendments
            
            return []
            
        except Exception as e:
            print(f"Error fetching amendments: {e}")
            return []


# ────────────────────────────────────────────────────────────────────────────
# PART 2: DEFENSE STOCK DATA FETCHER
# ────────────────────────────────────────────────────────────────────────────

class DefenseStockData:
    """Fetch historical price data for defense contractors."""
    
    DEFENSE_TICKERS = {
        'LMT': 'Lockheed Martin',
        'RTX': 'Raytheon Technologies',
        'BA': 'Boeing',
        'NOC': 'Northrop Grumman',
        'GD': 'General Dynamics',
        'TXT': 'Textron',
        'HII': 'Huntington Ingalls',
        'CACI': 'CACI International',
    }
    
    def __init__(self):
        self.prices = {}
    
    def fetch_prices(self, start_date: str = '2015-01-01', end_date: str = '2026-05-06') -> Dict[str, pd.DataFrame]:
        """
        Fetch historical prices for all defense contractors.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary {ticker: DataFrame with prices}
        """
        
        print("\n" + "="*70)
        print("Fetching Defense Stock Prices")
        print("="*70)
        
        for ticker, name in self.DEFENSE_TICKERS.items():
            try:
                print(f"  • {ticker} ({name})...", end=" ")
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                self.prices[ticker] = data
                print(f"✓ {len(data)} days")
            except Exception as e:
                print(f"❌ {e}")
        
        return self.prices
    
    def calculate_returns(self, window_days: int = 30) -> Dict[str, pd.Series]:
        """
        Calculate forward returns for each stock.
        
        Args:
            window_days: Days forward to look (e.g., 30 = 1-month forward return)
            
        Returns:
            Dictionary {ticker: Series of forward returns}
        """
        
        returns = {}
        
        for ticker, prices in self.prices.items():
            # Calculate forward returns
            forward_returns = prices['Close'].pct_change(periods=window_days).shift(-window_days)
            returns[ticker] = forward_returns
        
        return returns


# ────────────────────────────────────────────────────────────────────────────
# PART 3: CORRELATION ANALYSIS
# ────────────────────────────────────────────────────────────────────────────

class LawStockCorrelationAnalysis:
    """Correlate defense legislation events with stock performance."""
    
    def __init__(self, ndaa_df: pd.DataFrame, stock_prices: Dict[str, pd.DataFrame]):
        """
        Initialize correlation analysis.
        
        Args:
            ndaa_df: DataFrame of NDAA bills with dates
            stock_prices: Dictionary of stock price data
        """
        self.ndaa_df = ndaa_df
        self.stock_prices = stock_prices
        self.correlations = {}
    
    def create_legislation_events(self) -> pd.DataFrame:
        """
        Create binary events for each legislation date.
        
        Returns:
            DataFrame with date index and binary indicators for each law event
            
        Interpretation:
            1 = Bill passed on this date
            0 = No legislation event
        """
        
        print("\n" + "="*70)
        print("Creating Legislation Events Timeline")
        print("="*70)
        
        # Create date range covering all data
        min_date = self.ndaa_df['enacted_date'].min()
        max_date = self.ndaa_df['enacted_date'].max()
        
        # Get min date from stock prices
        for prices in self.stock_prices.values():
            if len(prices) > 0:
                max_date = max(max_date, prices.index.max())
        
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        events_df = pd.DataFrame(index=date_range)
        
        # Add binary event indicators
        events_df['ndaa_passed'] = 0
        events_df['defense_appropriation'] = 0
        
        # Mark NDAA passage dates
        for _, row in self.ndaa_df.iterrows():
            if pd.notna(row['enacted_date']):
                events_df.loc[row['enacted_date'], 'ndaa_passed'] = 1
                print(f"  • NDAA passed: {row['enacted_date'].date()}")
        
        return events_df
    
    def analyze_stock_reaction(self, 
                              event_dates: pd.DatetimeIndex,
                              ticker: str,
                              window_before: int = 5,
                              window_after: int = 30) -> Dict:
        """
        Analyze stock reaction to legislation events.
        
        Args:
            event_dates: Dates when legislation passed
            ticker: Stock ticker
            window_before: Days before event (for baseline)
            window_after: Days after event (for reaction)
            
        Returns:
            Dictionary with stats on stock reaction
            
        Interpretation:
            If stock rises 5%+ in 30 days after law passage → positive correlation
        """
        
        if ticker not in self.stock_prices:
            return None
        
        prices = self.stock_prices[ticker]['Close']
        reactions = []
        
        print(f"\n  Analyzing {ticker} reaction to {len(event_dates)} legislation events:")
        
        for event_date in event_dates:
            if event_date not in prices.index:
                # Find closest date
                closest_idx = prices.index.searchsorted(event_date)
                if closest_idx >= len(prices):
                    continue
                event_date = prices.index[closest_idx]
            
            try:
                event_idx = prices.index.get_loc(event_date)
                
                # Get prices before and after
                if event_idx > window_before and event_idx + window_after < len(prices):
                    price_before = prices.iloc[event_idx - window_before]
                    price_after = prices.iloc[event_idx + window_after]
                    
                    # Calculate return
                    reaction_return = (price_after - price_before) / price_before
                    reactions.append(reaction_return)
                    
                    print(f"    • {event_date.date()}: {reaction_return:+.2%} ({window_after}d)")
            
            except Exception as e:
                pass
        
        if len(reactions) == 0:
            return None
        
        reactions = np.array(reactions)
        
        return {
            'ticker': ticker,
            'num_events': len(reactions),
            'mean_reaction': reactions.mean(),
            'std_reaction': reactions.std(),
            'median_reaction': np.median(reactions),
            'positive_reactions': (reactions > 0).sum(),
            'win_rate': (reactions > 0).sum() / len(reactions),
            'avg_positive_return': reactions[reactions > 0].mean() if (reactions > 0).any() else 0,
            'avg_negative_return': reactions[reactions < 0].mean() if (reactions < 0).any() else 0,
        }
    
    def correlate_all_stocks(self, event_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation between legislation events and stock returns.
        
        Args:
            event_df: DataFrame of legislation events
            
        Returns:
            DataFrame with correlation results for each stock
        """
        
        print("\n" + "="*70)
        print("Calculating Law-Stock Correlations")
        print("="*70)
        
        results = []
        
        for ticker in self.stock_prices.keys():
            prices = self.stock_prices[ticker]['Close']
            
            # Calculate returns
            returns = prices.pct_change().fillna(0)
            
            # Align dates
            common_dates = event_df.index.intersection(returns.index)
            
            if len(common_dates) < 10:
                continue
            
            event_aligned = event_df.loc[common_dates, 'ndaa_passed']
            returns_aligned = returns.loc[common_dates]
            
            # Calculate correlation
            if (event_aligned.std() > 0) and (returns_aligned.std() > 0):
                spearman_corr, spearman_p = spearmanr(event_aligned, returns_aligned)
                pearson_corr, pearson_p = pearsonr(event_aligned, returns_aligned)
            else:
                spearman_corr = spearman_p = pearson_corr = pearson_p = 0
            
            results.append({
                'ticker': ticker,
                'company': CongressionalDefenseAPI.DEFENSE_CONTRACTORS.get(ticker, ''),
                'spearman_corr': spearman_corr,
                'spearman_pval': spearman_p,
                'pearson_corr': pearson_corr,
                'pearson_pval': pearson_p,
                'significant': (spearman_p < 0.05) or (pearson_p < 0.05),
            })
            
            print(f"  • {ticker}: Spearman ρ={spearman_corr:+.4f} (p={spearman_p:.4f})")
        
        return pd.DataFrame(results)


# ────────────────────────────────────────────────────────────────────────────
# PART 4: MAIN EXECUTION
# ────────────────────────────────────────────────────────────────────────────

def main():
    """
    Main execution: Fetch laws, correlate with stocks.
    
    Output:
        1. Print NDAA history grouped by 10-year increments (FEDERAL LAWS ONLY)
        2. Fetch defense stock prices
        3. Calculate correlation: Law passage → Stock returns
        4. Identify which laws have strongest stock reactions
    """
    
    print("\n" + "="*80)
    print("CONGRESSIONAL DEFENSE API: Federal Law-Stock Correlation Analysis")
    print("Federal Laws: NDAA (National Defense Authorization Act)")
    print("="*80)
    
    # Step 1: Initialize API
    print("\n[Step 1] Initializing Congress.gov API (Federal Laws Only)...")
    congress_api = CongressionalDefenseAPI()
    
    # Step 2: Fetch NDAA history grouped by decade
    print("\n[Step 2] Fetching Federal NDAA Bills (2000-2026) - 10-Year Increments...")
    ndaa_df = congress_api.get_ndaa_history(start_year=2000, end_year=2026)
    
    if len(ndaa_df) > 0:
        # Print organized by decade
        congress_api.print_ndaa_by_decade(ndaa_df)
        
        # Get decade breakdown
        decades = congress_api.get_ndaa_by_decade(ndaa_df)
        
        print("\n" + "="*80)
        print("FEDERAL NDAA SUMMARY BY DECADE")
        print("="*80)
        for decade in sorted(decades.keys()):
            bills_in_decade = decades[decade]
            enacted_dates = bills_in_decade['enacted_date'].dropna()
            if len(enacted_dates) > 0:
                first_law = enacted_dates.min().strftime('%Y-%m-%d')
                last_law = enacted_dates.max().strftime('%Y-%m-%d')
                print(f"\n{decade}: {len(bills_in_decade)} Federal NDAA Laws")
                print(f"  • First enacted: {first_law}")
                print(f"  • Last enacted:  {last_law}")
                print(f"  • Law type: Federal - NDAA (National Defense Authorization Act)")
    
    # Step 3: Fetch defense stock prices
    print("\n[Step 3] Fetching Defense Stock Prices (2015-2026)...")
    stock_data = DefenseStockData()
    prices = stock_data.fetch_prices(start_date='2015-01-01', end_date='2026-05-06')
    
    # Step 4: Analyze correlations
    print("\n[Step 4] Analyzing Federal Law-Stock Correlations...")
    correlation_analysis = LawStockCorrelationAnalysis(ndaa_df, prices)
    
    # Create legislation events timeline
    events_df = correlation_analysis.create_legislation_events()
    
    # Calculate correlations for all stocks
    corr_results = correlation_analysis.correlate_all_stocks(events_df)
    
    if len(corr_results) > 0:
        print("\n" + "="*80)
        print("CORRELATION RESULTS: Federal NDAA Laws → Defense Stock Returns")
        print("="*80)
        print(corr_results.to_string(index=False))
        
        # Identify significant correlations
        significant = corr_results[corr_results['significant']]
        if len(significant) > 0:
            print("\n✓ STATISTICALLY SIGNIFICANT CORRELATIONS (Federal Laws, p < 0.05):")
            print(significant.to_string(index=False))
        else:
            print("\n⚠️  No statistically significant correlations found (p < 0.05)")
    
    # Step 5: Analyze stock reactions to specific events (by decade)
    print("\n[Step 5] Stock Reactions to Federal NDAA Passage by Decade...")
    
    decades = congress_api.get_ndaa_by_decade(ndaa_df)
    
    for decade in sorted(decades.keys()):
        decade_bills = decades[decade]
        enacted_dates = decade_bills[decade_bills['enacted_date'].notna()]['enacted_date']
        
        if len(enacted_dates) == 0:
            continue
        
        print(f"\n{'='*80}")
        print(f"DECADE: {decade} - Federal NDAA Laws")
        print(f"{'='*80}")
        print(f"Federal laws enacted: {len(enacted_dates)}")
        
        for ticker in DefenseStockData.DEFENSE_TICKERS.keys():
            reaction = correlation_analysis.analyze_stock_reaction(
                enacted_dates,
                ticker,
                window_after=30  # 30 days after
            )
            
            if reaction and reaction['num_events'] > 0:
                print(f"\n{ticker}:")
                print(f"  • Federal laws in decade: {reaction['num_events']}")
                print(f"  • Average 30-day return: {reaction['mean_reaction']:+.2%}")
                print(f"  • Win rate: {reaction['win_rate']:.1%}")
                print(f"  • Avg positive: {reaction['avg_positive_return']:+.2%}")
                print(f"  • Avg negative: {reaction['avg_negative_return']:+.2%}")
    
    print("\n" + "="*80)
    print("Analysis Complete - Federal Laws Only (NDAA)")
    print("="*80)
    
    return {
        'ndaa_bills': ndaa_df,
        'stock_prices': prices,
        'correlation_results': corr_results,
        'decades': congress_api.get_ndaa_by_decade(ndaa_df),
    }


# ────────────────────────────────────────────────────────────────────────────
# EXECUTION
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = main()
    
    # Save results to CSV for further analysis
    if len(results['correlation_results']) > 0:
        results['correlation_results'].to_csv(
            'defense_law_stock_correlation.csv',
            index=False
        )
        print("\n✓ Results saved to defense_law_stock_correlation.csv")
