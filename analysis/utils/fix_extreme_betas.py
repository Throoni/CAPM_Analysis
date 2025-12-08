"""
fix_extreme_betas.py

Script to identify and fix extreme beta values caused by data errors.
Scans all stocks for extreme returns (>100% monthly) and fixes or filters problematic data points.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from analysis.utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, COUNTRIES

logger = logging.getLogger(__name__)


def scan_extreme_returns(threshold: float = 100.0) -> pd.DataFrame:
    """
    Scan all stock price files for extreme monthly returns.
    
    Parameters
    ----------
    threshold : float
        Threshold for extreme returns in percentage (default: 100%)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [country, ticker, date, price_before, price_after, return_pct, issue_type]
    """
    logger.info("="*70)
    logger.info("SCANNING FOR EXTREME RETURNS")
    logger.info("="*70)
    
    extreme_returns = []
    
    for country in COUNTRIES.keys():
        stock_file = os.path.join(DATA_RAW_DIR, f"prices_stocks_{country}.csv")
        
        if not os.path.exists(stock_file):
            continue
        
        logger.info(f"Scanning {country}...")
        prices = pd.read_csv(stock_file, index_col=0, parse_dates=True)
        
        for ticker in prices.columns:
            ticker_prices = prices[ticker].dropna()
            
            if len(ticker_prices) < 2:
                continue
            
            # Calculate returns
            returns = ticker_prices.pct_change() * 100
            
            # Find extreme returns
            extreme_mask = returns.abs() > threshold
            extreme_dates = returns[extreme_mask].index
            
            for date in extreme_dates:
                date_idx = ticker_prices.index.get_loc(date)
                if date_idx > 0:
                    price_before = ticker_prices.iloc[date_idx - 1]
                    price_after = ticker_prices.iloc[date_idx]
                    return_pct = returns.loc[date]
                    
                    # Classify issue type
                    if return_pct > 500:  # >500% likely data error
                        issue_type = "data_error"
                    elif return_pct > 200:  # 200-500% likely split/adjustment issue
                        issue_type = "possible_split"
                    elif return_pct < -90:  # <-90% likely data error or delisting
                        issue_type = "possible_delisting"
                    else:
                        issue_type = "extreme_but_possible"
                    
                    extreme_returns.append({
                        'country': country,
                        'ticker': ticker,
                        'date': date,
                        'price_before': price_before,
                        'price_after': price_after,
                        'return_pct': return_pct,
                        'issue_type': issue_type
                    })
    
    df = pd.DataFrame(extreme_returns)
    
    if len(df) > 0:
        logger.warning(f"Found {len(df)} extreme returns (>={threshold}%)")
        logger.info(f"  - Data errors (>500%): {len(df[df['issue_type'] == 'data_error'])}")
        logger.info(f"  - Possible splits (200-500%): {len(df[df['issue_type'] == 'possible_split'])}")
        logger.info(f"  - Possible delistings (<-90%): {len(df[df['issue_type'] == 'possible_delisting'])}")
    else:
        logger.info("No extreme returns found")
    
    return df


def investigate_stock_data(country: str, ticker: str) -> Dict:
    """
    Investigate a specific stock's price data for issues.
    
    Parameters
    ----------
    country : str
        Country name
    ticker : str
        Stock ticker
    
    Returns
    -------
    dict
        Dictionary with investigation results
    """
    stock_file = os.path.join(DATA_RAW_DIR, f"prices_stocks_{country}.csv")
    
    if not os.path.exists(stock_file):
        return {'error': 'File not found'}
    
    prices = pd.read_csv(stock_file, index_col=0, parse_dates=True)
    
    if ticker not in prices.columns:
        return {'error': 'Ticker not found'}
    
    ticker_prices = prices[ticker].dropna()
    returns = ticker_prices.pct_change() * 100
    
    # Find issues
    issues = []
    
    # Extreme returns
    extreme = returns[returns.abs() > 100]
    if len(extreme) > 0:
        for date, ret in extreme.items():
            issues.append({
                'date': date,
                'type': 'extreme_return',
                'value': ret,
                'price_before': ticker_prices.loc[ticker_prices.index[ticker_prices.index < date][-1]] if len(ticker_prices.index[ticker_prices.index < date]) > 0 else np.nan,
                'price_after': ticker_prices.loc[date]
            })
    
    # Price jumps
    price_ratios = ticker_prices / ticker_prices.shift(1)
    jumps = price_ratios[(price_ratios > 10) | (price_ratios < 0.1)]
    if len(jumps) > 0:
        for date, ratio in jumps.items():
            if date not in [i['date'] for i in issues]:
                issues.append({
                    'date': date,
                    'type': 'price_jump',
                    'value': ratio,
                    'price_before': ticker_prices.loc[ticker_prices.index[ticker_prices.index < date][-1]] if len(ticker_prices.index[ticker_prices.index < date]) > 0 else np.nan,
                    'price_after': ticker_prices.loc[date]
                })
    
    return {
        'ticker': ticker,
        'country': country,
        'n_observations': len(ticker_prices),
        'date_range': (ticker_prices.index.min(), ticker_prices.index.max()),
        'price_range': (ticker_prices.min(), ticker_prices.max()),
        'mean_return': returns.mean(),
        'std_return': returns.std(),
        'issues': issues
    }


def fix_stock_data(country: str, ticker: str, method: str = 'filter') -> bool:
    """
    Fix problematic data for a specific stock.
    
    Parameters
    ----------
    country : str
        Country name
    ticker : str
        Stock ticker
    method : str
        Fix method: 'filter' (remove problematic months) or 'exclude' (exclude entire stock)
    
    Returns
    -------
    bool
        True if fix was applied, False otherwise
    """
    stock_file = os.path.join(DATA_RAW_DIR, f"prices_stocks_{country}.csv")
    
    if not os.path.exists(stock_file):
        return False
    
    prices = pd.read_csv(stock_file, index_col=0, parse_dates=True)
    
    if ticker not in prices.columns:
        return False
    
    investigation = investigate_stock_data(country, ticker)
    
    if len(investigation.get('issues', [])) == 0:
        return False
    
    if method == 'filter':
        # Remove problematic months
        problematic_dates = [issue['date'] for issue in investigation['issues']]
        prices.loc[problematic_dates, ticker] = np.nan
        logger.info(f"Filtered {len(problematic_dates)} problematic months for {ticker}")
    elif method == 'exclude':
        # Remove entire stock
        prices = prices.drop(columns=[ticker])
        logger.info(f"Excluded {ticker} entirely")
    
    # Save fixed data
    prices.to_csv(stock_file)
    return True


def fix_all_extreme_betas(extreme_threshold: float = 5.0, return_threshold: float = 100.0) -> pd.DataFrame:
    """
    Identify and fix all stocks with extreme betas.
    
    Parameters
    ----------
    extreme_threshold : float
        Beta threshold for extreme values (default: 5.0)
    return_threshold : float
        Return threshold for extreme returns (default: 100%)
    
    Returns
    -------
    pd.DataFrame
        Summary of fixes applied
    """
    logger.info("="*70)
    logger.info("FIXING EXTREME BETA STOCKS")
    logger.info("="*70)
    
    # Load CAPM results
    capm_results_file = os.path.join('results', 'data', 'capm_results.csv')
    if not os.path.exists(capm_results_file):
        logger.error("CAPM results file not found. Run CAPM regressions first.")
        return pd.DataFrame()
    
    capm_results = pd.read_csv(capm_results_file)
    
    # Find extreme betas
    extreme_betas = capm_results[capm_results['beta'].abs() > extreme_threshold].copy()
    
    logger.info(f"Found {len(extreme_betas)} stocks with |beta| > {extreme_threshold}")
    
    if len(extreme_betas) == 0:
        logger.info("No extreme betas to fix")
        return pd.DataFrame()
    
    # Scan for extreme returns
    extreme_returns = scan_extreme_returns(threshold=return_threshold)
    
    fixes_applied = []
    
    for _, row in extreme_betas.iterrows():
        country = row['country']
        ticker = row['ticker']
        beta = row['beta']
        
        logger.info(f"\nInvestigating {ticker} ({country}) - Beta: {beta:.2f}")
        
        # Check if this stock has extreme returns
        stock_extreme = extreme_returns[
            (extreme_returns['country'] == country) & 
            (extreme_returns['ticker'] == ticker)
        ]
        
        if len(stock_extreme) > 0:
            logger.warning(f"  Found {len(stock_extreme)} extreme returns")
            for _, ext_row in stock_extreme.iterrows():
                logger.warning(f"    {ext_row['date']}: {ext_row['return_pct']:.1f}% ({ext_row['issue_type']})")
            
            # Fix based on issue type
            data_errors = stock_extreme[stock_extreme['issue_type'] == 'data_error']
            if len(data_errors) > 0:
                # Filter out data error months
                fix_applied = fix_stock_data(country, ticker, method='filter')
                fixes_applied.append({
                    'country': country,
                    'ticker': ticker,
                    'beta_before': beta,
                    'method': 'filter_extreme_months',
                    'n_issues': len(data_errors),
                    'fix_applied': fix_applied
                })
        else:
            # Investigate further
            investigation = investigate_stock_data(country, ticker)
            if len(investigation.get('issues', [])) > 0:
                logger.warning(f"  Found {len(investigation['issues'])} issues")
                # For now, flag for manual review
                fixes_applied.append({
                    'country': country,
                    'ticker': ticker,
                    'beta_before': beta,
                    'method': 'manual_review_needed',
                    'n_issues': len(investigation['issues']),
                    'fix_applied': False
                })
    
    summary = pd.DataFrame(fixes_applied)
    
    if len(summary) > 0:
        logger.info(f"\n{'='*70}")
        logger.info(f"FIX SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Stocks fixed: {summary['fix_applied'].sum()}")
        logger.info(f"Stocks needing manual review: {(~summary['fix_applied']).sum()}")
    
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run fix
    summary = fix_all_extreme_betas(extreme_threshold=5.0, return_threshold=100.0)
    
    if len(summary) > 0:
        output_file = os.path.join('results', 'data', 'extreme_beta_fixes.csv')
        summary.to_csv(output_file, index=False)
        print(f"\nFix summary saved to: {output_file}")

