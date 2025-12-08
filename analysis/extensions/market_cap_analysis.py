"""
market_cap_analysis.py

Calculate market-capitalization weighted average betas and compare to equal-weighted averages.
This addresses assignment requirement: "What is the market value weighted average of the betas?"
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
import yfinance as yf
from datetime import datetime

from analysis.utils.config import (
    DATA_RAW_DIR,
    RESULTS_DATA_DIR,
    RESULTS_REPORTS_DIR,
    COUNTRIES,
    ANALYSIS_SETTINGS
)

logger = logging.getLogger(__name__)


def fetch_market_cap(ticker: str, country: str, date: Optional[str] = None) -> Optional[float]:
    """
    Fetch market capitalization for a stock ticker.
    
    Parameters
    ----------
    ticker : str
        Stock ticker (e.g., 'ATO.PA')
    country : str
        Country name
    date : str, optional
        Date to fetch market cap (default: end of analysis period)
    
    Returns
    -------
    float or None
        Market capitalization in millions, or None if unavailable
    """
    if date is None:
        date = ANALYSIS_SETTINGS.end_date
    
    try:
        # Try to fetch from yfinance
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get market cap
        market_cap = info.get('marketCap')
        if market_cap is None:
            # Try alternative keys
            market_cap = info.get('totalAssets') or info.get('enterpriseValue')
        
        if market_cap is not None:
            # Convert to millions
            return market_cap / 1e6
        
        return None
    except Exception as e:
        logger.debug(f"Could not fetch market cap for {ticker}: {e}")
        return None


def estimate_market_cap_from_price(ticker: str, country: str, prices_df: pd.DataFrame) -> Optional[float]:
    """
    Estimate market cap from price data (rough approximation).
    Uses average price and assumes typical shares outstanding for large-cap stocks.
    
    Parameters
    ----------
    ticker : str
        Stock ticker
    country : str
        Country name
    prices_df : pd.DataFrame
        DataFrame with price data
    
    Returns
    -------
    float or None
        Estimated market cap in millions
    """
    if ticker not in prices_df.columns:
        return None
    
    # Get average price over analysis period
    prices = prices_df[ticker].dropna()
    if len(prices) == 0:
        return None
    
    avg_price = prices.mean()
    
    # Rough estimate: assume shares outstanding based on typical large-cap stock
    # This is a very rough approximation - real analysis would need actual shares outstanding
    # For European large-caps, typical shares outstanding: 100M - 1B shares
    # We'll use a conservative estimate based on price level
    
    if avg_price < 10:
        estimated_shares = 500_000_000  # 500M shares for low-priced stocks
    elif avg_price < 50:
        estimated_shares = 300_000_000  # 300M shares for mid-priced stocks
    else:
        estimated_shares = 200_000_000  # 200M shares for high-priced stocks
    
    estimated_market_cap = avg_price * estimated_shares / 1e6  # Convert to millions
    
    return estimated_market_cap


def load_market_caps_for_country(country: str, capm_results: pd.DataFrame) -> pd.DataFrame:
    """
    Load or estimate market capitalizations for all stocks in a country.
    
    Parameters
    ----------
    country : str
        Country name
    capm_results : pd.DataFrame
        CAPM results with ticker and country columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [ticker, market_cap_millions, source]
    """
    country_stocks = capm_results[capm_results['country'] == country].copy()
    
    # Load price data
    prices_file = os.path.join(DATA_RAW_DIR, f"prices_stocks_{country}.csv")
    if not os.path.exists(prices_file):
        logger.warning(f"Price file not found for {country}")
        return pd.DataFrame()
    
    prices_df = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    
    market_caps = []
    
    for _, row in country_stocks.iterrows():
        ticker = row['ticker']
        
        # Try to fetch real market cap
        market_cap = fetch_market_cap(ticker, country)
        source = 'yfinance'
        
        # If unavailable, estimate from prices
        if market_cap is None:
            market_cap = estimate_market_cap_from_price(ticker, country, prices_df)
            source = 'estimated'
        
        if market_cap is not None:
            market_caps.append({
                'ticker': ticker,
                'market_cap_millions': market_cap,
                'source': source
            })
        else:
            logger.warning(f"Could not determine market cap for {ticker}")
    
    return pd.DataFrame(market_caps)


def calculate_market_cap_weighted_betas(capm_results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate market-cap weighted average betas by country and overall.
    
    Parameters
    ----------
    capm_results : pd.DataFrame
        CAPM results with beta, country, ticker columns
    
    Returns
    -------
    pd.DataFrame
        Summary with equal-weighted and market-cap weighted betas by country
    """
    logger.info("="*70)
    logger.info("CALCULATING MARKET-CAP WEIGHTED BETAS")
    logger.info("="*70)
    
    results = []
    
    # Process each country
    for country in COUNTRIES.keys():
        logger.info(f"\nProcessing {country}...")
        
        country_stocks = capm_results[capm_results['country'] == country].copy()
        valid_stocks = country_stocks[country_stocks['is_valid'] == True]
        
        if len(valid_stocks) == 0:
            continue
        
        # Equal-weighted average beta
        ew_beta = valid_stocks['beta'].mean()
        
        # Load market caps
        market_caps = load_market_caps_for_country(country, valid_stocks)
        
        if len(market_caps) == 0:
            logger.warning(f"No market cap data for {country}, using equal-weighted only")
            results.append({
                'country': country,
                'n_stocks': len(valid_stocks),
                'equal_weighted_beta': ew_beta,
                'market_cap_weighted_beta': np.nan,
                'market_cap_total_millions': np.nan,
                'n_stocks_with_mcap': 0,
                'mcap_data_source': 'unavailable'
            })
            continue
        
        # Merge market caps with betas
        merged = valid_stocks.merge(market_caps, on='ticker', how='left')
        merged = merged[merged['market_cap_millions'].notna()]
        
        if len(merged) == 0:
            logger.warning(f"No valid market cap data for {country}")
            results.append({
                'country': country,
                'n_stocks': len(valid_stocks),
                'equal_weighted_beta': ew_beta,
                'market_cap_weighted_beta': np.nan,
                'market_cap_total_millions': np.nan,
                'n_stocks_with_mcap': 0,
                'mcap_data_source': 'unavailable'
            })
            continue
        
        # Calculate market-cap weighted beta
        total_mcap = merged['market_cap_millions'].sum()
        mcap_weights = merged['market_cap_millions'] / total_mcap
        mw_beta = (merged['beta'] * mcap_weights).sum()
        
        # Determine data source
        sources = merged['source'].unique()
        if len(sources) == 1:
            data_source = sources[0]
        else:
            data_source = 'mixed'
        
        results.append({
            'country': country,
            'n_stocks': len(valid_stocks),
            'equal_weighted_beta': ew_beta,
            'market_cap_weighted_beta': mw_beta,
            'market_cap_total_millions': total_mcap,
            'n_stocks_with_mcap': len(merged),
            'mcap_data_source': data_source
        })
        
        logger.info(f"  Equal-weighted beta: {ew_beta:.4f}")
        logger.info(f"  Market-cap weighted beta: {mw_beta:.4f}")
        logger.info(f"  Difference: {abs(mw_beta - ew_beta):.4f}")
        logger.info(f"  Total market cap: {total_mcap:.0f} million")
        logger.info(f"  Data source: {data_source}")
    
    # Overall (all countries combined)
    logger.info(f"\nCalculating overall statistics...")
    all_valid = capm_results[capm_results['is_valid'] == True].copy()
    overall_ew_beta = all_valid['beta'].mean()
    
    # Get all market caps
    all_market_caps = []
    for country in COUNTRIES.keys():
        country_caps = load_market_caps_for_country(country, all_valid)
        if len(country_caps) > 0:
            all_market_caps.append(country_caps)
    
    if len(all_market_caps) > 0:
        all_caps_df = pd.concat(all_market_caps, ignore_index=True)
        merged_all = all_valid.merge(all_caps_df, on='ticker', how='left')
        merged_all = merged_all[merged_all['market_cap_millions'].notna()]
        
        if len(merged_all) > 0:
            total_mcap_all = merged_all['market_cap_millions'].sum()
            mcap_weights_all = merged_all['market_cap_millions'] / total_mcap_all
            overall_mw_beta = (merged_all['beta'] * mcap_weights_all).sum()
            
            sources_all = merged_all['source'].unique()
            if len(sources_all) == 1:
                data_source_all = sources_all[0]
            else:
                data_source_all = 'mixed'
            
            results.append({
                'country': 'Overall',
                'n_stocks': len(all_valid),
                'equal_weighted_beta': overall_ew_beta,
                'market_cap_weighted_beta': overall_mw_beta,
                'market_cap_total_millions': total_mcap_all,
                'n_stocks_with_mcap': len(merged_all),
                'mcap_data_source': data_source_all
            })
            
            logger.info(f"  Overall equal-weighted beta: {overall_ew_beta:.4f}")
            logger.info(f"  Overall market-cap weighted beta: {overall_mw_beta:.4f}")
            logger.info(f"  Difference: {abs(overall_mw_beta - overall_ew_beta):.4f}")
    
    summary_df = pd.DataFrame(results)
    
    return summary_df


def generate_market_cap_report(summary_df: pd.DataFrame) -> None:
    """
    Generate market-cap weighted beta report and save to files.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary DataFrame from calculate_market_cap_weighted_betas
    """
    logger.info("\n" + "="*70)
    logger.info("GENERATING MARKET-CAP WEIGHTED BETA REPORT")
    logger.info("="*70)
    
    # Save CSV (create tables directory if needed)
    tables_dir = os.path.join(os.path.dirname(RESULTS_REPORTS_DIR), "tables")
    os.makedirs(tables_dir, exist_ok=True)
    output_file = os.path.join(tables_dir, "table7_market_cap_weighted_betas.csv")
    summary_df.to_csv(output_file, index=False)
    logger.info(f"✅ Saved: {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("MARKET-CAP WEIGHTED BETA SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    for _, row in summary_df.iterrows():
        country = row['country']
        ew_beta = row['equal_weighted_beta']
        mw_beta = row['market_cap_weighted_beta']
        
        if pd.isna(mw_beta):
            print(f"\n{country}:")
            print(f"  Equal-weighted beta: {ew_beta:.4f}")
            print(f"  Market-cap weighted beta: Not available (data limitations)")
        else:
            diff = abs(mw_beta - ew_beta)
            pct_diff = (diff / ew_beta) * 100 if ew_beta != 0 else 0
            
            print(f"\n{country}:")
            print(f"  Equal-weighted beta: {ew_beta:.4f}")
            print(f"  Market-cap weighted beta: {mw_beta:.4f}")
            print(f"  Difference: {diff:.4f} ({pct_diff:.1f}%)")
            
            if diff < 0.05:
                print(f"  → Betas are well-represented (small difference)")
            elif diff < 0.15:
                print(f"  → Moderate difference - large-cap stocks have slightly different betas")
            else:
                print(f"  → Significant difference - market-cap weighting matters")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load CAPM results
    capm_results_file = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
    if not os.path.exists(capm_results_file):
        logger.error("CAPM results not found. Run CAPM regressions first.")
        exit(1)
    
    capm_results = pd.read_csv(capm_results_file)
    
    # Calculate market-cap weighted betas
    summary = calculate_market_cap_weighted_betas(capm_results)
    
    # Generate report
    generate_market_cap_report(summary)

