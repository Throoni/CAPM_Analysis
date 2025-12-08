"""
data_collection.py

Module for collecting stock and index price data by country from Yahoo Finance.
Downloads monthly prices for all stocks and indices in the universe, organized by country.
"""

import os
import logging

import pandas as pd

from analysis.utils.config import ANALYSIS_SETTINGS, DATA_RAW_DIR
from analysis.utils.universe import load_stock_universe
from analysis.data.yf_helper import download_monthly_prices


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_data_date_range(df: pd.DataFrame, file_path: str, expected_start: str, expected_end: str) -> dict:
    """
    Validate that downloaded data has the expected date range.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    file_path : str
        Path to the file for logging
    expected_start : str
        Expected start date (YYYY-MM-DD)
    expected_end : str
        Expected end date (YYYY-MM-DD)
    
    Returns
    -------
    dict
        Dictionary with validation results: {'is_valid': bool, 'issues': list}
    """
    if df.empty:
        return {'is_valid': False, 'issues': ['File is empty']}
    
    df.index = pd.to_datetime(df.index)
    actual_start = df.index.min()
    actual_end = df.index.max()
    expected_start_dt = pd.to_datetime(expected_start)
    expected_end_dt = pd.to_datetime(expected_end)
    
    issues = []
    
    # Check start date (allow some flexibility)
    start_diff = (actual_start.year - expected_start_dt.year) * 12 + (actual_start.month - expected_start_dt.month)
    if start_diff > 1:
        issues.append(f"Start date is {start_diff} months after expected ({actual_start.strftime('%Y-%m-%d')} vs {expected_start})")
    
    # Check end date (this is more critical)
    end_diff = (expected_end_dt.year - actual_end.year) * 12 + (expected_end_dt.month - actual_end.month)
    if end_diff > 3:
        issues.append(
            f"⚠️  DATA QUALITY ISSUE: End date is {end_diff} months before expected "
            f"({actual_end.strftime('%Y-%m-%d')} vs {expected_end}). "
            f"This may indicate delisting or data availability issues."
        )
    elif end_diff > 0:
        issues.append(
            f"Note: End date is {end_diff} month(s) before expected "
            f"({actual_end.strftime('%Y-%m-%d')} vs {expected_end})"
        )
    
    if issues:
        logger.warning(f"Data validation issues for {os.path.basename(file_path)}:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return {'is_valid': len(issues) == 0, 'issues': issues, 'actual_start': actual_start, 'actual_end': actual_end}


def collect_prices_by_country():
    """
    Collect monthly stock and index prices for each country in the universe.
    
    For each country:
    1. Loads the stock universe
    2. Filters stocks and indices by country
    3. Downloads monthly prices from Yahoo Finance
    4. Saves prices to CSV files in data/raw/
    
    Files are saved as:
    - data/raw/prices_stocks_[country].csv
    - data/raw/prices_indices_[country].csv
    
    Each country is processed independently with error handling, so failures
    in one country don't stop processing of others.
    """
    logger.info("Starting data collection for all countries")
    
    # Load the stock universe
    try:
        universe = load_stock_universe()
        logger.info(f"Loaded universe with {len(universe)} total entries")
    except Exception as e:
        logger.error(f"Failed to load stock universe: {e}")
        raise
    
    # Get unique countries
    countries = universe['country'].unique()
    logger.info(f"Processing {len(countries)} countries: {list(countries)}")
    
    # Extract date settings
    start_date = ANALYSIS_SETTINGS.start_date
    end_date = ANALYSIS_SETTINGS.end_date
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Track data quality issues
    data_quality_issues = []
    
    # Process each country
    for country in countries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing country: {country}")
        logger.info(f"{'='*60}")
        
        try:
            # Get stock tickers for this country
            stocks = universe[
                (universe['country'] == country) & 
                (universe['asset_type'] == 'stock')
            ]['ticker_yf'].tolist()
            
            # Get index tickers for this country
            indices = universe[
                (universe['country'] == country) & 
                (universe['asset_type'] == 'index')
            ]['ticker_yf'].tolist()
            
            logger.info(f"Found {len(stocks)} stocks and {len(indices)} indices for {country}")
            
            # Download and save stock prices
            if stocks:
                logger.info(f"Downloading stock prices for {country}...")
                df_stocks = download_monthly_prices(
                    stocks, 
                    start=start_date, 
                    end=end_date
                )
                
                stocks_path = os.path.join(
                    DATA_RAW_DIR, 
                    f"prices_stocks_{country}.csv"
                )
                df_stocks.to_csv(stocks_path, index=True)
                logger.info(f"SUCCESS: Saved stock prices to {stocks_path}")
                logger.info(f"  Shape: {df_stocks.shape}, Columns: {list(df_stocks.columns)}")
                
                # Validate date range
                validation = validate_data_date_range(df_stocks, stocks_path, start_date, end_date)
                if not validation['is_valid']:
                    data_quality_issues.append({
                        'file': stocks_path,
                        'country': country,
                        'type': 'stocks',
                        'issues': validation['issues']
                    })
            else:
                logger.warning(f"No stocks found for {country}, skipping stock price download")
            
            # Download and save index prices
            if indices:
                logger.info(f"Downloading index prices for {country}...")
                df_indices = download_monthly_prices(
                    indices, 
                    start=start_date, 
                    end=end_date
                )
                
                indices_path = os.path.join(
                    DATA_RAW_DIR, 
                    f"prices_indices_{country}.csv"
                )
                df_indices.to_csv(indices_path, index=True)
                logger.info(f"SUCCESS: Saved index prices to {indices_path}")
                logger.info(f"  Shape: {df_indices.shape}, Columns: {list(df_indices.columns)}")
                
                # Validate date range
                validation = validate_data_date_range(df_indices, indices_path, start_date, end_date)
                if not validation['is_valid']:
                    data_quality_issues.append({
                        'file': indices_path,
                        'country': country,
                        'type': 'indices',
                        'issues': validation['issues']
                    })
            else:
                logger.warning(f"No indices found for {country}, skipping index price download")
            
            logger.info(f"Completed processing for {country}")
            
        except Exception as e:
            logger.error(f"ERROR processing {country}: {e}")
            logger.exception("Full error details:")
            # Continue with next country instead of stopping
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info("Data collection completed for all countries")
    logger.info(f"{'='*60}")
    
    # Report data quality issues summary
    if data_quality_issues:
        logger.warning(f"\n{'='*60}")
        logger.warning("DATA QUALITY ISSUES SUMMARY")
        logger.warning(f"{'='*60}")
        logger.warning(f"Found {len(data_quality_issues)} file(s) with data quality issues:")
        for issue in data_quality_issues:
            logger.warning(f"\n  {issue['country']} ({issue['type']}): {os.path.basename(issue['file'])}")
            for detail in issue['issues']:
                logger.warning(f"    - {detail}")
        logger.warning(f"{'='*60}\n")
    else:
        logger.info("\n✅ All downloaded files have complete date ranges within expected parameters.")


if __name__ == "__main__":
    """
    Run data collection when script is executed directly.
    """
    try:
        collect_prices_by_country()
        print("\n" + "="*60)
        print("Data collection script completed successfully.")
        print("="*60)
    except Exception as e:
        logger.error(f"Fatal error in data collection: {e}")
        logger.exception("Full error details:")
        print("\n" + "="*60)
        print("Data collection script failed. Check logs for details.")
        print("="*60)
        raise

