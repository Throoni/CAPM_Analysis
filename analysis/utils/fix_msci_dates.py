"""
fix_msci_dates.py

Fix MSCI index date alignment: Convert month-start dates to month-end dates.

Issue: yfinance returns month-start dates (2020-12-01) but we need month-end (2020-12-31)
for proper alignment with returns panel and risk-free rates.
"""

import os
import logging
import pandas as pd
from analysis.utils.config import DATA_RAW_DIR, ANALYSIS_SETTINGS

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# MSCI index tickers
msci_tickers = {
    "Germany": "EWG",
    "France": "EWQ",
    "Italy": "EWI",
    "Spain": "EWP",
    "Sweden": "EWD",
    "UnitedKingdom": "EWU",
    "Switzerland": "EWL",
}

def fix_stock_dates():
    """
    Convert all stock price dates from month-start to month-end.
    """
    logger.info("="*70)
    logger.info("FIXING STOCK PRICE DATE ALIGNMENT")
    logger.info("="*70)
    
    from analysis.utils.config import COUNTRIES
    
    expected_start = pd.to_datetime(ANALYSIS_SETTINGS.start_date)
    expected_end = pd.to_datetime(ANALYSIS_SETTINGS.end_date)
    
    fixed_count = 0
    
    for country in COUNTRIES.keys():
        file_path = os.path.join(DATA_RAW_DIR, f"prices_stocks_{country}.csv")
        
        if not os.path.exists(file_path):
            continue
        
        logger.info(f"\n{country}:")
        
        # Load data
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        
        # Check current dates
        current_dates = df.index
        month_end_dates = current_dates.to_period('M').to_timestamp('M')
        
        # Check if conversion is needed
        if (current_dates == month_end_dates).all():
            logger.info(f"  ✅ Already month-end dates - no fix needed")
            continue
        
        # Update index to month-end
        df.index = month_end_dates
        
        # Save fixed data
        df.to_csv(file_path, index=True)
        logger.info(f"  ✅ Fixed {len(df.columns)} stocks, saved to {file_path}")
        
        fixed_count += 1
    
    logger.info(f"\n✅ Fixed dates for {fixed_count} stock price files")


def fix_msci_dates():
    """
    Convert all MSCI index dates from month-start to month-end.
    """
    logger.info("="*70)
    logger.info("FIXING MSCI INDEX DATE ALIGNMENT")
    logger.info("="*70)
    
    expected_start = pd.to_datetime(ANALYSIS_SETTINGS.start_date)
    expected_end = pd.to_datetime(ANALYSIS_SETTINGS.end_date)
    expected_dates = pd.date_range(start=expected_start, end=expected_end, freq='ME')
    
    logger.info(f"\nExpected month-end dates: {len(expected_dates)} months")
    logger.info(f"  First: {expected_dates[0].strftime('%Y-%m-%d')}")
    logger.info(f"  Last: {expected_dates[-1].strftime('%Y-%m-%d')}")
    
    fixed_count = 0
    
    for country, ticker in msci_tickers.items():
        file_path = os.path.join(DATA_RAW_DIR, f"prices_indices_msci_{country}.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"⚠️  {country} ({ticker}): File not found - {file_path}")
            continue
        
        logger.info(f"\n{country} ({ticker}):")
        
        # Load data
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        price_col = df.columns[0]
        prices = df[price_col].dropna()
        
        # Check current dates
        current_dates = prices.index
        logger.info(f"  Current dates: {len(current_dates)} observations")
        logger.info(f"  Sample: {current_dates[:3].strftime('%Y-%m-%d').tolist()}")
        
        # Convert to month-end
        month_end_dates = current_dates.to_period('M').to_timestamp('M')
        
        # Check if conversion is needed
        if (current_dates == month_end_dates).all():
            logger.info(f"  ✅ Already month-end dates - no fix needed")
            continue
        
        # Update index to month-end
        prices.index = month_end_dates
        
        # Verify we have the right dates
        actual_dates_set = set(prices.index)
        expected_dates_set = set(expected_dates)
        
        missing = expected_dates_set - actual_dates_set
        extra = actual_dates_set - expected_dates_set
        
        if missing:
            logger.warning(f"  ⚠️  Missing {len(missing)} expected dates after conversion")
        if extra:
            logger.warning(f"  ⚠️  Has {len(extra)} extra dates after conversion")
        
        if not missing and not extra:
            logger.info(f"  ✅ Perfect alignment after conversion!")
        
        # Save fixed data
        prices_df = prices.to_frame()
        prices_df.to_csv(file_path, index=True)
        logger.info(f"  ✅ Saved fixed dates to {file_path}")
        logger.info(f"  New date range: {prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}")
        
        fixed_count += 1
    
    logger.info("\n" + "="*70)
    logger.info(f"✅ Fixed dates for {fixed_count} MSCI index files")
    logger.info("="*70)
    
    # Verify all files now
    logger.info("\nVerifying all files after fix...")
    verify_all_msci_dates()


def verify_all_msci_dates():
    """
    Verify all MSCI index files have correct month-end dates.
    """
    expected_start = pd.to_datetime(ANALYSIS_SETTINGS.start_date)
    expected_end = pd.to_datetime(ANALYSIS_SETTINGS.end_date)
    expected_dates = pd.date_range(start=expected_start, end=expected_end, freq='ME')
    expected_dates_set = set(expected_dates)
    
    logger.info("\n" + "="*70)
    logger.info("VERIFICATION: MSCI Index Date Alignment")
    logger.info("="*70)
    
    all_ok = True
    
    for country, ticker in msci_tickers.items():
        file_path = os.path.join(DATA_RAW_DIR, f"prices_indices_msci_{country}.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"⚠️  {country} ({ticker}): File not found")
            all_ok = False
            continue
        
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        price_col = df.columns[0]
        prices = df[price_col].dropna()
        
        actual_dates = set(prices.index)
        
        # Check month-end
        month_ends = prices.index.to_period('M').to_timestamp('M')
        is_month_end = (prices.index == month_ends).all()
        
        # Check alignment
        missing = expected_dates_set - actual_dates
        extra = actual_dates - expected_dates_set
        
        status = "✅"
        if not is_month_end or missing or extra:
            status = "❌"
            all_ok = False
        
        logger.info(f"\n{country} ({ticker}): {status}")
        logger.info(f"  Prices: {len(prices)}")
        logger.info(f"  Month-end dates: {is_month_end}")
        logger.info(f"  Missing dates: {len(missing)}")
        logger.info(f"  Extra dates: {len(extra)}")
        
        if missing:
            logger.warning(f"    Missing: {sorted(list(missing))[:3]}")
        if extra:
            logger.warning(f"    Extra: {sorted(list(extra))[:3]}")
        
        # Check returns
        returns = prices.pct_change().dropna()
        logger.info(f"  Returns: {len(returns)} (expected: 59)")
        if len(returns) != 59:
            status = "❌"
            all_ok = False
    
    logger.info("\n" + "="*70)
    if all_ok:
        logger.info("✅ All MSCI indices have correct month-end dates and alignment!")
    else:
        logger.warning("⚠️  Some issues remain - check above")
    logger.info("="*70)


if __name__ == "__main__":
    # Fix both MSCI indices and stock prices
    fix_msci_dates()
    fix_stock_dates()
    
    logger.info("\n" + "="*70)
    logger.info("✅ All date fixes complete!")
    logger.info("="*70)

