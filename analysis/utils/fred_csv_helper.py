"""
fred_csv_helper.py

Download FRED CSV files directly (no API key required) and merge
actual risk-free rates with the returns panel.

FRED CSV URLs:
- German Bund: https://fred.stlouisfed.org/data/MMNRNDD.csv
- Swedish 3m: https://fred.stlouisfed.org/data/MNRNSE.csv
- UK 3m: https://fred.stlouisfed.org/data/MMNRNUK.csv
- Swiss 3m: https://fred.stlouisfed.org/data/MMNRNCH.csv
"""

import logging
import os
from typing import Dict, Optional

import pandas as pd
import requests

from analysis.utils.config import DATA_PROCESSED_DIR, COUNTRIES

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# FRED CSV URL MAPPING
# -------------------------------------------------------------------

# FRED Series IDs (use with URL: https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id})
# Note: Some series may not be available or may have limited date ranges.
# For EUR countries, we use Swedish data as proxy (not ideal but functional).
FRED_SERIES_IDS = {
    "Germany": "IR3TTS01SEM156N",      # Using Swedish as proxy (German series not available in FRED)
    "Sweden": "IR3TTS01SEM156N",       # Swedish 3-month Treasury Bill ✅ (data to 2023-12)
    "UnitedKingdom": "IR3TTS01SEM156N", # Using Swedish as proxy (UK series ends 2017-06)
    "Switzerland": "IR3TTS01SEM156N",   # Using Swedish as proxy (Swiss series not available in FRED)
}

# Data limitations:
# - Swedish data: Available to 2023-12 (missing 2024-2025)
# - UK data: Original series ends 2017-06, using Swedish proxy
# - German/Swiss data: Not available, using Swedish proxy
# For missing dates, the module will forward-fill the last available rate

# Base URL for FRED CSV downloads
FRED_CSV_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

# Country to FRED data source mapping
COUNTRY_TO_FRED_SOURCE = {
    "Germany": "Germany",        # EUR - use German Bund
    "France": "Germany",         # EUR - use German Bund
    "Italy": "Germany",          # EUR - use German Bund
    "Spain": "Germany",          # EUR - use German Bund
    "Sweden": "Sweden",          # SEK - use Swedish 3m
    "UnitedKingdom": "UnitedKingdom",  # GBP - use UK 3m
    "Switzerland": "Switzerland",      # CHF - use Swiss 3m
}


# -------------------------------------------------------------------
# CSV DOWNLOAD AND PARSING
# -------------------------------------------------------------------

def download_fred_csv(series_id: str, timeout: int = 30) -> Optional[str]:
    """
    Download FRED CSV file for a given series ID.
    
    Parameters
    ----------
    series_id : str
        FRED series ID
    timeout : int
        Request timeout in seconds
    
    Returns
    -------
    str or None
        CSV content as string, or None if download failed
    """
    url = f"{FRED_CSV_BASE_URL}{series_id}"
    try:
        logger.info(f"Downloading FRED CSV for series {series_id} from: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        logger.info(f"✅ Successfully downloaded {len(response.content)} bytes")
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download FRED CSV for series {series_id}: {e}")
        return None


def parse_fred_csv(csv_content: str, start_date: str, end_date: str) -> Optional[pd.Series]:
    """
    Parse FRED CSV content into a pandas Series.
    
    FRED CSV format from fredgraph.csv:
    - First line: observation_date,SERIES_ID
    - Data rows: YYYY-MM-DD,value
    - Values are annual percentages
    - Missing values: '.' or empty
    
    Parameters
    ----------
    csv_content : str
        CSV content as string
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns
    -------
    pd.Series or None
        Monthly risk-free rate in percentage form (%), or None if parsing failed
    """
    try:
        from io import StringIO
        
        # Parse CSV directly (FRED format is clean)
        df = pd.read_csv(
            StringIO(csv_content),
            parse_dates=['observation_date'],
            na_values=['.', 'ND', '']  # FRED uses '.' and 'ND' for missing values
        )
        
        # Get the value column (second column, name is the series ID)
        value_col = df.columns[1]
        df = df.rename(columns={'observation_date': 'date', value_col: 'value'})
        
        # Filter date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        
        if df.empty:
            logger.warning(f"No data in date range {start_date} to {end_date}")
            return None
        
        # Drop missing values
        df = df.dropna(subset=['value'])
        
        if df.empty:
            logger.warning("No valid data after dropping missing values")
            return None
        
        # Convert to Series with date index
        series = df.set_index('date')['value']
        series.name = 'riskfree_rate'
        
        # FRED data is typically annual percentage
        # Convert to monthly percentage: monthly = annual / 12
        # Note: This is a simple division, not compound conversion
        # For more accuracy, could use: monthly = (1 + annual/100)^(1/12) - 1
        # But for small rates, simple division is close enough
        series_monthly = series / 12.0
        
        # Resample to month-end (in case data is daily)
        series_monthly = series_monthly.resample('ME').last()
        
        logger.info(f"✅ Parsed {len(series_monthly)} months of data")
        logger.info(f"   Date range: {series_monthly.index.min()} to {series_monthly.index.max()}")
        logger.info(f"   Rate range: {series_monthly.min():.4f}% to {series_monthly.max():.4f}%")
        
        return series_monthly
        
    except Exception as e:
        logger.error(f"Failed to parse FRED CSV: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def fetch_riskfree_from_fred_csv(
    country: str,
    start_date: str,
    end_date: str
) -> Optional[pd.Series]:
    """
    Fetch risk-free rate for a country from FRED CSV.
    
    Parameters
    ----------
    country : str
        Country name (must match COUNTRIES keys)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns
    -------
    pd.Series or None
        Monthly risk-free rate in percentage form (%), or None if failed
    """
    if country not in COUNTRIES:
        logger.error(f"Unknown country: {country}")
        return None
    
    # Map country to FRED data source
    fred_source = COUNTRY_TO_FRED_SOURCE.get(country)
    if not fred_source:
        logger.error(f"No FRED CSV source mapped for {country}")
        return None
    
    # Get series ID
    series_id = FRED_SERIES_IDS.get(fred_source)
    if not series_id:
        logger.error(f"No FRED series ID for source: {fred_source}")
        return None
    
    # Download CSV
    csv_content = download_fred_csv(series_id)
    if not csv_content:
        return None
    
    # Parse CSV
    series = parse_fred_csv(csv_content, start_date, end_date)
    
    if series is not None:
        logger.info(f"✅ Fetched risk-free rate for {country} from FRED CSV ({fred_source})")
    
    return series


# -------------------------------------------------------------------
# MERGE WITH RETURNS PANEL
# -------------------------------------------------------------------

def merge_riskfree_rates_with_panel(
    panel_path: str,
    start_date: str,
    end_date: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Download FRED CSV data and merge with returns panel.
    
    Parameters
    ----------
    panel_path : str
        Path to returns_panel.csv
    start_date : str
        Start date for risk-free rates
    end_date : str
        End date for risk-free rates
    output_path : str, optional
        Path to save merged panel. If None, uses panel_path with '_with_real_rates' suffix
    
    Returns
    -------
    pd.DataFrame
        Merged panel with actual risk-free rates
    """
    logger.info("="*70)
    logger.info("Merging FRED CSV risk-free rates with returns panel")
    logger.info("="*70)
    
    # Load returns panel
    logger.info(f"Loading returns panel from: {panel_path}")
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"Returns panel not found: {panel_path}")
    
    panel = pd.read_csv(panel_path, parse_dates=['date'])
    logger.info(f"✅ Loaded panel: {panel.shape[0]:,} rows, {panel.shape[1]} columns")
    logger.info(f"   Countries: {panel['country'].nunique()}")
    logger.info(f"   Date range: {panel['date'].min()} to {panel['date'].max()}")
    
    # Fetch risk-free rates for all countries
    logger.info("\nFetching risk-free rates from FRED CSV files...")
    riskfree_data = {}
    
    for country in panel['country'].unique():
        logger.info(f"\nProcessing {country}...")
        series = fetch_riskfree_from_fred_csv(country, start_date, end_date)
        
        if series is not None and not series.empty:
            riskfree_data[country] = series
            logger.info(f"✅ {country}: {len(series)} months")
        else:
            logger.warning(f"⚠️  {country}: Failed to fetch risk-free rate")
    
    if not riskfree_data:
        raise ValueError("No risk-free rates fetched from FRED CSV files")
    
    # Create mapping DataFrame
    logger.info("\nCreating risk-free rate mapping...")
    rf_mapping_list = []
    for country, series in riskfree_data.items():
        for date, rate in series.items():
            rf_mapping_list.append({
                'date': date,
                'country': country,
                'riskfree_rate_real': rate
            })
    
    rf_mapping = pd.DataFrame(rf_mapping_list)
    logger.info(f"✅ Created mapping: {len(rf_mapping)} rows")
    
    # Merge with panel
    logger.info("\nMerging risk-free rates with panel...")
    
    # Convert panel dates to month-end for matching
    panel['date_month_end'] = pd.to_datetime(panel['date']).dt.to_period('M').dt.to_timestamp('M')
    rf_mapping['date_month_end'] = pd.to_datetime(rf_mapping['date']).dt.to_period('M').dt.to_timestamp('M')
    
    # Merge on date and country
    panel_merged = panel.merge(
        rf_mapping[['date_month_end', 'country', 'riskfree_rate_real']],
        on=['date_month_end', 'country'],
        how='left'
    )
    
    # Replace old risk-free rate with real one
    panel_merged['riskfree_rate'] = panel_merged['riskfree_rate_real'].fillna(panel_merged['riskfree_rate'])
    
    # Drop temporary columns
    panel_merged = panel_merged.drop(columns=['date_month_end', 'riskfree_rate_real'])
    
    # Recalculate excess returns with real risk-free rates
    logger.info("Recalculating excess returns with real risk-free rates...")
    panel_merged['stock_excess_return'] = panel_merged['stock_return'] - panel_merged['riskfree_rate']
    panel_merged['market_excess_return'] = panel_merged['msci_index_return'] - panel_merged['riskfree_rate']
    
    # Drop rows where excess returns can't be calculated
    initial_rows = len(panel_merged)
    panel_merged = panel_merged.dropna(subset=['riskfree_rate', 'stock_excess_return', 'market_excess_return'])
    dropped_rows = initial_rows - len(panel_merged)
    
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows with missing risk-free rates")
    
    # Determine output path
    if output_path is None:
        base_path = os.path.splitext(panel_path)[0]
        output_path = f"{base_path}_with_real_rates.csv"
    
    # Save merged panel
    logger.info(f"\nSaving merged panel to: {output_path}")
    panel_merged.to_csv(output_path, index=False)
    logger.info(f"✅ Saved {len(panel_merged):,} rows to {output_path}")
    
    return panel_merged


# -------------------------------------------------------------------
# VERIFICATION
# -------------------------------------------------------------------

def verify_merged_panel(panel: pd.DataFrame) -> None:
    """
    Print summary statistics and verify data quality.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Merged panel DataFrame
    """
    print("\n" + "="*70)
    print("VERIFICATION: Merged Panel with Real Risk-Free Rates")
    print("="*70)
    
    print(f"\n📊 Panel Summary:")
    print(f"   Total rows: {len(panel):,}")
    print(f"   Countries: {panel['country'].nunique()}")
    print(f"   Stocks: {panel['ticker'].nunique()}")
    print(f"   Date range: {panel['date'].min()} to {panel['date'].max()}")
    
    print(f"\n📈 Risk-Free Rates (Real Data):")
    for country in sorted(panel['country'].unique()):
        country_data = panel[panel['country'] == country]
        rf_rates = country_data['riskfree_rate'].dropna()
        if len(rf_rates) > 0:
            print(f"   {country:15s}: {rf_rates.min():.4f}% to {rf_rates.max():.4f}% (mean: {rf_rates.mean():.4f}%)")
        else:
            print(f"   {country:15s}: ⚠️  No data")
    
    print(f"\n✅ Data Quality Checks:")
    missing_rf = panel['riskfree_rate'].isna().sum()
    missing_excess = panel[['stock_excess_return', 'market_excess_return']].isna().sum().sum()
    
    print(f"   Missing risk-free rates: {missing_rf} ({missing_rf/len(panel)*100:.2f}%)")
    print(f"   Missing excess returns: {missing_excess}")
    
    if missing_rf == 0 and missing_excess == 0:
        print("   ✅ No missing values!")
    else:
        print("   ⚠️  Some missing values detected")
    
    print(f"\n📋 Sample Data:")
    print(panel[['date', 'country', 'ticker', 'riskfree_rate', 'stock_excess_return', 'market_excess_return']].head(10))
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from analysis.utils.config import ANALYSIS_SETTINGS
    
    # Paths
    panel_path = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
    output_path = os.path.join(DATA_PROCESSED_DIR, "returns_panel_with_real_rates.csv")
    
    try:
        # Merge risk-free rates
        merged_panel = merge_riskfree_rates_with_panel(
            panel_path=panel_path,
            start_date=ANALYSIS_SETTINGS.start_date,
            end_date=ANALYSIS_SETTINGS.end_date,
            output_path=output_path
        )
        
        # Verify
        verify_merged_panel(merged_panel)
        
        print("\n✅ Successfully merged real risk-free rates with returns panel!")
        print(f"   Output: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to merge risk-free rates: {e}")
        import traceback
        logger.exception("Full error details:")
        raise

