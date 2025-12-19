"""
Risk-Free Rate Data Acquisition Module.

This module downloads 3-month government bond yields from the Federal Reserve
Economic Data (FRED) database without requiring an API key.

Data acquisition strategy:
    1. Direct CSV download from FRED public endpoints
    2. Multiple series ID patterns tested for each country
    3. Automatic fallback to German Bund for all EUR-denominated countries
    4. Conversion from annual to monthly rates

Supported countries and their primary sources:
    - Germany: German 3-month Treasury Bill (FRED: IR3TIB01DEM156N)
    - France, Italy, Spain, Netherlands: Use German Bund (EUR zone)
    - UK: UK 3-month Treasury Bill (FRED: IR3TIB01GBM156N)
    - Switzerland: Swiss 3-month Government Bond

Rate conventions:
    - All rates are annualized percentages from original sources
    - Converted to monthly rates: monthly_rate = annual_rate / 12
    - Final output in percentage form (e.g., 0.25 means 0.25% per month)

References
----------
Federal Reserve Bank of St. Louis. FRED Economic Data.
    https://fred.stlouisfed.org/
"""

import logging
import os
from typing import Dict, Optional, Tuple
from io import StringIO

import pandas as pd
import requests

from analysis.utils.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    COUNTRIES,
    ANALYSIS_SETTINGS,
)

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# -------------------------------------------------------------------
# FRED SERIES ID RESEARCH AND TESTING
# -------------------------------------------------------------------

# Known working series IDs (from research)
KNOWN_SERIES_IDS = {
    "Sweden": "IR3TTS01SEM156N",  # Swedish 3-month Treasury Bill (confirmed working)
}

# Series ID patterns to test for each country
SERIES_ID_PATTERNS = {
    "Germany": [
        "IR3TTS01DEM156N",  # Germany 3-month Treasury Bill
        "IR3TCD01DEM156N",  # Germany 3-month CD rate
        "IR3TTS01DE156N",   # Shorter format
        "IR3TTS01DE156M",   # Monthly
        "IR3TTS01DE156Q",   # Quarterly
        "IR3TTS01DE156A",   # Annual
        "IR3TTS01DE156D",   # Daily
    ],
    "France": [
        "IR3TTS01FRM156N",  # France 3-month Treasury Bill
        "IR3TCD01FRM156N",  # France 3-month CD rate
        "IR3TTS01FR156N",   # Shorter format
        "IR3TTS01FR156M",   # Monthly
    ],
    "Italy": [
        "IR3TTS01ITM156N",  # Italy 3-month Treasury Bill
        "IR3TCD01ITM156N",  # Italy 3-month CD rate
        "IR3TTS01IT156N",   # Shorter format
        "IR3TTS01IT156M",   # Monthly
    ],
    "Spain": [
        "IR3TTS01ESM156N",  # Spain 3-month Treasury Bill
        "IR3TCD01ESM156N",  # Spain 3-month CD rate
        "IR3TTS01ES156N",   # Shorter format
        "IR3TTS01ES156M",   # Monthly
    ],
    "Sweden": [
        "IR3TTS01SEM156N",  # Swedish 3-month Treasury Bill (known working)
    ],
    "UnitedKingdom": [
        "IR3TTS01GBM156N",  # UK 3-month Treasury Bill (ends 2017)
        "IR3TCD01GBM156N",  # UK 3-month CD rate
        "IR3TTS01GB156N",   # Shorter format
        "IR3TTS01GB156M",   # Monthly
        "IR3TTS01GB156Q",   # Quarterly
    ],
    "Switzerland": [
        "IR3TTS01CHM156N",  # Swiss 3-month Treasury Bill
        "IR3TCD01CHM156N",  # Swiss 3-month CD rate
        "IR3TTS01CH156N",   # Shorter format
        "IR3TTS01CH156M",   # Monthly
    ],
}

# FRED CSV download URL formats to test
FRED_URL_FORMATS = [
    lambda sid: f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
    lambda sid: f"https://fred.stlouisfed.org/data/{sid}.csv",
]


def test_fred_series_id(series_id: str, start_date: str, end_date: str) -> Optional[Tuple[pd.Series, str]]:
    """
    Test if a FRED series ID is valid and has data in the required date range.
    
    Parameters
    ----------
    series_id : str
        FRED series ID to test
    start_date : str
        Required start date (YYYY-MM-DD)
    end_date : str
        Required end date (YYYY-MM-DD)
    
    Returns
    -------
    tuple or None
        (Series, URL) if successful, None otherwise
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    for url_format in FRED_URL_FORMATS:
        url = url_format(series_id)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Try to parse CSV
                try:
                    df = pd.read_csv(
                        StringIO(response.text),
                        parse_dates=['observation_date'],
                        na_values=['.', 'ND', '']
                    )
                    
                    if df.empty:
                        continue
                    
                    # Get value column (second column)
                    value_col = df.columns[1]
                    df = df.rename(columns={'observation_date': 'date', value_col: 'value'})
                    
                    # Filter date range
                    df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
                    df = df.dropna(subset=['value'])
                    
                    if df.empty:
                        # Check if data exists but outside range
                        max_date = pd.to_datetime(df['date']).max() if len(df) > 0 else None
                        if max_date and max_date < start_dt:
                            logger.debug(f"Series {series_id} exists but ends before {start_date}")
                        continue
                    
                    # Convert to Series
                    series = df.set_index('date')['value']
                    series.name = 'riskfree_rate'
                    
                    logger.info(f" Found working series: {series_id}")
                    logger.info(f"   URL: {url}")
                    logger.info(f"   Data points: {len(series)}")
                    logger.info(f"   Date range: {series.index.min()} to {series.index.max()}")
                    
                    return (series, url)
                    
                except Exception as e:
                    logger.debug(f"Failed to parse CSV for {series_id}: {e}")
                    continue
                    
        except requests.exceptions.RequestException as e:
            logger.debug(f"Request failed for {series_id} ({url}): {e}")
            continue
    
    return None


def scrape_trading_economics(country: str, start_date: str, end_date: str) -> Optional[pd.Series]:
    """
    Scrape 3-month government bond yield from Trading Economics.
    
    Note: This is a fallback when FRED doesn't have data.
    Trading Economics provides historical data but may require subscription for full access.
    
    Parameters
    ----------
    country : str
        Country name
    start_date : str
        Start date
    end_date : str
        End date
    
    Returns
    -------
    pd.Series or None
        Risk-free rate series
    """
    # Trading Economics country slugs
    country_slugs = {
        "Germany": "germany",
        "UnitedKingdom": "united-kingdom",
        "Switzerland": "switzerland",
    }
    
    slug = country_slugs.get(country)
    if not slug:
        return None
    
    url = f"https://tradingeconomics.com/{slug}/3-month-bill-yield"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            # Trading Economics may require JavaScript rendering or subscription
            # For now, log that we tried
            logger.info(f"Trading Economics page accessible for {country}")
            logger.warning("Trading Economics may require subscription or JavaScript rendering")
            logger.info("Manual download recommended from: " + url)
            return None
        else:
            logger.warning(f"Trading Economics returned status {response.status_code}")
            return None
            
    except Exception as e:
        logger.warning(f"Failed to access Trading Economics for {country}: {e}")
        return None


def scrape_alternative_source(country: str, start_date: str, end_date: str) -> Optional[pd.Series]:
    """
    Fallback: Scrape data from alternative sources when FRED doesn't have it.
    
    Tries multiple sources in order:
    1. Trading Economics (if available)
    2. Central bank websites (if downloadable)
    
    Parameters
    ----------
    country : str
        Country name
    start_date : str
        Start date
    end_date : str
        End date
    
    Returns
    -------
    pd.Series or None
        Risk-free rate series
    """
    logger.info(f"Attempting to scrape alternative source for {country}...")
    
    # Try Trading Economics
    te_data = scrape_trading_economics(country, start_date, end_date)
    if te_data is not None and not te_data.empty:
        return te_data
    
    # If no alternative source works, provide instructions
    logger.warning(f"  No alternative source available for {country}")
    logger.info("Manual data download required from:")
    if country == "Germany":
        logger.info("  - Bundesbank: https://www.bundesbank.de/en/statistics/money-and-capital-markets")
        logger.info("  - ECB: https://data.ecb.europa.eu/")
        logger.info("  - Trading Economics: https://tradingeconomics.com/germany/3-month-bill-yield")
    elif country == "UnitedKingdom":
        logger.info("  - Bank of England: https://www.bankofengland.co.uk/statistics/yield-curves")
        logger.info("  - Trading Economics: https://tradingeconomics.com/united-kingdom/3-month-bill-yield")
    elif country == "Switzerland":
        logger.info("  - SNB: https://www.snb.ch/en/statistics")
        logger.info("  - Trading Economics: https://tradingeconomics.com/switzerland/3-month-bill-yield")
    
    logger.info("\nRECOMMENDED: Get free FRED API key to access more series:")
    logger.info("  https://fred.stlouisfed.org/docs/api/api_key.html")
    
    return None


def find_fred_series_id(country: str, start_date: str, end_date: str) -> Optional[Tuple[pd.Series, str]]:
    """
    Research and find working FRED series ID for a country.
    
    Parameters
    ----------
    country : str
        Country name
    start_date : str
        Required start date
    end_date : str
        Required end date
    
    Returns
    -------
    tuple or None
        (Series, URL) if found, None otherwise
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Researching FRED series ID for: {country}")
    logger.info(f"{'='*70}")
    
    # Check known working series first
    if country in KNOWN_SERIES_IDS:
        series_id = KNOWN_SERIES_IDS[country]
        result = test_fred_series_id(series_id, start_date, end_date)
        if result:
            return result
    
    # Test patterns
    patterns = SERIES_ID_PATTERNS.get(country, [])
    for series_id in patterns:
        if series_id in KNOWN_SERIES_IDS.get(country, ""):
            continue  # Already tested
        
        logger.debug(f"Testing series ID: {series_id}")
        result = test_fred_series_id(series_id, start_date, end_date)
        if result:
            return result
    
    logger.warning(f" No working FRED series ID found for {country}")
    
    # Try alternative source as fallback
    logger.info(f"Trying alternative data source for {country}...")
    alt_series = scrape_alternative_source(country, start_date, end_date)
    if alt_series is not None:
        logger.info(f" Found data from alternative source for {country}")
        return (alt_series, "alternative_source")
    
    return None


# -------------------------------------------------------------------
# DATA DOWNLOAD AND PROCESSING
# -------------------------------------------------------------------

def download_fred_series(series_id: str, url: str, start_date: str, end_date: str) -> pd.Series:
    """
    Download and process FRED series data.
    
    Parameters
    ----------
    series_id : str
        FRED series ID
    url : str
        CSV download URL
    start_date : str
        Start date filter
    end_date : str
        End date filter
    
    Returns
    -------
    pd.Series
        Monthly risk-free rate in percentage form (%)
    """
    logger.info(f"Downloading {series_id} from FRED...")
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # Parse CSV
    df = pd.read_csv(
        StringIO(response.text),
        parse_dates=['observation_date'],
        na_values=['.', 'ND', '']
    )
    
    # Get value column
    value_col = df.columns[1]
    df = df.rename(columns={'observation_date': 'date', value_col: 'value'})
    
    # Filter date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
    df = df.dropna(subset=['value'])
    
    # Convert to Series
    series = df.set_index('date')['value']
    series.name = 'riskfree_rate'
    
    # FRED data is typically annual percentage
    # Convert to monthly: monthly = annual / 12
    series_monthly = series / 12.0
    
    # Resample to month-end if needed
    series_monthly = series_monthly.resample('ME').last()
    
    logger.info(f" Processed {len(series_monthly)} months of data")
    logger.info(f"   Rate range: {series_monthly.min():.4f}% to {series_monthly.max():.4f}%")
    
    return series_monthly


# -------------------------------------------------------------------
# COUNTRY MAPPING AND EUR HANDLING
# -------------------------------------------------------------------

def get_country_riskfree_rate(
    country: str,
    start_date: str,
    end_date: str,
    country_rates: Dict[str, Tuple[pd.Series, str]]
) -> Optional[pd.Series]:
    """
    Get risk-free rate for a country.
    
    All countries use German Bund rate (EUR) since all stock returns and market returns
    are converted to EUR. Interest rates are percentages and should NOT be multiplied
    by exchange rates.
    
    Parameters
    ----------
    country : str
        Country name (all countries will receive German Bund rate)
    start_date : str
        Start date
    end_date : str
        End date
    country_rates : dict
        Dictionary of already-fetched rates
    
    Returns
    -------
    pd.Series or None
        German Bund rate series (EUR) for all countries
    """
    # All countries use German Bund rate (EUR)
    # Since all stock returns and market returns are converted to EUR,
    # all countries should use the same EUR risk-free rate
    if "Germany" in country_rates:
        logger.info(f"{country} - using German Bund rate (EUR)")
        return country_rates["Germany"][0].copy()
    else:
        # Try to fetch German rate
        result = find_fred_series_id("Germany", start_date, end_date)
        if result:
            series, url = result
            country_rates["Germany"] = (series, url)
            logger.info(f"{country} - using German Bund rate (EUR)")
            return series.copy()
        else:
            logger.warning(f"German Bund rate not found - {country} will use placeholder")
            return None


# -------------------------------------------------------------------
# MAIN PROCESSING FUNCTION
# -------------------------------------------------------------------

def download_and_merge_riskfree_rates() -> pd.DataFrame:
    """
    Main function: Download FRED data and merge with returns panel.
    
    Returns
    -------
    pd.DataFrame
        Merged panel with real risk-free rates
    """
    logger.info("="*70)
    logger.info("DOWNLOADING RISK-FREE RATES FROM FRED (NO API KEY)")
    logger.info("="*70)
    
    start_date = ANALYSIS_SETTINGS.start_date
    end_date = ANALYSIS_SETTINGS.end_date
    
    # Step 1: Research and download rates for each country
    logger.info("\nStep 1: Researching and downloading FRED series...")
    country_rates = {}
    
    # First, try to get German rate (needed for EUR countries)
    logger.info("\nPriority: Fetching German Bund rate (needed for EUR countries)...")
    result = find_fred_series_id("Germany", start_date, end_date)
    if result:
        series, url = result
        country_rates["Germany"] = (series, url)
        # Save to raw data
        series.to_csv(os.path.join(DATA_RAW_DIR, "riskfree_rate_Germany.csv"))
        logger.info(f" Saved German rate to data/raw/riskfree_rate_Germany.csv")
    else:
        logger.warning("  German Bund rate not found in FRED - EUR countries will need alternative source")
    
    # All countries use German Bund rate (EUR)
    # Since all stock returns and market returns are converted to EUR,
    # all countries should use the same EUR risk-free rate
    logger.info("\nAll countries will use German Bund rate (EUR)")
    for country in COUNTRIES.keys():
        if country == "Germany":
            continue  # Already fetched
        logger.info(f"{country} - will use German Bund rate (EUR)")
    
    # Step 2: Create mapping for all countries (all use German Bund)
    logger.info("\nStep 2: Creating country-to-rate mapping (all countries use German Bund)...")
    country_to_rate = {}
    
    # Get German Bund rate for all countries
    german_rate = None
    if "Germany" in country_rates:
        german_rate = country_rates["Germany"][0]
        logger.info(f" German Bund rate: {len(german_rate)} months")
    
    if german_rate is None:
        raise ValueError("German Bund rate not found - required for all countries")
    
    # All countries use German Bund rate
    for country in COUNTRIES.keys():
        country_to_rate[country] = german_rate
        logger.info(f" {country}: Using German Bund rate ({len(german_rate)} months)")
    
    if not country_to_rate:
        raise ValueError("No risk-free rates found for any country!")
    
    # Step 3: Load returns panel
    logger.info("\nStep 3: Loading returns panel...")
    panel_path = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"Returns panel not found: {panel_path}")
    
    panel = pd.read_csv(panel_path, parse_dates=['date'])
    logger.info(f" Loaded panel: {len(panel):,} rows")
    
    # Step 4: Merge risk-free rates
    logger.info("\nStep 4: Merging risk-free rates with panel...")
    
    # Create mapping DataFrame
    rf_mapping_list = []
    for country, series in country_to_rate.items():
        for date, rate in series.items():
            rf_mapping_list.append({
                'date': date,
                'country': country,
                'riskfree_rate_real': rate
            })
    
    rf_mapping = pd.DataFrame(rf_mapping_list)
    logger.info(f" Created mapping: {len(rf_mapping)} rows")
    
    # Convert panel dates to month-end for matching
    panel['date_month_end'] = pd.to_datetime(panel['date']).dt.to_period('M').dt.to_timestamp('M')
    rf_mapping['date_month_end'] = pd.to_datetime(rf_mapping['date']).dt.to_period('M').dt.to_timestamp('M')
    
    # Merge
    panel_merged = panel.merge(
        rf_mapping[['date_month_end', 'country', 'riskfree_rate_real']],
        on=['date_month_end', 'country'],
        how='left'
    )
    
    # Replace old risk-free rate with real one (forward fill if needed)
    panel_merged['riskfree_rate'] = panel_merged['riskfree_rate_real'].fillna(panel_merged['riskfree_rate'])
    
    # Drop temporary columns
    panel_merged = panel_merged.drop(columns=['date_month_end', 'riskfree_rate_real'])
    
    # Step 5: Recalculate excess returns
    logger.info("\nStep 5: Recalculating excess returns...")
    panel_merged['stock_excess_return'] = panel_merged['stock_return'] - panel_merged['riskfree_rate']
    panel_merged['market_excess_return'] = panel_merged['msci_index_return'] - panel_merged['riskfree_rate']
    
    # Drop rows where excess returns can't be calculated
    initial_rows = len(panel_merged)
    panel_merged = panel_merged.dropna(subset=['riskfree_rate', 'stock_excess_return', 'market_excess_return'])
    dropped_rows = initial_rows - len(panel_merged)
    
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows with missing risk-free rates")
    
    # Step 6: Save merged panel
    output_path = os.path.join(DATA_PROCESSED_DIR, "returns_panel_with_real_rates.csv")
    logger.info(f"\nStep 6: Saving merged panel to: {output_path}")
    panel_merged.to_csv(output_path, index=False)
    logger.info(f" Saved {len(panel_merged):,} rows")
    
    return panel_merged


# -------------------------------------------------------------------
# VERIFICATION
# -------------------------------------------------------------------

def verify_results(panel: pd.DataFrame) -> None:
    """
    Print verification statistics.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Merged panel DataFrame
    """
    print("\n" + "="*70)
    print("VERIFICATION: Merged Panel with Real Risk-Free Rates")
    print("="*70)
    
    print(f"\n Panel Summary:")
    print(f"   Total rows: {len(panel):,}")
    print(f"   Countries: {panel['country'].nunique()}")
    print(f"   Stocks: {panel['ticker'].nunique()}")
    print(f"   Date range: {panel['date'].min()} to {panel['date'].max()}")
    
    print(f"\n Risk-Free Rates by Country:")
    for country in sorted(panel['country'].unique()):
        country_data = panel[panel['country'] == country]
        rf_rates = country_data['riskfree_rate'].dropna()
        if len(rf_rates) > 0:
            print(f"   {country:15s}: {rf_rates.min():.4f}% to {rf_rates.max():.4f}% (mean: {rf_rates.mean():.4f}%)")
            print(f"                  Coverage: {len(rf_rates)}/{len(country_data)} rows ({len(rf_rates)/len(country_data)*100:.1f}%)")
        else:
            print(f"   {country:15s}:   No data")
    
    print(f"\n Data Quality:")
    missing_rf = panel['riskfree_rate'].isna().sum()
    missing_excess = panel[['stock_excess_return', 'market_excess_return']].isna().sum().sum()
    
    print(f"   Missing risk-free rates: {missing_rf} ({missing_rf/len(panel)*100:.2f}%)")
    print(f"   Missing excess returns: {missing_excess}")
    
    if missing_rf == 0 and missing_excess == 0:
        print("    No missing values!")
    else:
        print("     Some missing values detected")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        # Download and merge
        merged_panel = download_and_merge_riskfree_rates()
        
        # Verify
        verify_results(merged_panel)
        
        print("\n Successfully downloaded and merged real risk-free rates!")
        print(f"   Output: {DATA_PROCESSED_DIR}/returns_panel_with_real_rates.csv")
        
    except Exception as e:
        logger.error(f"Failed to download and merge risk-free rates: {e}")
        import traceback
        logger.exception("Full error details:")
        raise

