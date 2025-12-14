"""
riskfree_helper.py

Functions to fetch and process risk-free rates (3-month government bonds)
for CAPM analysis.

Per methodology:
- All countries use German 3-month Bund (EUR) as the risk-free rate
- Since all stock returns and market returns are converted to EUR,
  all countries should use the same EUR risk-free rate (German Bund)
- Interest rates are percentages and should NOT be multiplied by exchange rates

Data Sources (in order of preference):
1. CSV files (processed risk-free rate files, primary source)
2. ECB API - For EUR countries (free, no API key required)
3. FRED API - For all countries (free, requires API key)
4. WRDS - For academic users with WRDS access
5. Yahoo Finance - Limited availability (fallback)

Note: System requires real data - CSV files are available for all countries.
No placeholder values are used.
"""

import logging
import os
from typing import Optional

import pandas as pd
import yfinance as yf

from analysis.utils.config import (
    COUNTRIES,
    FRED_SERIES_CODES,
    ECB_SERIES_KEYS,
    RISKFREE_SOURCE_ORDER,
    DATA_RAW_DIR,
)
from analysis.utils.env_loader import load_env

# Load environment variables from .env file if available
load_env()

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# RISK-FREE RATE CONVERSION
# -------------------------------------------------------------------

def convert_annual_to_monthly_rate(annual_rate: float) -> float:
    """
    Convert annual risk-free rate to monthly rate.
    
    Formula: R_monthly = (1 + R_annual)^(1/12) - 1
    
    Parameters
    ----------
    annual_rate : float
        Annual rate in decimal form (e.g., 0.05 for 5%)
    
    Returns
    -------
    float
        Monthly rate in decimal form
    """
    if annual_rate <= -1:
        logger.warning(f"Invalid annual rate: {annual_rate}, returning 0")
        return 0.0
    
    monthly_rate = (1 + annual_rate) ** (1/12) - 1
    return monthly_rate


def convert_annual_pct_to_monthly_pct(annual_rate_pct: float) -> float:
    """
    Convert annual rate in percentage form to monthly rate in percentage form.
    
    Parameters
    ----------
    annual_rate_pct : float
        Annual rate in percentage form (e.g., 5.0 for 5%)
    
    Returns
    -------
    float
        Monthly rate in percentage form
    """
    annual_decimal = annual_rate_pct / 100.0
    monthly_decimal = convert_annual_to_monthly_rate(annual_decimal)
    return monthly_decimal * 100.0


# -------------------------------------------------------------------
# ECB API FETCHING (For EUR Countries)
# -------------------------------------------------------------------

def fetch_riskfree_from_ecb(
    country: str,
    start_date: str,
    end_date: str
) -> Optional[pd.Series]:
    """
    Fetch 3-month government bond yield from ECB Statistical Data Warehouse.
    
    Works for EUR countries (Germany, France, Italy, Spain).
    Uses ECB SDMX RESTful API (free, no API key required).
    
    Note: ECB API structure is complex. This implementation uses a simplified approach.
    For production use, consider using the ecbdata or sdw-api Python packages.
    
    Parameters
    ----------
    country : str
        Country name
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns
    -------
    pd.Series or None
        Monthly risk-free rate in percentage form (%), or None if failed
    """
    # Only EUR countries use ECB
    if COUNTRIES[country].currency != "EUR":
        logger.debug(f"ECB API only for EUR countries, skipping {country}")
        return None
    
    # All EUR countries use German 3-month Bund
    if country != "Germany":
        logger.info(f"Using German 3-month Bund for {country} (EUR)")
        return fetch_riskfree_from_ecb("Germany", start_date, end_date)
    
    try:
        # Try using pandas_datareader for ECB data (simpler)
        try:
            import pandas_datareader.data as web
            logger.info("Attempting to fetch ECB data via pandas_datareader")
            # Note: This may not work directly - ECB data access via pandas_datareader is limited
            # For now, return None to try other sources
            return None
        except ImportError:
            pass
        
        # Alternative: Direct ECB API call (complex structure)
        import requests
        
        # ECB provides data via SDMX API
        # For 3-month government bond yields, the series key format is complex
        # Using a simplified approach - try to download CSV directly
        # ECB Data Portal: https://data.ecb.europa.eu/data
        
        # For now, ECB API implementation is complex and may require specific series keys
        # Returning None to fall back to other sources
        logger.debug("ECB API requires specific series keys - using fallback sources")
        return None
        
    except Exception as e:
        logger.debug(f"ECB API fetch failed for {country}: {e}")
        return None


# -------------------------------------------------------------------
# FRED API FETCHING
# -------------------------------------------------------------------

def fetch_riskfree_from_fred(
    country: str,
    start_date: str,
    end_date: str
) -> Optional[pd.Series]:
    """
    Fetch 3-month government bond yield from FRED API.
    
    Requires FRED API key set in environment variable FRED_API_KEY.
    Get free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
    
    Parameters
    ----------
    country : str
        Country name
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns
    -------
    pd.Series or None
        Monthly risk-free rate in percentage form (%), or None if failed
    """
    try:
        from fredapi import Fred
    except ImportError:
        logger.warning("fredapi package not installed - install with: pip install fredapi")
        return None
    
    # Determine target country (EUR countries use Germany)
    if COUNTRIES[country].currency == "EUR":
        target_country = "Germany"
    else:
        target_country = country
    
    # Get FRED series code
    series_code = FRED_SERIES_CODES.get(target_country)
    if not series_code:
        logger.warning(f"No FRED series code configured for {target_country}")
        return None
    
    try:
        # Get API key from environment
        api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            logger.warning(
                "FRED_API_KEY environment variable not set. "
                "Get free API key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            return None
        
        # Initialize FRED client
        fred = Fred(api_key=api_key)
        
        logger.info(f"Fetching FRED series {series_code} for {target_country}")
        
        # Fetch data
        data = fred.get_series(series_code, start=start_date, end=end_date)
        
        if data is None or data.empty:
            logger.warning(f"FRED returned no data for series {series_code}")
            return None
        
        # FRED returns annual rates in percentage form
        # Convert to monthly percentage
        monthly_rates = data.apply(convert_annual_pct_to_monthly_pct)
        
        # Resample to month-end
        monthly_rates = monthly_rates.resample('ME').last()
        
        # Rename
        monthly_rates.name = 'riskfree_rate'
        
        logger.info(f"✅ FRED: Fetched {len(monthly_rates)} months of data for {target_country}")
        return monthly_rates
        
    except Exception as e:
        logger.warning(f"FRED API fetch failed for {target_country}: {e}")
        return None


# -------------------------------------------------------------------
# WRDS FETCHING
# -------------------------------------------------------------------

def fetch_riskfree_from_wrds(
    country: str,
    start_date: str,
    end_date: str
) -> Optional[pd.Series]:
    """
    Fetch 3-month government bond yield from WRDS.
    
    Requires WRDS access. Credentials can be provided via:
    1. Environment variables: WRDS_USERNAME and WRDS_PASSWORD
    2. .pgpass file in home directory
    3. WRDS config file: ~/.wrdsrc
    
    Parameters
    ----------
    country : str
        Country name
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns
    -------
    pd.Series or None
        Monthly risk-free rate in percentage form (%), or None if failed
    """
    try:
        import wrds
        
        # Determine target country (EUR countries use Germany)
        if COUNTRIES[country].currency == "EUR":
            target_country = "Germany"
        else:
            target_country = country
        
        # Get WRDS credentials
        username = os.getenv('WRDS_USERNAME')
        password = os.getenv('WRDS_PASSWORD')
        
        logger.info(f"Connecting to WRDS to fetch data for {target_country}")
        
        # Try to connect - WRDS will use .pgpass if available, or prompt
        # For non-interactive use, credentials should be in .pgpass file
        try:
            if username and password:
                # Try with credentials
                db = wrds.Connection(wrds_username=username, wrds_password=password)
            else:
                # Try without (will use .pgpass or prompt)
                db = wrds.Connection()
        except Exception as e:
            logger.warning(f"WRDS connection failed: {e}")
            logger.info("To use WRDS, set up .pgpass file or set WRDS_USERNAME/WRDS_PASSWORD")
            return None
        
        # Try Federal Reserve Board H.15 data (has international rates)
        try:
            # Query for 3-month Treasury bills
            # FRED data in WRDS: fred.tbill3m (US) or similar
            # For international, may need different table
            query = f"""
            SELECT date, value AS tbill3m
            FROM fred.tbill3m
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date
            """
            
            df = db.raw_sql(query, date_cols=['date'])
            db.close()
            
            if df.empty:
                logger.warning("WRDS query returned no data - trying alternative tables")
                # Try CRSP Treasury if available
                try:
                    query2 = f"""
                    SELECT caldt AS date, tbill3m AS value
                    FROM crsp.treasury
                    WHERE caldt >= '{start_date}' AND caldt <= '{end_date}'
                    ORDER BY caldt
                    """
                    df = db.raw_sql(query2, date_cols=['date'])
                    db.close()
                except:
                    pass
            
            if df.empty:
                logger.warning("WRDS query returned no data")
                return None
            
            # Convert to Series
            df['date'] = pd.to_datetime(df['date'])
            series = df.set_index('date')['tbill3m']
            series.name = 'riskfree_rate'
            
            # Convert to monthly percentage (assuming annual percentage)
            series = series.apply(convert_annual_pct_to_monthly_pct)
            
            # Resample to month-end
            series = series.resample('ME').last()
            
            logger.info(f"✅ WRDS: Fetched {len(series)} months of data for {target_country}")
            return series
            
        except Exception as e:
            logger.warning(f"WRDS query failed: {e}")
            try:
                db.close()
            except:
                pass
            return None
            
    except ImportError:
        logger.warning("wrds package not installed - cannot use WRDS")
        return None
    except Exception as e:
        logger.warning(f"WRDS connection failed: {e}")
        return None


# -------------------------------------------------------------------
# YAHOO FINANCE FETCHING (Limited)
# -------------------------------------------------------------------

def fetch_riskfree_from_yfinance(
    country: str,
    start_date: str,
    end_date: str
) -> Optional[pd.Series]:
    """
    Fetch 3-month government bond yield from Yahoo Finance.
    
    Note: Yahoo Finance has very limited international government bond data.
    This is primarily a fallback option.
    
    Parameters
    ----------
    country : str
        Country name
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns
    -------
    pd.Series or None
        Monthly risk-free rate in percentage form (%), or None if failed
    """
    # Yahoo Finance has limited international bond data
    # For now, this is a placeholder that returns None
    # In the future, could try to find country-specific bond ETFs
    
    logger.warning("Yahoo Finance has limited international government bond data")
    return None


# -------------------------------------------------------------------
# MAIN RISK-FREE RATE FETCHER
# -------------------------------------------------------------------

def load_riskfree_from_csv(
    country: str,
    start_date: str,
    end_date: str
) -> Optional[pd.Series]:
    """
    Load risk-free rate from processed CSV file.
    
    Parameters
    ----------
    country : str
        Country name
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns
    -------
    pd.Series or None
        Monthly risk-free rate in percentage form (%), or None if file not found
    """
    csv_path = os.path.join(DATA_RAW_DIR, "riskfree_rates", f"riskfree_rate_{country}.csv")
    
    if not os.path.exists(csv_path):
        logger.debug(f"CSV file not found: {csv_path}")
        return None
    
    try:
        # Load CSV (index should be dates, column should be monthly_rate_pct)
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        # Get the rate column (should be 'monthly_rate_pct')
        if 'monthly_rate_pct' in df.columns:
            rate_col = 'monthly_rate_pct'
        else:
            # Fallback: use first column
            rate_col = df.columns[0]
        
        series = df[rate_col].copy()
        series.name = 'riskfree_rate'
        
        # Filter date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        series = series[(series.index >= start_dt) & (series.index <= end_dt)]
        
        if len(series) == 0:
            logger.warning(f"No data in date range for {country} from CSV")
            return None
        
        logger.info(f"✅ Loaded risk-free rate from CSV for {country}: {len(series)} months")
        return series
        
    except Exception as e:
        logger.warning(f"Error loading CSV for {country}: {e}")
        return None


def get_riskfree_rate(
    country: str,
    start_date: str,
    end_date: str,
    source: Optional[str] = None
) -> pd.Series:
    """
    Fetch risk-free rate for a given country using best available source.
    
    Tries sources in order: CSV -> ECB -> FRED -> WRDS -> yfinance
    System requires real data - raises ValueError if all sources fail (no placeholder)
    
    Per methodology:
    - All countries use German 3-month Bund (EUR) as the risk-free rate
    - Since all stock returns and market returns are converted to EUR,
      all countries should use the same EUR risk-free rate (German Bund)
    - Interest rates are percentages and should NOT be multiplied by exchange rates
    
    Parameters
    ----------
    country : str
        Country name (must match COUNTRIES keys)
        Note: All countries will receive German Bund rate regardless of input country
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    source : str, optional
        Specific source to use ('csv', 'ecb', 'fred', 'wrds', 'yfinance')
        If None, tries sources in order until one succeeds
    
    Returns
    -------
    pandas.Series
        Monthly risk-free rate in percentage form (%)
        Index: dates (monthly, month-end)
        Values: risk-free rate (%)
        Always returns German Bund rate (EUR) for all countries
    
    Raises
    ------
    ValueError
        If all data sources fail and no risk-free rate can be obtained
    """
    if country not in COUNTRIES:
        raise ValueError(f"Unknown country: {country}. Must be one of {list(COUNTRIES.keys())}")
    
    # All countries use German Bund rate (EUR)
    # Since all stock returns and market returns are converted to EUR,
    # all countries should use the same EUR risk-free rate
    target_country = "Germany"
    logger.info(f"Using German 3-month Bund (EUR) for {country} - all countries use same EUR risk-free rate")
    
    # Create monthly date range (fallback)
    date_range = pd.date_range(start=start_date, end=end_date, freq='ME')
    
    # FIRST: Try loading from CSV (processed files)
    if source is None or source == "csv":
        result = load_riskfree_from_csv(target_country, start_date, end_date)
        if result is not None and not result.empty:
            return result
    
    # If specific source requested and it's not CSV, try only that source
    if source and source != "csv":
        sources_to_try = [source]
    else:
        sources_to_try = RISKFREE_SOURCE_ORDER
    
    # Try each source until one succeeds
    for src in sources_to_try:
        logger.debug(f"Trying source: {src} for {target_country}")
        
        if src == "ecb":
            result = fetch_riskfree_from_ecb(target_country, start_date, end_date)
            if result is not None and not result.empty:
                return result
        
        elif src == "fred":
            result = fetch_riskfree_from_fred(target_country, start_date, end_date)
            if result is not None and not result.empty:
                return result
        
        elif src == "wrds":
            result = fetch_riskfree_from_wrds(target_country, start_date, end_date)
            if result is not None and not result.empty:
                return result
        
        elif src == "yfinance":
            result = fetch_riskfree_from_yfinance(target_country, start_date, end_date)
            # fetch_riskfree_from_yfinance returns Optional[pd.Series], check properly
            if result is not None and not result.empty:
                return result
    
    # If all sources failed, raise error (no placeholder fallback)
    error_msg = (
        f"All data sources failed for {target_country} ({country}). "
        f"Tried: CSV, ECB, FRED, WRDS, yfinance. "
        f"Please ensure risk-free rate data is available in data/raw/riskfree_rates/ "
        f"or set up FRED_API_KEY/WRDS credentials. "
        f"System requires real data - no placeholder values allowed."
    )
    logger.error(error_msg)
    raise ValueError(error_msg)


def align_riskfree_with_returns(
    riskfree_series: pd.Series,
    returns_dates: pd.DatetimeIndex
) -> pd.Series:
    """
    Align risk-free rate series with return dates.
    
    Parameters
    ----------
    riskfree_series : pd.Series
        Risk-free rate series with date index
    returns_dates : pd.DatetimeIndex
        Dates from returns data
    
    Returns
    -------
    pd.Series
        Aligned risk-free rate series matching returns_dates
    """
    # Reindex to match returns dates
    aligned = riskfree_series.reindex(returns_dates)
    
    # Forward fill if needed (risk-free rates change less frequently)
    aligned = aligned.ffill()
    
    # If still missing at start, backward fill
    aligned = aligned.bfill()
    
    return aligned


if __name__ == "__main__":
    # Test risk-free rate fetching
    from analysis.utils.config import ANALYSIS_SETTINGS
    
    print("Testing risk-free rate functions:")
    print("="*70)
    
    for country in COUNTRIES.keys():
        try:
            rf = get_riskfree_rate(
                country,
                ANALYSIS_SETTINGS.start_date,
                ANALYSIS_SETTINGS.end_date
            )
            print(f"{country}: {len(rf)} months, range: {rf.index.min()} to {rf.index.max()}")
            print(f"  Sample values: {rf.head(3).tolist()}")
        except Exception as e:
            print(f"{country}: Error - {e}")
