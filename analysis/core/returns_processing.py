"""
Returns Processing Module for CAPM Analysis.

This module handles the conversion of raw price data into monthly excess returns
suitable for Capital Asset Pricing Model (CAPM) regressions.

The processing follows standard academic methodology:
    - Simple returns: R_t = [(P_t - P_{t-1}) / P_{t-1}] x 100 (percentage form)
    - Market proxy: MSCI Europe Index (single pan-European benchmark)
    - Risk-free rate: German 3-month Treasury Bill (Bund)
    - Estimation window: 60 months (January 2018 - December 2023)
    - Minimum observations: 59 monthly returns required per stock

The module ensures data quality through:
    - Handling of missing values and outliers
    - Currency consistency (EUR-denominated returns)
    - Date alignment across stocks and market indices
    - Proper calculation of excess returns (stock return minus risk-free rate)

References
----------
Fama, E. F., & French, K. R. (1993). Common Risk Factors in the Returns on 
    Stocks and Bonds. Journal of Financial Economics, 33(1), 3-56.
"""

import os
import logging
# Note: Optional not currently used but kept for type hints

import pandas as pd

from analysis.utils.config import (
    ANALYSIS_SETTINGS,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_EXCHANGE_DIR,
    COUNTRIES,
)
from analysis.utils.universe import load_stock_universe
from analysis.data.riskfree_helper import get_riskfree_rate, align_riskfree_with_returns

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------

MIN_OBSERVATIONS = 59  # 59 months of returns (60 months of prices = 59 returns)
MAX_MISSING_PCT = 0.10  # Drop stocks with >10% missing data


# -------------------------------------------------------------------
# PRICE TO RETURNS CONVERSION
# -------------------------------------------------------------------

def prices_to_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert prices to simple monthly returns in percentage form.
    
    Formula: R_t = [(P_t - P_{t-1}) / P_{t-1}] × 100 (%)
    
    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame with prices, index=dates, columns=tickers
    
    Returns
    -------
    pd.DataFrame
        Returns in percentage form (%), same structure as input
    """
    if prices_df.empty:
        logger.warning("Empty prices DataFrame provided")
        return pd.DataFrame()
    
    # Ensure datetime index
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df = prices_df.sort_index()
    
    # Calculate returns: (P_t / P_{t-1} - 1) × 100
    returns = prices_df.pct_change() * 100
    
    # First row will be NaN (no previous price), drop it
    returns = returns.dropna(how='all')
    
    logger.info(f"Converted prices to returns: {returns.shape}")
    logger.info(f"  Date range: {returns.index.min()} to {returns.index.max()}")
    
    return returns


def filter_stocks_by_observations(
    returns_df: pd.DataFrame,
    min_observations: int = MIN_OBSERVATIONS
) -> pd.DataFrame:
    """
    Drop stocks with insufficient observations.
    
    Note: 60 months of prices = 59 months of returns (first month has no return).
    So we require 59 months of returns for the full period.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Returns DataFrame with dates as index, tickers as columns
    min_observations : int
        Minimum number of observations required (default: 59)
    
    Returns
    -------
    pd.DataFrame
        Returns DataFrame with stocks having >= min_observations
    """
    if returns_df.empty:
        return returns_df
    
    original_count = len(returns_df.columns)
    
    # Count valid (non-NaN) observations per stock
    valid_counts = returns_df.count()
    
    # Filter stocks with sufficient observations
    valid_stocks = valid_counts >= min_observations
    filtered_df = returns_df.loc[:, valid_stocks]
    
    dropped_count = original_count - len(filtered_df.columns)
    if dropped_count > 0:
        dropped_stocks = returns_df.columns[~valid_stocks].tolist()
        logger.warning(
            f"Dropped {dropped_count} stocks with < {min_observations} observations: "
            f"{dropped_stocks[:10]}{'...' if len(dropped_stocks) > 10 else ''}"
        )
    
    logger.info(f"Filtered stocks: {original_count} → {len(filtered_df.columns)}")
    
    return filtered_df


def handle_missing_values(returns_df: pd.DataFrame, max_missing_pct: float = MAX_MISSING_PCT) -> pd.DataFrame:
    """
    Handle missing values in returns DataFrame.
    
    Strategy:
    1. Forward fill up to 2 months (handles temporary data gaps)
    2. Drop stocks with >max_missing_pct missing data after forward fill
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Returns DataFrame with dates as index, tickers as columns
    max_missing_pct : float
        Maximum percentage of missing data allowed (default: 0.10 = 10%)
    
    Returns
    -------
    pd.DataFrame
        Returns DataFrame with missing values handled
    """
    """
    Handle missing values in returns data.
    
    Strategy:
    - Forward fill up to 2 consecutive months
    - Drop stocks with >max_missing_pct missing data
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Returns DataFrame
    max_missing_pct : float
        Maximum percentage of missing data allowed (default: 10%)
    
    Returns
    -------
    pd.DataFrame
        Returns DataFrame with missing values handled
    """
    if returns_df.empty:
        return returns_df
    
    original_count = len(returns_df.columns)
    
    # Forward fill up to 2 consecutive months
    # (limited forward fill to avoid filling large gaps)
    filled_df = returns_df.ffill(limit=2)
    
    # Calculate missing percentage per stock
    missing_pct = filled_df.isna().sum() / len(filled_df) * 100
    
    # Drop stocks with excessive missing data
    valid_stocks = missing_pct <= max_missing_pct * 100
    cleaned_df = filled_df.loc[:, valid_stocks]
    
    dropped_count = original_count - len(cleaned_df.columns)
    if dropped_count > 0:
        dropped_stocks = filled_df.columns[~valid_stocks].tolist()
        logger.warning(
            f"Dropped {dropped_count} stocks with >{max_missing_pct*100}% missing data: "
            f"{dropped_stocks[:10]}{'...' if len(dropped_stocks) > 10 else ''}"
        )
    
    # Drop any remaining rows with all NaN
    cleaned_df = cleaned_df.dropna(how='all')
    
    logger.info(f"Missing values handled: {original_count} → {len(cleaned_df.columns)} stocks")
    
    return cleaned_df


# -------------------------------------------------------------------
# CURRENCY CONVERSION
# -------------------------------------------------------------------

def load_exchange_rates_from_ecb(base_currency: str, target_currency: str = "EUR") -> pd.Series:
    """
    Load exchange rates from ECB data portal.
    
    Tries to load from existing CSV files or downloads from ECB if needed.
    Currently supports: USD/EUR, GBP/EUR, SEK/EUR, CHF/EUR
    
    Parameters
    ----------
    base_currency : str
        Base currency (USD, GBP, SEK, CHF)
    target_currency : str
        Target currency (default: EUR)
    
    Returns
    -------
    pd.Series
        Monthly exchange rates, index=month-end dates, values=base_currency per target_currency
    """
    # ECB series codes for exchange rates
    ecb_series_codes = {
        "USD/EUR": "EXR.D.USD.EUR.SP00.A",
        "GBP/EUR": "EXR.D.GBP.EUR.SP00.A",
        "SEK/EUR": "EXR.D.SEK.EUR.SP00.A",
        "CHF/EUR": "EXR.D.CHF.EUR.SP00.A",
    }
    
    pair = f"{base_currency}/{target_currency}"
    if pair not in ecb_series_codes:
        raise ValueError(f"Exchange rate pair {pair} not supported. Available: {list(ecb_series_codes.keys())}")
    
    series_code = ecb_series_codes[pair]
    
    # Try to find existing CSV file
    exch_dir = DATA_RAW_EXCHANGE_DIR
    # Look for files with the currency pair in name (various formats)
    pattern1 = f"{base_currency.lower()}_{target_currency.lower()}"
    pattern2 = f"{base_currency}_{target_currency}"
    pattern3 = f"{base_currency}/{target_currency}".lower()
    
    exch_files = []
    if os.path.exists(exch_dir):
        for f in os.listdir(exch_dir):
            if f.endswith('.csv'):
                f_lower = f.lower()
                # Match various file naming patterns
                if (pattern1 in f_lower or pattern2 in f_lower or pattern3 in f_lower or 
                    (base_currency.lower() in f_lower and target_currency.lower() in f_lower) or
                    (base_currency.lower() in f_lower and target_currency.lower() in f_lower and 'ecb' in f_lower)):
                    exch_files.append(f)
    
    # Special case for USD/EUR - also check ECB file (has different naming)
    if pair == "USD/EUR" and not exch_files:
        ecb_files = [f for f in os.listdir(exch_dir) if f.endswith('.csv') and 'ecb' in f.lower()]
        if ecb_files:
            exch_files = ecb_files
    
    if exch_files:
        # Load from existing file (prefer most recent)
        exch_files.sort(reverse=True)  # Most recent first
        exch_file = os.path.join(exch_dir, exch_files[0])
        logger.info(f"Loading {pair} exchange rates from: {exch_file}")
        df = pd.read_csv(exch_file)
    else:
        raise FileNotFoundError(
            f"No {pair} exchange rate file found in {exch_dir}. "
            f"Please download using: python -m analysis.data.data_collection (download_exchange_rates function)"
        )
    
    # Parse dates (first column should be DATE or Date)
    date_col = df.columns[0]
    
    # Find rate column
    # Could be: "GBP/EUR", "GBP_EUR", "EXR.D.GBP.EUR.SP00.A", or similar
    rate_col = None
    for col in df.columns:
        col_upper = col.upper()
        if (series_code in col or 
            f"{base_currency}/{target_currency}".upper() in col_upper or
            f"{base_currency}_{target_currency}".upper() in col_upper or
            (base_currency.upper() in col_upper and target_currency.upper() in col_upper and 'EXR' not in col_upper)):
            rate_col = col
            break
    
    if rate_col is None:
        # Fallback: check column names
        # ECB format: DATE, TIME PERIOD, "US dollar/Euro (EXR.D.USD.EUR.SP00.A)"
        # yfinance format: Date, GBP/EUR
        for col in df.columns:
            if 'eur' in col.lower() or 'exchange' in col.lower() or 'rate' in col.lower():
                if 'time' not in col.lower() and 'period' not in col.lower():
                    rate_col = col
                    break
        
        # If still not found, use appropriate column by position
        if rate_col is None:
            if len(df.columns) >= 3:
                rate_col = df.columns[2]  # ECB format (third column)
            elif len(df.columns) >= 2:
                rate_col = df.columns[1]  # yfinance format (second column)
            else:
                raise ValueError(f"Could not find exchange rate column in {exch_file}. Columns: {list(df.columns)}")
    
    df['date'] = pd.to_datetime(df[date_col])
    df = df.sort_values('date')
    
    # Convert rate to float (remove quotes if present, handle commas)
    df[rate_col] = df[rate_col].astype(str).str.replace('"', '').str.replace(',', '').astype(float)
    
    # Get monthly month-end rates
    df['year_month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('year_month').last()
    
    # Convert Period index to month-end timestamps
    monthly.index = monthly.index.to_timestamp('M')
    
    # Create Series with exchange rates
    exchange_rates = monthly[rate_col].copy()
    exchange_rates.name = f'{base_currency.lower()}_{target_currency.lower()}_rate'
    exchange_rates.index.name = 'date'
    
    logger.info(f"Loaded {len(exchange_rates)} monthly {pair} exchange rates")
    logger.info(f"  Date range: {exchange_rates.index.min()} to {exchange_rates.index.max()}")
    logger.info(f"  Rate range: [{exchange_rates.min():.4f}, {exchange_rates.max():.4f}]")
    
    return exchange_rates


def load_usd_eur_exchange_rates() -> pd.Series:
    """
    Load USD/EUR exchange rates from ECB data and convert to monthly month-end rates.
    
    Returns
    -------
    pd.Series
        Monthly USD/EUR exchange rates, index=month-end dates, values=USD per EUR
    """
    return load_exchange_rates_from_ecb("USD", "EUR")


def convert_stock_prices_to_eur(
    stock_prices: pd.DataFrame,
    country: str
) -> pd.DataFrame:
    """
    Convert stock prices from local currency to EUR.
    
    For GBP stocks: Price_EUR = Price_GBP × GBP_EUR_Rate
    For SEK stocks: Price_EUR = Price_SEK × SEK_EUR_Rate
    For CHF stocks: Price_EUR = Price_CHF × CHF_EUR_Rate
    For EUR stocks: No conversion needed
    
    Parameters
    ----------
    stock_prices : pd.DataFrame
        Stock prices in local currency, index=dates, columns=tickers
    country : str
        Country name (determines currency)
    
    Returns
    -------
    pd.DataFrame
        Stock prices in EUR, same structure as input
    """
    country_config = COUNTRIES.get(country)
    if country_config is None:
        logger.warning(f"Unknown country {country}, skipping currency conversion")
        return stock_prices
    
    currency = country_config.currency
    
    # EUR stocks don't need conversion
    if currency == "EUR":
        logger.debug(f"{country} stocks already in EUR, no conversion needed")
        return stock_prices
    
    # Load appropriate exchange rate
    try:
        exchange_rates = load_exchange_rates_from_ecb(currency, "EUR")
        logger.info(f"Converting {country} stock prices from {currency} to EUR...")
    except FileNotFoundError as e:
        logger.error(f"Could not load {currency}/EUR exchange rates: {e}")
        logger.error(f"Skipping currency conversion for {country} - results will have currency mismatch")
        return stock_prices
    
    # Align exchange rates with price dates
    stock_prices.index = pd.to_datetime(stock_prices.index)
    exchange_rates.index = pd.to_datetime(exchange_rates.index)
    
    # Convert to month-end for alignment
    prices_month_end = stock_prices.index.to_period('M').to_timestamp('M')
    
    # Reindex exchange rates to match price dates
    aligned_rates = exchange_rates.reindex(prices_month_end)
    aligned_rates = aligned_rates.ffill().bfill()  # Forward and backward fill
    
    # Handle any remaining NaN
    if aligned_rates.isna().any():
        aligned_rates = aligned_rates.fillna(exchange_rates.iloc[0])
    
    # Convert prices: EUR = Local_Currency × (Local_Currency/EUR_Rate)
    # Exchange rate is "local currency per EUR", so multiply
    # Vectorized conversion (much faster than loop)
    # Broadcast aligned_rates.values across all columns
    eur_prices = stock_prices.mul(aligned_rates.values, axis=0)
    
    logger.info(f"Converted {len(stock_prices.columns)} {country} stocks from {currency} to EUR")
    if len(stock_prices) > 0:
        sample_ticker = stock_prices.columns[0]
        logger.info(f"  Sample: {stock_prices[sample_ticker].iloc[0]:.2f} {currency} → {eur_prices[sample_ticker].iloc[0]:.2f} EUR")
    
    return eur_prices


def convert_riskfree_rate_to_eur(
    riskfree_rate: pd.Series,
    country: str
) -> pd.Series:
    """
    Get EUR risk-free rate for all countries.
    
    Since all stock returns and market returns are in EUR,
    all countries should use the EUR risk-free rate (German Bund).
    Interest rates are percentages and should NOT be multiplied by exchange rates.
    
    CORRECT APPROACH: Use German Bund rate for all countries (no conversion needed)
    WRONG APPROACH: Multiply interest rate by exchange rate (treats percentage as currency)
    
    Parameters
    ----------
    riskfree_rate : pd.Series
        Risk-free rate (ignored, kept for compatibility)
        Index should be dates
    country : str
        Country name
    
    Returns
    -------
    pd.Series
        German Bund rate in EUR (monthly, percentage form), same index as input
    """
    # For all countries, use German Bund rate (EUR)
    # Interest rates are percentages and should NOT be converted by exchange rates
    from analysis.data.riskfree_helper import get_riskfree_rate
    from analysis.utils.config import ANALYSIS_SETTINGS
    
    logger.info(f"Using German Bund rate (EUR) for {country} - all countries use same EUR risk-free rate")
    
    german_rate = get_riskfree_rate(
        "Germany",
        ANALYSIS_SETTINGS.start_date,
        ANALYSIS_SETTINGS.end_date
    )
    
    # Align dates with input riskfree_rate if needed
    if len(riskfree_rate) > 0:
        german_rate.index = pd.to_datetime(german_rate.index)
        riskfree_rate.index = pd.to_datetime(riskfree_rate.index)
        
        # Convert to month-end for alignment
        rf_dates_month_end = riskfree_rate.index.to_period('M').to_timestamp('M')
        
        # Reindex German rate to match input dates
        german_rate_aligned = german_rate.reindex(rf_dates_month_end)
        german_rate_aligned = german_rate_aligned.ffill().bfill()
        
        # Handle any remaining NaN
        if german_rate_aligned.isna().any():
            german_rate_aligned = german_rate_aligned.fillna(german_rate.iloc[0])
        
        # Update index to match original input
        german_rate_aligned.index = riskfree_rate.index
        
        logger.info(f"Aligned German Bund rate for {country}: {len(german_rate_aligned)} months")
        if len(german_rate_aligned) > 0:
            logger.info(f"  Sample: {german_rate_aligned.iloc[0]:.4f}% EUR (German Bund)")
        
        return german_rate_aligned
    
    return german_rate


def convert_usd_prices_to_eur(
    usd_prices: pd.Series,
    exchange_rates: pd.Series
) -> pd.Series:
    """
    Convert USD-denominated prices to EUR using USD/EUR exchange rates.
    
    Formula: Price_EUR = Price_USD / USD_EUR_Rate
    (If USD_EUR_Rate = 1.20, then 1 USD = 1/1.20 = 0.833 EUR)
    
    Parameters
    ----------
    usd_prices : pd.Series
        Prices in USD, index=dates
    exchange_rates : pd.Series
        USD/EUR exchange rates, index=dates (month-end)
    
    Returns
    -------
    pd.Series
        Prices in EUR, same index as input
    """
    # Ensure datetime indices
    usd_prices.index = pd.to_datetime(usd_prices.index)
    # Exchange rates should already be DatetimeIndex, but ensure it
    if isinstance(exchange_rates.index, pd.PeriodIndex):
        exchange_rates.index = exchange_rates.index.to_timestamp('M')
    exchange_rates.index = pd.to_datetime(exchange_rates.index)
    
    # Convert both to month-end for alignment
    usd_prices_month_end = usd_prices.index.to_period('M').to_timestamp('M')
    
    # Reindex exchange rates to match price dates (forward fill for missing)
    aligned_rates = exchange_rates.reindex(usd_prices_month_end)
    aligned_rates = aligned_rates.ffill()  # Forward fill
    
    # Handle any remaining NaN (use backward fill if needed)
    if aligned_rates.isna().any():
        aligned_rates = aligned_rates.bfill()  # Backward fill
        if aligned_rates.isna().any():
            # If still NaN, use the first available rate
            aligned_rates = aligned_rates.fillna(exchange_rates.iloc[0])
    
    # Convert prices: EUR = USD / (USD/EUR rate)
    # The exchange rate is "USD per EUR", so to convert USD to EUR: divide by rate
    eur_prices = usd_prices / aligned_rates.values
    eur_prices.index = usd_prices.index  # Keep original index
    eur_prices.name = usd_prices.name
    
    logger.info(f"Converted {len(usd_prices)} USD prices to EUR")
    if len(usd_prices) > 0:
        logger.info(f"  Sample: {usd_prices.iloc[0]:.2f} USD @ {aligned_rates.iloc[0]:.4f} USD/EUR → {eur_prices.iloc[0]:.2f} EUR")
    
    return eur_prices


# -------------------------------------------------------------------
# ALIGNMENT WITH MSCI INDEX
# -------------------------------------------------------------------

def align_stocks_with_index(
    stock_returns: pd.DataFrame,
    msci_index_returns: pd.Series
) -> pd.DataFrame:
    """
    Align stock returns with MSCI index returns on dates.
    
    Parameters
    ----------
    stock_returns : pd.DataFrame
        Stock returns, index=dates, columns=tickers
    msci_index_returns : pd.Series
        MSCI index returns, index=dates
    
    Returns
    -------
    pd.DataFrame
        Aligned stock returns (inner join on dates)
    """
    # Ensure datetime indices
    stock_returns.index = pd.to_datetime(stock_returns.index)
    msci_index_returns.index = pd.to_datetime(msci_index_returns.index)
    
    # Inner join to ensure same dates
    aligned = stock_returns.join(msci_index_returns.to_frame('msci_index'), how='inner')
    
    # Drop the MSCI column (we just needed it for alignment)
    aligned = aligned.drop(columns=['msci_index'], errors='ignore')
    
    logger.info(
        f"Aligned stocks with MSCI index: {aligned.shape}, "
        f"date range: {aligned.index.min()} to {aligned.index.max()}"
    )
    
    return aligned


def load_price_data(country: str, data_type: str = "stocks") -> pd.DataFrame:
    """
    Load price data for a country.
    
    Parameters
    ----------
    country : str
        Country name (or "Europe" for MSCI Europe index)
    data_type : str
        'stocks' or 'indices' or 'msci' or 'msci_europe'
    
    Returns
    -------
    pd.DataFrame
        Price data with date index
    """
    if data_type == "msci_europe":
        filename = "prices_indices_msci_Europe.csv"
    elif data_type == "msci":
        filename = f"prices_indices_msci_{country}.csv"
    elif data_type == "indices":
        filename = f"prices_indices_{country}.csv"
    else:
        filename = f"prices_stocks_{country}.csv"
    
    filepath = os.path.join(DATA_RAW_DIR, "prices", filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Price file not found: {filepath}")
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    return df


def create_country_panel(country: str) -> pd.DataFrame:
    """
    Create aligned panel for one country with stock returns and MSCI Europe index returns.
    
    NOTE: All countries now use the same MSCI Europe index (IEUR) as the market proxy,
    reflecting the integrated nature of European financial markets and providing a
    consistent benchmark for cross-country analysis.
    
    Parameters
    ----------
    country : str
        Country name
    
    Returns
    -------
    pd.DataFrame
        Long format panel with columns: [date, ticker, stock_return, msci_index_return, country]
    """
    logger.info(f"Processing country: {country}")
    
    # Load stock prices and convert to EUR (if needed)
    try:
        stock_prices = load_price_data(country, "stocks")
        
        # CRITICAL: Convert stock prices to EUR to match market proxy currency
        # This eliminates currency mismatch for GBP/SEK/CHF stocks
        stock_prices_eur = convert_stock_prices_to_eur(stock_prices, country)
        
        # Calculate returns from EUR-denominated prices
        stock_returns = prices_to_returns(stock_prices_eur)
        stock_returns = handle_missing_values(stock_returns)
        stock_returns = filter_stocks_by_observations(stock_returns, min_observations=MIN_OBSERVATIONS)
    except FileNotFoundError as e:
        logger.error(f"Could not load stock prices for {country}: {e}")
        return pd.DataFrame()
    
    # Load MSCI Europe index prices and convert to returns
    # ALL countries now use the same MSCI Europe index
    # CRITICAL: Convert from USD to EUR to match stock currencies
    try:
        msci_prices_usd = load_price_data("Europe", "msci_europe")
        
        # Convert dates to month-end if they are month-start
        # MSCI Europe data may have month-start dates, but we need month-end for alignment
        msci_prices_usd.index = pd.to_datetime(msci_prices_usd.index)
        # Check if dates are month-start (day == 1) and convert to month-end
        if (msci_prices_usd.index.day == 1).all():
            logger.info("Converting MSCI Europe dates from month-start to month-end")
            msci_prices_usd.index = msci_prices_usd.index.to_period('M').to_timestamp('M')
        
        # CRITICAL FIX: Convert IEUR prices from USD to EUR
        # IEUR is USD-denominated, but we need EUR returns for German investors
        logger.info("Converting MSCI Europe (IEUR) prices from USD to EUR...")
        try:
            exchange_rates = load_usd_eur_exchange_rates()
            # Get the price series (should be single column)
            if len(msci_prices_usd.columns) > 0:
                price_series_usd = msci_prices_usd.iloc[:, 0]
                # Convert to EUR
                price_series_eur = convert_usd_prices_to_eur(price_series_usd, exchange_rates)
                # Create DataFrame with EUR prices
                msci_prices = pd.DataFrame({price_series_eur.name: price_series_eur})
                msci_prices.index = price_series_eur.index
                logger.info(" Successfully converted MSCI Europe prices from USD to EUR")
            else:
                logger.error("No price data in MSCI Europe file")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to convert MSCI Europe to EUR: {e}")
            logger.error("Falling back to USD prices (WARNING: currency mismatch will affect results)")
            msci_prices = msci_prices_usd
        
        # Calculate returns from EUR-denominated prices
        msci_returns = prices_to_returns(msci_prices)
        # MSCI index should be a single column
        if len(msci_returns.columns) > 0:
            msci_returns_series = msci_returns.iloc[:, 0]
        else:
            logger.error(f"No MSCI Europe index data available")
            return pd.DataFrame()
    except FileNotFoundError as e:
        logger.error(f"Could not load MSCI Europe index prices: {e}")
        logger.error(f"  Expected file: {os.path.join(DATA_RAW_DIR, 'prices', 'prices_indices_msci_Europe.csv')}")
        return pd.DataFrame()
    
    # Align stocks with MSCI Europe index
    aligned_stock_returns = align_stocks_with_index(stock_returns, msci_returns_series)
    
    # Re-align MSCI returns to match stock dates
    msci_aligned = msci_returns_series.reindex(aligned_stock_returns.index)
    
    # Convert to long format panel (vectorized using stack/melt - much faster than loop)
    # Reset index to get dates as column, then stack to create long format
    aligned_stock_returns_reset = aligned_stock_returns.reset_index()
    aligned_stock_returns_reset.columns = ['date'] + list(aligned_stock_returns.columns)
    
    # Melt to long format (vectorized operation)
    country_panel = aligned_stock_returns_reset.melt(
        id_vars=['date'],
        var_name='ticker',
        value_name='stock_return'
    )
    
    # Add MSCI index returns (align by date)
    msci_aligned_reset = msci_aligned.reset_index()
    msci_aligned_reset.columns = ['date', 'msci_index_return']
    country_panel = country_panel.merge(msci_aligned_reset, on='date', how='left')
    
    # Add country column
    country_panel['country'] = country
    
    # Drop rows with missing data
    country_panel = country_panel.dropna(subset=['stock_return', 'msci_index_return'])
    
    if country_panel.empty:
        logger.warning(f"No valid data for {country}")
        return pd.DataFrame()
    
    logger.info(f"Created panel for {country}: {len(country_panel)} rows, {len(aligned_stock_returns.columns)} stocks")
    logger.info(f"  Using MSCI Europe index (IEUR) as market proxy")
    
    return country_panel


# -------------------------------------------------------------------
# EXCESS RETURNS CALCULATION
# -------------------------------------------------------------------

def create_excess_returns(
    stock_returns: pd.Series,
    msci_index_returns: pd.Series,
    riskfree_rate: pd.Series
) -> pd.DataFrame:
    """
    Calculate excess returns for stock and market.
    
    Formulas (per methodology):
    - Excess Stock Return: E_i,t = R_{i,t} - R_{f,t} (%)
    - Market Excess Return: E_{m,t} = R_{m,t} - R_{f,t} (%)
    
    Parameters
    ----------
    stock_returns : pd.Series
        Stock returns in percentage form (%)
    msci_index_returns : pd.Series
        MSCI index returns in percentage form (%)
    riskfree_rate : pd.Series
        Risk-free rate in percentage form (%)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [stock_excess_return, market_excess_return]
    """
    # Align all series on dates
    aligned_data = pd.DataFrame({
        'stock_return': stock_returns,
        'msci_index_return': msci_index_returns,
        'riskfree_rate': riskfree_rate,
    })
    
    # Calculate excess returns (simple subtraction in percentage form)
    aligned_data['stock_excess_return'] = aligned_data['stock_return'] - aligned_data['riskfree_rate']
    aligned_data['market_excess_return'] = aligned_data['msci_index_return'] - aligned_data['riskfree_rate']
    
    # Drop rows where risk-free rate is missing (can't calculate excess returns)
    aligned_data = aligned_data.dropna(subset=['riskfree_rate', 'stock_excess_return', 'market_excess_return'])
    
    return aligned_data[['stock_excess_return', 'market_excess_return', 'riskfree_rate']]


# -------------------------------------------------------------------
# MAIN PROCESSING FUNCTION
# -------------------------------------------------------------------

def process_all_countries() -> pd.DataFrame:
    """
    Process all countries and create final returns panel.
    
    Returns
    -------
    pd.DataFrame
        Final panel with columns:
        [date, country, ticker, stock_return, msci_index_return, 
         riskfree_rate, stock_excess_return, market_excess_return]
    """
    logger.info("="*70)
    logger.info("Starting returns processing for all countries")
    logger.info("="*70)
    
    universe = load_stock_universe()
    countries = sorted(universe['country'].unique())
    
    logger.info(f"Processing {len(countries)} countries: {countries}")
    
    all_panels = []
    
    for country in countries:
        try:
            # Create country panel (stocks + MSCI index)
            country_panel = create_country_panel(country)
            
            if country_panel.empty:
                logger.warning(f"Skipping {country} - no valid data")
                continue
            
            # Get risk-free rate
            # NOTE: All countries use German 3-month Bund (EUR) as the risk-free rate.
            # The get_riskfree_rate() function forces target_country = "Germany" for all countries,
            # ensuring consistency across the entire analysis.
            try:
                riskfree_rate = get_riskfree_rate(
                    country,
                    ANALYSIS_SETTINGS.start_date,
                    ANALYSIS_SETTINGS.end_date
                )
                
                # Align risk-free rate with panel dates
                # Convert panel dates to month-end for matching
                panel_dates_array = country_panel['date'].unique()
                panel_dates = pd.DatetimeIndex(pd.to_datetime(panel_dates_array))
                # Normalize to month-end for matching
                panel_dates_month_end = panel_dates.to_period('M').to_timestamp('M')
                
                riskfree_aligned = align_riskfree_with_returns(riskfree_rate, panel_dates_month_end)
                
                # CRITICAL: Convert risk-free rate to EUR to match stock and market returns
                # All returns are now in EUR, so risk-free rates must also be in EUR
                riskfree_aligned_eur = convert_riskfree_rate_to_eur(riskfree_aligned, country)
                
                # Create mapping: convert panel dates to month-end for lookup
                # Cache date conversion to avoid repeated operations
                country_panel_dates = pd.to_datetime(country_panel['date'])
                country_panel_dates_dt = pd.DatetimeIndex(country_panel_dates)
                country_panel_dates_month_end = country_panel_dates_dt.to_period('M').to_timestamp('M')
                
                # Use reindex for faster lookup (more efficient than dict + map)
                # Set month-end dates as index temporarily for alignment
                country_panel_indexed = country_panel.set_index(country_panel_dates_month_end)
                riskfree_aligned_eur_indexed = riskfree_aligned_eur.reindex(country_panel_indexed.index, method='ffill')
                country_panel['riskfree_rate'] = riskfree_aligned_eur_indexed.values
                # Reset index if needed (country_panel should keep original index)
                if country_panel.index.name is not None or not country_panel.index.equals(pd.RangeIndex(len(country_panel))):
                    country_panel = country_panel.reset_index(drop=True)
                
            except Exception as e:
                logger.error(f"Error fetching risk-free rate for {country}: {e}")
                logger.error(f"Skipping {country} - real risk-free rate data required (no placeholders)")
                # Skip this country - don't use placeholder
                continue
            
            # Calculate excess returns directly in the panel
            country_panel['stock_excess_return'] = country_panel['stock_return'] - country_panel['riskfree_rate']
            country_panel['market_excess_return'] = country_panel['msci_index_return'] - country_panel['riskfree_rate']
            
            # Drop rows where we can't calculate excess returns (missing risk-free rate)
            country_panel = country_panel.dropna(subset=['stock_excess_return', 'market_excess_return'])
            
            if len(country_panel) > 0:
                all_panels.append(country_panel)
                logger.info(f" {country}: {len(country_panel)} rows processed")
            else:
                logger.warning(f"  {country}: No excess returns calculated (missing risk-free rate)")
                
        except Exception as e:
            logger.error(f"Error processing {country}: {e}")
            logger.exception("Full error details:")
            continue
    
    if not all_panels:
        logger.error("No panels created - check data availability")
        return pd.DataFrame()
    
    # Combine all countries
    final_panel = pd.concat(all_panels, ignore_index=True)
    
    # Ensure proper column order
    column_order = [
        'date', 'country', 'ticker',
        'stock_return', 'msci_index_return', 'riskfree_rate',
        'stock_excess_return', 'market_excess_return'
    ]
    final_panel = final_panel[column_order]
    
    # Sort by country, then date, then ticker
    final_panel = final_panel.sort_values(['country', 'date', 'ticker'])
    
    logger.info("="*70)
    logger.info(f"Final panel created: {len(final_panel)} rows")
    logger.info(f"  Countries: {final_panel['country'].nunique()}")
    logger.info(f"  Stocks: {final_panel['ticker'].nunique()}")
    logger.info(f"  Date range: {final_panel['date'].min()} to {final_panel['date'].max()}")
    logger.info("="*70)
    
    # Save to processed data directory
    output_path = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
    final_panel.to_csv(output_path, index=False)
    logger.info(f" Saved final panel to: {output_path}")
    
    return final_panel


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process all countries
    panel = process_all_countries()
    
    if not panel.empty:
        print("\n" + "="*70)
        print("RETURNS PROCESSING COMPLETE")
        print("="*70)
        print(f"Panel shape: {panel.shape}")
        print(f"\nSample data:")
        print(panel.head(10))
        print("\n" + "="*70)
    else:
        print("\n Returns processing failed - check logs for details")

