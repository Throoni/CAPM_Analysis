"""
returns_processing.py

Convert raw price data into monthly excess returns ready for CAPM regressions.

Per methodology:
- Returns: R_t = [(P_t - P_{t-1}) / P_{t-1}] × 100 (%) - percentage form
- Beta benchmark: MSCI country indices (NOT local indices)
- Risk-free: 3-month government bonds
- Estimation window: Exactly 60 months per stock
- All calculations in local currency
"""

import os
import logging
from typing import Optional

import pandas as pd
import numpy as np

from analysis.utils.config import (
    ANALYSIS_SETTINGS,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
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
        Country name
    data_type : str
        'stocks' or 'indices' or 'msci'
    
    Returns
    -------
    pd.DataFrame
        Price data with date index
    """
    if data_type == "msci":
        filename = f"prices_indices_msci_{country}.csv"
    elif data_type == "indices":
        filename = f"prices_indices_{country}.csv"
    else:
        filename = f"prices_stocks_{country}.csv"
    
    filepath = os.path.join(DATA_RAW_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Price file not found: {filepath}")
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    return df


def create_country_panel(country: str) -> pd.DataFrame:
    """
    Create aligned panel for one country with stock returns and MSCI index returns.
    
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
    
    # Load stock prices and convert to returns
    try:
        stock_prices = load_price_data(country, "stocks")
        stock_returns = prices_to_returns(stock_prices)
        stock_returns = handle_missing_values(stock_returns)
        stock_returns = filter_stocks_by_observations(stock_returns, min_observations=MIN_OBSERVATIONS)
    except FileNotFoundError as e:
        logger.error(f"Could not load stock prices for {country}: {e}")
        return pd.DataFrame()
    
    # Load MSCI index prices and convert to returns
    try:
        msci_prices = load_price_data(country, "msci")
        msci_returns = prices_to_returns(msci_prices)
        # MSCI index should be a single column
        if len(msci_returns.columns) > 0:
            msci_returns_series = msci_returns.iloc[:, 0]
        else:
            logger.error(f"No MSCI index data for {country}")
            return pd.DataFrame()
    except FileNotFoundError as e:
        logger.error(f"Could not load MSCI index prices for {country}: {e}")
        return pd.DataFrame()
    
    # Align stocks with MSCI index
    aligned_stock_returns = align_stocks_with_index(stock_returns, msci_returns_series)
    
    # Re-align MSCI returns to match stock dates
    msci_aligned = msci_returns_series.reindex(aligned_stock_returns.index)
    
    # Convert to long format panel
    panel_list = []
    
    for ticker in aligned_stock_returns.columns:
        stock_ret = aligned_stock_returns[ticker]
        
        # Create DataFrame for this stock
        stock_panel = pd.DataFrame({
            'date': stock_ret.index,
            'ticker': ticker,
            'stock_return': stock_ret.values,
            'msci_index_return': msci_aligned.values,
            'country': country,
        })
        
        panel_list.append(stock_panel)
    
    if not panel_list:
        logger.warning(f"No valid data for {country}")
        return pd.DataFrame()
    
    country_panel = pd.concat(panel_list, ignore_index=True)
    
    logger.info(f"Created panel for {country}: {len(country_panel)} rows, {len(aligned_stock_returns.columns)} stocks")
    
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
                
                # Create mapping: convert panel dates to month-end for lookup
                country_panel_dates = pd.to_datetime(country_panel['date'])
                # Convert to DatetimeIndex for to_period operation
                country_panel_dates_dt = pd.DatetimeIndex(country_panel_dates)
                country_panel_dates_month_end = country_panel_dates_dt.to_period('M').to_timestamp('M')
                riskfree_dict = dict(zip(riskfree_aligned.index, riskfree_aligned.values))
                country_panel['riskfree_rate'] = country_panel_dates_month_end.map(riskfree_dict)
                
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
                logger.info(f"✅ {country}: {len(country_panel)} rows processed")
            else:
                logger.warning(f"⚠️  {country}: No excess returns calculated (missing risk-free rate)")
                
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
    logger.info(f"✅ Saved final panel to: {output_path}")
    
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
        print("\n❌ Returns processing failed - check logs for details")

