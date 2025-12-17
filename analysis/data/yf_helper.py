"""
yf_helper.py

Helper functions for downloading price data from Yahoo Finance (yfinance).

Core function:
- download_monthly_prices(tickers, start, end)

This function:
- Accepts a list of tickers (or a single string)
- Downloads monthly prices from Yahoo Finance, ONE TICKER AT A TIME
- Returns a clean DataFrame:
    - index: dates
    - columns: tickers

It is intentionally conservative and robust:
- If a ticker has no usable data, it is skipped
- We log all failures and keep all successes
"""

import logging
from typing import List, Union

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _normalize_tickers(tickers: Union[str, List[str]]) -> List[str]:
    """
    Ensure we always work with a clean list of ticker strings.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    # Remove None / empty values and strip whitespace
    clean = [str(t).strip() for t in tickers if pd.notna(t) and str(t).strip() != ""]
    # Deduplicate & sort for consistency
    return sorted(set(clean))


def _download_single_ticker(
    ticker: str, start: str, end: str
) -> pd.Series:
    """
    Download monthly adjusted (or close) prices for a SINGLE ticker.

    Returns
    -------
    pandas.Series
        Index: dates (DatetimeIndex)
        Name: ticker
        Values: price data

    If no usable data is found, returns an empty Series.
    """
    try:
        data = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval="1mo",
            auto_adjust=True,   # adjusted for splits/dividends
            actions=False,
            progress=False,
            threads=False,      # single ticker => no need for threads
        )
    except Exception as e:
        logger.error(f"yfinance.download failed for ticker {ticker}: {e}")
        return pd.Series(dtype="float64", name=ticker)

    if data is None or data.empty:
        logger.warning(f"No data returned by yfinance for ticker: {ticker}")
        return pd.Series(dtype="float64", name=ticker)

    # Handle MultiIndex columns (when auto_adjust=True, yfinance returns MultiIndex)
    # Format: [('Close', 'TICKER'), ('High', 'TICKER'), ...]
    if isinstance(data.columns, pd.MultiIndex):
        # When auto_adjust=True, 'Close' contains adjusted prices
        # Check level 0 for column names
        level_0_cols = data.columns.levels[0] if hasattr(data.columns, 'levels') else data.columns.get_level_values(0).unique()
        
        col = None
        if "Adj Close" in level_0_cols:
            col = "Adj Close"
        elif "Close" in level_0_cols:
            col = "Close"
        
        if col is None:
            logger.warning(
                f"No 'Adj Close' or 'Close' column for ticker {ticker}. "
                f"Available columns (level 0): {list(level_0_cols)}"
            )
            return pd.Series(dtype="float64", name=ticker)
        
        # Access the column - this returns a DataFrame with one column
        col_data = data[col]
        
        # Extract the Series from the DataFrame
        if isinstance(col_data, pd.DataFrame):
            # Get the first (and only) column as a Series
            s = col_data.iloc[:, 0].copy()
        elif isinstance(col_data, pd.Series):
            s = col_data.copy()
        else:
            logger.warning(
                f"Unexpected data type for column {col} in ticker {ticker}: {type(col_data)}"
            )
            return pd.Series(dtype="float64", name=ticker)
    
    else:
        # Handle simple Index columns (non-MultiIndex case)
        col = None
        if "Adj Close" in data.columns:
            col = "Adj Close"
        elif "Close" in data.columns:
            col = "Close"
        
        if col is None:
            logger.warning(
                f"No 'Adj Close' or 'Close' column for ticker {ticker}. "
                f"Available columns: {list(data.columns)}"
            )
            return pd.Series(dtype="float64", name=ticker)
        
        s = data[col].copy()
        
        # If we got a DataFrame instead of Series, extract the Series
        if isinstance(s, pd.DataFrame):
            if len(s.columns) == 1:
                s = s.iloc[:, 0].copy()
            else:
                logger.warning(
                    f"Column {col} returned DataFrame with {len(s.columns)} columns for ticker {ticker}"
                )
                return pd.Series(dtype="float64", name=ticker)
    
    # Ensure we have a Series
    if not isinstance(s, pd.Series):
        logger.warning(
            f"Expected Series but got {type(s)} for ticker {ticker}"
        )
        return pd.Series(dtype="float64", name=ticker)
    
    s.name = ticker
    # Ensure datetime index and sorted
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    # If series is completely NaN, treat as unusable
    if len(s) == 0 or s.isna().all():
        logger.warning(
            f"All-NaN or empty price series for ticker {ticker}. Treating as no data."
        )
        return pd.Series(dtype="float64", name=ticker)

    # Validate date range - check if data ends early
    expected_end = pd.to_datetime(end)
    actual_end = s.index.max()
    
    # Calculate months difference
    months_diff = (expected_end.year - actual_end.year) * 12 + (expected_end.month - actual_end.month)
    
    if months_diff > 3:
        logger.warning(
            f"  DATA QUALITY WARNING for {ticker}: "
            f"Data ends at {actual_end.strftime('%Y-%m-%d')}, "
            f"expected until {expected_end.strftime('%Y-%m-%d')} "
            f"({months_diff} months early). This may indicate delisting or data issues."
        )
    elif months_diff > 0:
        logger.info(
            f"Note: {ticker} data ends {months_diff} month(s) before expected end date "
            f"({actual_end.strftime('%Y-%m-%d')} vs {expected_end.strftime('%Y-%m-%d')})"
        )

    return s


def download_monthly_prices(
    tickers: Union[str, List[str]],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download monthly prices for a list of tickers by looping
    over tickers one by one.

    Parameters
    ----------
    tickers : str or list of str
        Yahoo Finance tickers.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pandas.DataFrame
        DataFrame of prices:
            - index: dates (monthly)
            - columns: tickers
    """
    tickers_list = _normalize_tickers(tickers)

    if not tickers_list:
        logger.warning("download_monthly_prices called with empty ticker list.")
        return pd.DataFrame()

    logger.info(
        f"[yf_helper] Downloading monthly prices for {len(tickers_list)} tickers "
        f"from {start} to {end}, one ticker at a time."
    )

    series_list = []
    failed_tickers = []

    for t in tickers_list:
        s = _download_single_ticker(t, start=start, end=end)

        # Check if Series is empty (explicit check to avoid ambiguity)
        if not isinstance(s, pd.Series) or len(s) == 0 or s.isna().all():
            failed_tickers.append(t)
            continue

        series_list.append(s)

    if failed_tickers:
        logger.warning(
            f"[yf_helper] {len(failed_tickers)} tickers with no usable data: {failed_tickers}"
        )

    if not series_list:
        logger.warning(
            "[yf_helper] No valid price data collected for ANY ticker in list. "
            f"Tickers attempted: {tickers_list}"
        )
        return pd.DataFrame()

    # Combine all per-ticker Series into a single DataFrame
    prices = pd.concat(series_list, axis=1)

    # Sort index by date
    prices = prices.sort_index()

    logger.info(
        f"[yf_helper] Returning price DataFrame with shape {prices.shape} "
        f"for {len(prices.columns)} tickers."
    )

    return prices
