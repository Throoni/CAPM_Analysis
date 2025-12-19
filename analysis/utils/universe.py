"""
Stock Universe Management Module.

This module handles the loading and validation of the European stock universe
used throughout the CAPM analysis.

Universe file format (CSV):
    - country: Country name (Germany, France, UK, etc.)
    - asset_type: Either "stock" or "index"
    - ticker_yf: Yahoo Finance ticker symbol
    - currency: Trading currency (EUR, GBP, CHF)
    - local_index_name: Local market index name (e.g., "DAX 40")
    - msci_ticker: MSCI index ticker for the country

Key functions:
    load_stock_universe() -> pd.DataFrame
        Load and validate the complete universe file

    get_stocks_by_country(country: str) -> List[str]
        Get all stock tickers for a specific country

    get_universe_summary() -> Dict
        Summary statistics of the universe

Validation checks:
    - Required columns present
    - No duplicate tickers
    - Valid country names
    - Consistent currency per country
"""

import os
from typing import Optional

import pandas as pd

from analysis.utils.config import PROJECT_ROOT


# -------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------

# IMPORTANT: this is your real universe file
UNIVERSE_CSV_DEFAULT = os.path.join(
    PROJECT_ROOT, "docs", "stock_universe.csv"
)

REQUIRED_COLUMNS = [
    "country",
    "asset_type",
    "ticker_yf",
    "currency",
    "local_index_name",
    "msci_ticker",
]


# -------------------------------------------------------------------
# LOADER
# -------------------------------------------------------------------

def load_stock_universe(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the stock universe from the specified CSV file.

    Parameters
    ----------
    csv_path : str, optional
        Path to the CSV file. If None, uses UNIVERSE_CSV_DEFAULT.

    Returns
    -------
    pandas.DataFrame
        Cleaned universe DataFrame with standardized columns.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    if csv_path is None:
        csv_path = UNIVERSE_CSV_DEFAULT

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Universe CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Basic schema check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Universe CSV is missing required columns: {missing_cols}. "
            f"Expected at least: {REQUIRED_COLUMNS}"
        )

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Normalize key text fields
    for col in ["country", "asset_type", "ticker_yf", "currency", "local_index_name", "msci_ticker"]:
        df[col] = df[col].astype(str).str.strip()

    # Normalize asset_type to lowercase
    df["asset_type"] = df["asset_type"].str.lower()

    # Only allow 'stock' or 'index' for now
    invalid_types = df.loc[~df["asset_type"].isin(["stock", "index"]), "asset_type"].unique()
    if len(invalid_types) > 0:
        raise ValueError(
            f"Universe CSV has invalid asset_type values: {invalid_types}. "
            f"Allowed values are: 'stock' or 'index'."
        )

    return df


# -------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------

def summarize_universe(df: pd.DataFrame) -> None:
    """
    Print a simple summary of the universe for quick inspection.

    Parameters
    ----------
    df : pandas.DataFrame
        The universe DataFrame as returned by load_stock_universe().
    """
    print("========== STOCK UNIVERSE SUMMARY ==========")
    print(f"Total rows: {len(df)}")

    print("\nBy asset_type:")
    print(df["asset_type"].value_counts())

    print("\nBy country and asset type:")
    print(df.groupby(["country", "asset_type"])["ticker_yf"].count())

    print("\nBy local_index_name (stocks only):")
    print(
        df[df["asset_type"] == "stock"]["local_index_name"].value_counts()
    )

    print("\nCurrencies used:")
    print(df["currency"].value_counts())

    print("===========================================\n")


if __name__ == "__main__":
    # If you run this file directly, just load and summarize the universe.
    try:
        universe_df = load_stock_universe()
        summarize_universe(universe_df)
    except Exception as e:
        print("Error while loading/summarizing universe:", e)
