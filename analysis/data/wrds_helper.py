"""
wrds_helper.py

Utility functions to connect to WRDS and fetch sample data.
Stage 1: basic connection + a tiny smoke test for comp_global_daily.
"""

import os
from typing import Optional

import pandas as pd
import wrds

from analysis.utils.config import DATA_RAW_DIR
from analysis.utils.env_loader import load_env

# Load environment variables from .env file if available
load_env()


# -------------------------------------------------------------------
# WRDS CONNECTION
# -------------------------------------------------------------------

def get_wrds_connection(username: Optional[str] = None, password: Optional[str] = None) -> wrds.Connection:
    """
    Create and return a WRDS connection.

    WRDS will use credentials from:
    1. Provided username/password parameters
    2. Environment variables (WRDS_USERNAME, WRDS_PASSWORD)
    3. ~/.pgpass file
    4. Prompt interactively (last resort)
    
    Parameters
    ----------
    username : str, optional
        WRDS username. If None, checks environment variables, then .pgpass.
    password : str, optional
        WRDS password. If None, checks environment variables, then .pgpass.

    Returns
    -------
    wrds.Connection
        An active WRDS connection object.
    """
    # Try to get credentials from environment if not provided
    if not username:
        username = os.getenv('WRDS_USERNAME')
    if not password:
        password = os.getenv('WRDS_PASSWORD')
    
    # Use credentials if available, otherwise let WRDS use .pgpass or prompt
    if username and password:
        db = wrds.Connection(wrds_username=username, wrds_password=password)
    else:
        # WRDS will try .pgpass file automatically, or prompt if needed
        db = wrds.Connection()
    return db


# -------------------------------------------------------------------
# SAMPLE FETCH FROM COMP_GLOBAL_DAILY
# -------------------------------------------------------------------

def fetch_sample_global_daily_prices(
    db: Optional[wrds.Connection] = None,
    n_obs: int = 5,
) -> pd.DataFrame:
    """
    Fetch a small sample of global daily stock price data from comp_global_daily.g_secd.

    Parameters
    ----------
    db : wrds.Connection, optional
        Existing WRDS connection. If None, a new connection is created.
    n_obs : int
        Number of observations to fetch (for a small smoke test).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the first `n_obs` rows from comp_global_daily.g_secd.
    """
    close_conn = False
    if db is None:
        db = get_wrds_connection()
        close_conn = True

    print(f"Requesting first {n_obs} rows from comp_global_daily.g_secd ...")

    df = db.get_table(
        library="comp_global_daily",
        table="g_secd",
        obs=n_obs,
    )

    if close_conn:
        db.close()

    return df


def save_sample_global_daily_prices(
    n_obs: int = 5,
    filename: str = "wrds_comp_global_daily_sample.csv",
) -> str:
    """
    Fetch a small sample from comp_global_daily.g_secd and save it to CSV.

    Parameters
    ----------
    n_obs : int
        Number of observations to fetch.
    filename : str
        Name of the CSV file to save in data/raw/.

    Returns
    -------
    str
        Full path to the saved CSV file.
    """
    db = get_wrds_connection()
    df = fetch_sample_global_daily_prices(db=db, n_obs=n_obs)

    output_path = os.path.join(DATA_RAW_DIR, filename)
    df.to_csv(output_path, index=False)

    print(f"Saved WRDS sample to: {output_path}")
    db.close()
    return output_path


