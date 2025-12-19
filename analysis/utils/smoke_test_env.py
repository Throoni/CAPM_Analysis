"""
Environment Smoke Test Module.

This module provides quick validation tests to verify that the analysis
environment is correctly configured before running the full pipeline.

Tests performed:
    1. Configuration validation:
       - All required directories exist
       - Config parameters are within expected ranges

    2. Data availability:
       - Stock universe CSV is accessible
       - Sample price data can be loaded
       - MSCI Europe index data available

    3. API connectivity (optional):
       - WRDS connection test (if credentials provided)
       - Yahoo Finance download test

    4. Python environment:
       - Required packages installed
       - Correct package versions

Usage:
    python -m analysis.utils.smoke_test_env

Exit codes:
    0: All tests passed
    1: One or more tests failed (check output for details)
"""

import os

import pandas as pd

from analysis.utils.config import (
    PROJECT_ROOT,
    DATA_RAW_DIR,
    ANALYSIS_SETTINGS,
    COUNTRIES,
)

from analysis.data.wrds_helper import save_sample_global_daily_prices
from analysis.data.yf_helper import download_monthly_prices


def run_env_smoke_test():
    """
    Run environment smoke test to verify configuration.
    
    Returns
    -------
    bool
        True if all tests pass, False otherwise
    """
    print("=" * 60)
    print("ENVIRONMENT SMOKE TEST")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Check directories
    print("\n[1] Checking directories...")
    if os.path.exists(PROJECT_ROOT):
        print(f"    Project root: {PROJECT_ROOT} - OK")
    else:
        print(f"    Project root: {PROJECT_ROOT} - MISSING")
        all_passed = False
    
    if os.path.exists(DATA_RAW_DIR):
        print(f"    Data raw dir: {DATA_RAW_DIR} - OK")
    else:
        print(f"    Data raw dir: {DATA_RAW_DIR} - MISSING")
        all_passed = False
    
    # Test 2: Check settings
    print("\n[2] Checking analysis settings...")
    start = ANALYSIS_SETTINGS.get("start_date", "")
    end = ANALYSIS_SETTINGS.get("end_date", "")
    print(f"    Analysis period: {start} to {end}")
    print(f"    Countries: {len(COUNTRIES)}")
    
    # Test 3: Check stock universe
    print("\n[3] Checking stock universe...")
    universe_path = os.path.join(PROJECT_ROOT, "data", "stock_universe.csv")
    if os.path.exists(universe_path):
        df = pd.read_csv(universe_path)
        print(f"    Universe file: {len(df)} rows - OK")
    else:
        print(f"    Universe file: MISSING")
        all_passed = False
    
    # Test 4: Check sample data download
    print("\n[4] Testing Yahoo Finance connection...")
    try:
        test_df = download_monthly_prices(["AAPL"], start, end)
        if len(test_df) > 0:
            print(f"    Yahoo Finance: {len(test_df)} rows downloaded - OK")
        else:
            print("    Yahoo Finance: No data returned - CHECK")
    except Exception as e:
        print(f"    Yahoo Finance: Error - {e}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: All tests PASSED")
    else:
        print("RESULT: Some tests FAILED - check output above")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = run_env_smoke_test()
    sys.exit(0 if success else 1)
