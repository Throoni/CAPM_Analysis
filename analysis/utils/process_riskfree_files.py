"""
Risk-Free Rate File Processing Module.

This module processes raw risk-free rate data files into the standardized
format required for CAPM excess return calculations.

Processing steps:
    1. Load raw CSV files (daily or weekly frequency)
    2. Parse dates and validate format
    3. Resample to monthly frequency (month-end values)
    4. Convert annual percentage rates to monthly:
       monthly_rate = annual_rate / 12
    5. Save processed files to standardized location

Expected input format:
    - CSV with date column and rate column
    - Rates in annualized percentage form (e.g., 3.0 = 3% per year)
    - Date formats: YYYY-MM-DD or DD/MM/YYYY

Output format:
    - Monthly CSV with columns: date, rate
    - Rates in monthly percentage form (e.g., 0.25 = 0.25% per month)
    - Dates as month-end (last day of each month)

Note: German Bund is used as the risk-free rate for all countries since
all returns are EUR-denominated.
"""

import os
import logging
from typing import Optional

import pandas as pd
import numpy as np

from analysis.utils.config import DATA_RAW_DIR, COUNTRIES
from analysis.data.riskfree_helper import convert_annual_pct_to_monthly_pct

logger = logging.getLogger(__name__)


def process_riskfree_file(
    filepath: str,
    country: str,
    start_date: str = "2020-12-01",
    end_date: str = "2025-11-30"
) -> Optional[pd.Series]:
    """
    Process a single risk-free rate CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
    country : str
        Country name for logging
    start_date : str
        Required start date (YYYY-MM-DD)
    end_date : str
        Required end date (YYYY-MM-DD)
    
    Returns
    -------
    pd.Series
        Monthly risk-free rates (month-end dates as index, monthly % as values)
        Returns None if file cannot be processed
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return None
    
    try:
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Find date and price columns
        date_col = None
        price_col = None
        
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
            if 'price' in col.lower() and price_col is None:
                price_col = col
        
        if date_col is None or price_col is None:
            logger.error(f"Could not find date or price column in {filepath}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            return None
        
        # Parse dates
        df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y', errors='coerce')
        df = df.dropna(subset=[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Filter date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]
        
        if len(df) == 0:
            logger.warning(f"No data in date range for {country}")
            return None
        
        # Convert to Series with date as index
        series = pd.Series(df[price_col].values, index=df[date_col])
        
        # Convert daily to monthly (take month-end values)
        # Group by year-month and take last value of each month
        monthly_series = series.resample('ME').last()
        
        # If last month is incomplete, forward fill to ensure we have Nov 2025
        if monthly_series.index[-1] < pd.to_datetime(end_date):
            # Add Nov 30, 2025 with last available value
            last_value = monthly_series.iloc[-1]
            monthly_series[pd.to_datetime(end_date)] = last_value
            monthly_series = monthly_series.sort_index()
        
        # Convert annual percentage to monthly percentage using compounding formula
        # This uses the same formula as riskfree_helper.py: (1 + R_annual)^(1/12) - 1
        monthly_rates = monthly_series.apply(convert_annual_pct_to_monthly_pct)
        
        # Ensure month-end dates
        monthly_rates.index = monthly_rates.index.to_period('M').to_timestamp('M')
        
        logger.info(f" Processed {country}: {len(monthly_rates)} months")
        logger.info(f"   Date range: {monthly_rates.index.min()} to {monthly_rates.index.max()}")
        logger.info(f"   Rate range: {monthly_rates.min():.4f}% to {monthly_rates.max():.4f}% (monthly)")
        
        return monthly_rates
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return None


def process_all_riskfree_files(
    source_dir: str = "venv/Risk free rates",
    output_dir: Optional[str] = None
) -> dict:
    """
    Process all risk-free rate files and save monthly rates.
    
    Parameters
    ----------
    source_dir : str
        Directory containing source CSV files
    output_dir : str, optional
        Directory to save processed files. If None, uses DATA_RAW_DIR
    
    Returns
    -------
    dict
        Dictionary mapping country names to processed Series
    """
    if output_dir is None:
        output_dir = DATA_RAW_DIR
    
    logger.info("="*70)
    logger.info("PROCESSING RISK-FREE RATE FILES")
    logger.info("="*70)
    
    # File mapping
    file_mapping = {
        "Germany": "Germany 3 Month Bond Yield Historical Data.csv",
        "UnitedKingdom": "United Kingdom 3-Month Bond Yield Historical Data.csv",
        "Sweden": "Sweden 3-Month Bond Yield Historical Data.csv",
        "Switzerland": "Switzerland 3-Month Bond Yield Historical Data.csv",
    }
    
    processed_rates = {}
    
    # Process each file
    for country, filename in file_mapping.items():
        filepath = os.path.join(source_dir, filename)
        logger.info(f"\nProcessing {country}...")
        
        monthly_rates = process_riskfree_file(filepath, country)
        
        if monthly_rates is not None:
            processed_rates[country] = monthly_rates
            
            # Save to CSV
            output_path = os.path.join(output_dir, f"riskfree_rate_{country}.csv")
            monthly_rates.to_csv(output_path, header=['monthly_rate_pct'])
            logger.info(f"   Saved: {output_path}")
        else:
            logger.warning(f"   Failed to process {country}")
    
    # Copy Germany's rate for EUR countries (France, Italy, Spain)
    if "Germany" in processed_rates:
        logger.info("\nCopying Germany rate for EUR countries...")
        german_rates = processed_rates["Germany"]
        
        for eur_country in ["France", "Italy", "Spain"]:
            output_path = os.path.join(output_dir, f"riskfree_rate_{eur_country}.csv")
            german_rates.to_csv(output_path, header=['monthly_rate_pct'])
            processed_rates[eur_country] = german_rates.copy()
            logger.info(f"    {eur_country}: Using German Bund rate")
            logger.info(f"   Saved: {output_path}")
    
    logger.info("\n" + "="*70)
    logger.info(f" Processed {len(processed_rates)} countries")
    logger.info("="*70)
    
    return processed_rates


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    process_all_riskfree_files()

