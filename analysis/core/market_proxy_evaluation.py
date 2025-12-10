"""
market_proxy_evaluation.py

Evaluate whether country-specific MSCI indices improve R² compared to MSCI Europe.
This helps determine if the market proxy choice is optimal.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple

from analysis.utils.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    RESULTS_REPORTS_DIR,
    COUNTRIES,
)
from analysis.core.returns_processing import (
    load_price_data,
    prices_to_returns,
    convert_stock_prices_to_eur,
    convert_usd_prices_to_eur,
    load_usd_eur_exchange_rates,
)
from analysis.core.capm_regression import run_capm_regression

logger = logging.getLogger(__name__)


def compare_market_proxies() -> pd.DataFrame:
    """
    Compare R² using MSCI Europe vs country-specific MSCI indices.
    
    For each country, calculates CAPM R² using:
    1. MSCI Europe (current approach)
    2. Country-specific MSCI index (alternative)
    
    Returns
    -------
    pd.DataFrame
        Comparison of R² values for each stock with both proxies
    """
    logger.info("="*70)
    logger.info("MARKET PROXY EVALUATION: MSCI Europe vs Country-Specific Indices")
    logger.info("="*70)
    
    # Load current CAPM results (using MSCI Europe)
    from analysis.utils.config import RESULTS_DATA_DIR
    capm_europe = pd.read_csv(os.path.join(RESULTS_DATA_DIR, "capm_results.csv"))
    
    # Load returns panel
    panel = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv"), parse_dates=['date'])
    
    comparison_results = []
    
    for country in COUNTRIES.keys():
        logger.info(f"\nEvaluating {country}...")
        
        # Get stocks for this country
        country_stocks = panel[panel['country'] == country]['ticker'].unique()
        
        if len(country_stocks) == 0:
            continue
        
        # Load country-specific MSCI index
        try:
            msci_country_prices = load_price_data(country, "msci")
            # Convert to EUR if needed (country indices are USD-denominated)
            msci_country_prices.index = pd.to_datetime(msci_country_prices.index)
            if (msci_country_prices.index.day == 1).all():
                msci_country_prices.index = msci_country_prices.index.to_period('M').to_timestamp('M')
            
            # Convert USD to EUR
            try:
                usd_eur_rates = load_usd_eur_exchange_rates()
                if len(msci_country_prices.columns) > 0:
                    price_series_usd = msci_country_prices.iloc[:, 0]
                    price_series_eur = convert_usd_prices_to_eur(price_series_usd, usd_eur_rates)
                    msci_country_prices = pd.DataFrame({price_series_eur.name: price_series_eur})
                    msci_country_prices.index = price_series_eur.index
            except Exception as e:
                logger.warning(f"Could not convert {country} MSCI index to EUR: {e}")
            
            msci_country_returns = prices_to_returns(msci_country_prices)
            if len(msci_country_returns.columns) > 0:
                msci_country_returns_series = msci_country_returns.iloc[:, 0]
            else:
                logger.warning(f"No {country} MSCI index data")
                continue
        except FileNotFoundError:
            logger.warning(f"No country-specific MSCI index for {country}")
            continue
        
        # For each stock, calculate R² with country-specific index
        for ticker in country_stocks[:10]:  # Limit to first 10 for speed
            stock_data = panel[(panel['country'] == country) & (panel['ticker'] == ticker)].set_index('date')
            
            if len(stock_data) < 30:
                continue
            
            # Align stock and country index returns
            aligned = pd.DataFrame({
                'stock_return': stock_data['stock_return'],
                'country_index_return': msci_country_returns_series
            }).dropna()
            
            if len(aligned) < 30:
                continue
            
            # Calculate R² with country index
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    aligned['country_index_return'], aligned['stock_return']
                )
                r2_country = r_value ** 2
            except:
                r2_country = np.nan
            
            # Get R² with MSCI Europe from current results
            stock_capm = capm_europe[(capm_europe['country'] == country) & (capm_europe['ticker'] == ticker)]
            if len(stock_capm) > 0:
                r2_europe = stock_capm.iloc[0]['r_squared']
            else:
                r2_europe = np.nan
            
            comparison_results.append({
                'country': country,
                'ticker': ticker,
                'r2_europe': r2_europe,
                'r2_country': r2_country,
                'r2_improvement': r2_country - r2_europe if not (np.isnan(r2_country) or np.isnan(r2_europe)) else np.nan
            })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    if len(comparison_df) > 0:
        logger.info(f"\nComparison Summary:")
        logger.info(f"  Stocks compared: {len(comparison_df)}")
        logger.info(f"  Average R² (MSCI Europe): {comparison_df['r2_europe'].mean():.3f}")
        logger.info(f"  Average R² (Country-specific): {comparison_df['r2_country'].mean():.3f}")
        logger.info(f"  Average improvement: {comparison_df['r2_improvement'].mean():.3f}")
        logger.info(f"  Stocks with better R² using country index: {(comparison_df['r2_improvement'] > 0).sum()}")
        
        # Save results
        output_file = os.path.join(RESULTS_REPORTS_DIR, "market_proxy_comparison.csv")
        comparison_df.to_csv(output_file, index=False)
        logger.info(f"✅ Saved: {output_file}")
    
    return comparison_df
