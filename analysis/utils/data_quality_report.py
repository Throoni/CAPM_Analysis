"""
data_quality_report.py

Generate comprehensive data quality report.

Provides summary statistics, missing data patterns, extreme value flags,
and data completeness metrics for all data sources.
"""

import os
import logging
from typing import Dict
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from analysis.utils.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_PRICES_DIR,
    DATA_RAW_RISKFREE_DIR,
    RESULTS_REPORTS_DIR,
    COUNTRIES,
)

logger = logging.getLogger(__name__)


def generate_data_quality_report() -> Dict:
    """
    Generate comprehensive data quality report.
    
    Returns
    -------
    Dict
        Dictionary with data quality metrics and recommendations
    """
    logger.info("="*70)
    logger.info("GENERATING DATA QUALITY REPORT")
    logger.info("="*70)
    
    project_root = Path(__file__).parent.parent.parent
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'data_sources': {},
        'summary_statistics': {},
        'missing_data_patterns': {},
        'extreme_value_flags': {},
        'data_completeness': {},
        'recommendations': []
    }
    
    # 1. Price Data Quality
    logger.info("\nAnalyzing price data...")
    price_stats = {}
    for country in COUNTRIES.keys():
        stock_file = project_root / DATA_RAW_PRICES_DIR / f"prices_stocks_{country}.csv"
        if stock_file.exists():
            try:
                prices_df = pd.read_csv(stock_file, index_col=0, parse_dates=True)
                
                price_stats[country] = {
                    'n_stocks': len(prices_df.columns),
                    'n_months': len(prices_df),
                    'date_range': f"{prices_df.index.min()} to {prices_df.index.max()}",
                    'negative_prices': (prices_df < 0).sum().sum(),
                    'zero_prices': (prices_df == 0).sum().sum(),
                    'missing_pct': prices_df.isna().sum().sum() / (len(prices_df) * len(prices_df.columns)) * 100,
                    'extreme_jumps': (prices_df.pct_change().abs() > 0.5).sum().sum()
                }
            except Exception as e:
                logger.warning(f"Error analyzing {country} prices: {e}")
                price_stats[country] = {'error': str(e)}
    
    report_data['data_sources']['prices'] = price_stats
    
    # 2. Processed Panel Quality
    logger.info("\nAnalyzing processed panel...")
    panel_path = project_root / DATA_PROCESSED_DIR / "returns_panel.csv"
    if panel_path.exists():
        try:
            panel_df = pd.read_csv(panel_path, parse_dates=['date'])
            
            panel_stats = {
                'total_rows': len(panel_df),
                'n_stocks': panel_df['ticker'].nunique(),
                'n_countries': panel_df['country'].nunique(),
                'date_range': f"{panel_df['date'].min()} to {panel_df['date'].max()}",
                'missing_data': {
                    'stock_return': panel_df['stock_return'].isna().sum(),
                    'msci_index_return': panel_df['msci_index_return'].isna().sum(),
                    'riskfree_rate': panel_df['riskfree_rate'].isna().sum(),
                    'stock_excess_return': panel_df['stock_excess_return'].isna().sum(),
                    'market_excess_return': panel_df['market_excess_return'].isna().sum()
                },
                'extreme_returns': {
                    'stock_returns_>100pct': (panel_df['stock_return'].abs() > 100).sum(),
                    'market_returns_>50pct': (panel_df['msci_index_return'].abs() > 50).sum()
                },
                'return_statistics': {
                    'stock_return_mean': panel_df['stock_return'].mean(),
                    'stock_return_std': panel_df['stock_return'].std(),
                    'stock_return_min': panel_df['stock_return'].min(),
                    'stock_return_max': panel_df['stock_return'].max(),
                    'market_return_mean': panel_df['msci_index_return'].mean(),
                    'market_return_std': panel_df['msci_index_return'].std()
                }
            }
            
            # Missing data by country
            missing_by_country = {}
            for country in COUNTRIES.keys():
                country_data = panel_df[panel_df['country'] == country]
                if len(country_data) > 0:
                    missing_by_country[country] = {
                        'total_rows': len(country_data),
                        'missing_stock_return': country_data['stock_return'].isna().sum(),
                        'missing_riskfree': country_data['riskfree_rate'].isna().sum(),
                        'missing_pct': country_data.isna().sum().sum() / (len(country_data) * len(country_data.columns)) * 100
                    }
            
            panel_stats['missing_by_country'] = missing_by_country
            
            report_data['data_sources']['processed_panel'] = panel_stats
            
        except Exception as e:
            logger.warning(f"Error analyzing processed panel: {e}")
            report_data['data_sources']['processed_panel'] = {'error': str(e)}
    
    # 3. Risk-Free Rate Quality
    logger.info("\nAnalyzing risk-free rates...")
    rf_stats = {}
    for country in COUNTRIES.keys():
        rf_file = project_root / DATA_RAW_RISKFREE_DIR / f"riskfree_rate_{country}.csv"
        if rf_file.exists():
            try:
                rf_df = pd.read_csv(rf_file, index_col=0, parse_dates=True)
                rf_series = rf_df.iloc[:, 0]
                
                rf_stats[country] = {
                    'n_months': len(rf_series),
                    'date_range': f"{rf_series.index.min()} to {rf_series.index.max()}",
                    'min_rate': rf_series.min(),
                    'max_rate': rf_series.max(),
                    'mean_rate': rf_series.mean(),
                    'missing_count': rf_series.isna().sum(),
                    'missing_pct': rf_series.isna().sum() / len(rf_series) * 100
                }
            except Exception as e:
                logger.warning(f"Error analyzing {country} risk-free rates: {e}")
                rf_stats[country] = {'error': str(e)}
    
    report_data['data_sources']['riskfree_rates'] = rf_stats
    
    # 4. Generate Recommendations
    recommendations = []
    
    # Check for critical issues
    if 'processed_panel' in report_data['data_sources']:
        panel_stats = report_data['data_sources']['processed_panel']
        if isinstance(panel_stats, dict) and 'missing_data' in panel_stats:
            missing_rf = panel_stats['missing_data'].get('riskfree_rate', 0)
            if missing_rf > 100:
                recommendations.append({
                    'severity': 'high',
                    'issue': f'{missing_rf} missing risk-free rates',
                    'recommendation': 'Verify risk-free rate data sources and processing'
                })
    
    # Check for data completeness
    if 'processed_panel' in report_data['data_sources']:
        panel_stats = report_data['data_sources']['processed_panel']
        if isinstance(panel_stats, dict) and 'n_stocks' in panel_stats:
            n_stocks = panel_stats['n_stocks']
            if n_stocks < 200:
                recommendations.append({
                    'severity': 'medium',
                    'issue': f'Only {n_stocks} stocks in final sample',
                    'recommendation': 'Consider expanding stock universe or relaxing filters'
                })
    
    report_data['recommendations'] = recommendations
    
    # 5. Save Report
    report_path = project_root / RESULTS_REPORTS_DIR / "data_quality_report.json"
    import json
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    logger.info(f"\n✅ Data quality report saved to: {report_path}")
    
    # 6. Generate Summary
    logger.info("\n" + "="*70)
    logger.info("DATA QUALITY SUMMARY")
    logger.info("="*70)
    
    if 'processed_panel' in report_data['data_sources']:
        panel_stats = report_data['data_sources']['processed_panel']
        if isinstance(panel_stats, dict):
            logger.info(f"Total Stocks: {panel_stats.get('n_stocks', 'N/A')}")
            logger.info(f"Total Rows: {panel_stats.get('total_rows', 'N/A')}")
            if 'missing_data' in panel_stats:
                logger.info(f"Missing Risk-Free Rates: {panel_stats['missing_data'].get('riskfree_rate', 0)}")
    
    logger.info(f"Recommendations: {len(recommendations)}")
    for rec in recommendations:
        logger.info(f"  [{rec['severity'].upper()}] {rec['issue']}")
    
    return report_data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    report = generate_data_quality_report()
    print("\n✅ Data quality report generated successfully")

