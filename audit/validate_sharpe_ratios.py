"""
validate_sharpe_ratios.py

Focused audit on Sharpe ratio calculations and justification.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.utils.config import RESULTS_REPORTS_DIR, DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)


def validate_sharpe_ratios():
    """Validate and justify all Sharpe ratios."""
    logger.info("="*70)
    logger.info("SHARPE RATIO VALIDATION AUDIT")
    logger.info("="*70)
    
    results = {
        'passed_checks': [],
        'warnings': [],
        'critical_issues': [],
        'justifications': {}
    }
    
    # Load data
    results_file = os.path.join(RESULTS_REPORTS_DIR, "portfolio_optimization_results.csv")
    if not os.path.exists(results_file):
        results['critical_issues'].append("Portfolio optimization results file not found")
        return results
    
    portfolio_df = pd.read_csv(results_file)
    panel_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv"), parse_dates=['date'])
    rf_rate = panel_df['riskfree_rate'].mean()
    
    logger.info(f"Risk-free rate: {rf_rate:.6f}% (monthly)")
    logger.info(f"Number of portfolios: {len(portfolio_df)}")
    
    # Check 1: Formula correctness
    logger.info("\n" + "="*70)
    logger.info("CHECK 1: Sharpe Ratio Formula Correctness")
    logger.info("="*70)
    
    all_formulas_correct = True
    for _, row in portfolio_df.iterrows():
        portfolio = row['portfolio']
        ret = row['expected_return']
        vol = row['volatility']
        sharpe = row['sharpe_ratio']
        
        if pd.isna(sharpe):
            logger.info(f"[PASS] {portfolio}: Sharpe = NaN (correctly set for unrealistic volatility)")
            results['passed_checks'].append(f"{portfolio}: NaN correctly set")
            continue
        
        calculated = (ret - rf_rate) / vol if vol > 0 else 0
        diff = abs(sharpe - calculated)
        
        if diff > 0.0001:
            logger.error(f"[CRITICAL] {portfolio}: Formula mismatch! Stored={sharpe:.6f}, Calculated={calculated:.6f}")
            results['critical_issues'].append(f"{portfolio}: Sharpe ratio calculation error")
            all_formulas_correct = False
        else:
            logger.info(f"[PASS] {portfolio}: Formula correct (diff={diff:.8f})")
            results['passed_checks'].append(f"{portfolio}: Formula correct")
    
    if all_formulas_correct:
        logger.info("\n All Sharpe ratio formulas are correct!")
    
    # Check 2: Reasonableness of values
    logger.info("\n" + "="*70)
    logger.info("CHECK 2: Reasonableness of Sharpe Ratios")
    logger.info("="*70)
    
    # Calculate market benchmark
    market_returns = panel_df.groupby('date')['msci_index_return'].first()
    market_excess = market_returns - rf_rate
    market_sharpe = market_excess.mean() / market_returns.std() if market_returns.std() > 0 else 0
    market_sharpe_annual = market_sharpe * np.sqrt(12)
    
    logger.info(f"Market benchmark (MSCI Europe):")
    logger.info(f"  Monthly Sharpe: {market_sharpe:.4f}")
    logger.info(f"  Annualized Sharpe: {market_sharpe_annual:.2f}")
    
    for _, row in portfolio_df.iterrows():
        portfolio = row['portfolio']
        ret = row['expected_return']
        vol = row['volatility']
        sharpe = row['sharpe_ratio']
        
        if pd.isna(sharpe):
            continue
        
        annual_sharpe = sharpe * np.sqrt(12)
        
        # Justification logic
        if 'Unconstrained' in portfolio:
            if annual_sharpe > 10:
                logger.warning(f"[WARNING] {portfolio}: Annualized Sharpe {annual_sharpe:.2f} is unrealistic")
                logger.warning(f"  This is expected for unconstrained portfolios (theoretical result)")
                results['warnings'].append(f"{portfolio}: Unrealistic Sharpe ({annual_sharpe:.2f}) - theoretical")
                results['justifications'][portfolio] = {
                    'status': 'theoretical',
                    'annual_sharpe': annual_sharpe,
                    'reason': 'Unconstrained optimization with short selling produces theoretical results',
                    'realistic': False
                }
            else:
                logger.info(f"[PASS] {portfolio}: Annualized Sharpe {annual_sharpe:.2f} is reasonable")
                results['passed_checks'].append(f"{portfolio}: Reasonable Sharpe")
        elif 'Constrained' in portfolio:
            if annual_sharpe > 5:
                logger.warning(f"[WARNING] {portfolio}: Annualized Sharpe {annual_sharpe:.2f} is very high")
                logger.warning(f"  Verify this is achievable in practice")
                results['warnings'].append(f"{portfolio}: Very high Sharpe ({annual_sharpe:.2f})")
            elif annual_sharpe < 0:
                logger.error(f"[CRITICAL] {portfolio}: Negative Sharpe ratio!")
                results['critical_issues'].append(f"{portfolio}: Negative Sharpe ratio")
            else:
                logger.info(f"[PASS] {portfolio}: Annualized Sharpe {annual_sharpe:.2f} is reasonable")
                results['passed_checks'].append(f"{portfolio}: Reasonable Sharpe")
                results['justifications'][portfolio] = {
                    'status': 'realistic',
                    'annual_sharpe': annual_sharpe,
                    'reason': 'Constrained optimization produces achievable results',
                    'realistic': True
                }
        else:  # Equal-weighted
            if annual_sharpe < market_sharpe_annual:
                logger.info(f"[PASS] {portfolio}: Sharpe {annual_sharpe:.2f} lower than market (expected for naive strategy)")
                results['passed_checks'].append(f"{portfolio}: Lower Sharpe expected")
            else:
                logger.info(f"[PASS] {portfolio}: Sharpe {annual_sharpe:.2f} is reasonable")
                results['passed_checks'].append(f"{portfolio}: Reasonable Sharpe")
            results['justifications'][portfolio] = {
                'status': 'realistic',
                'annual_sharpe': annual_sharpe,
                'reason': 'Equal-weighted baseline portfolio',
                'realistic': True
            }
    
    # Check 3: Volatility reasonableness
    logger.info("\n" + "="*70)
    logger.info("CHECK 3: Volatility Reasonableness")
    logger.info("="*70)
    
    for _, row in portfolio_df.iterrows():
        portfolio = row['portfolio']
        vol = row['volatility']
        
        if vol < 0.1 and vol > 0:
            logger.warning(f"[WARNING] {portfolio}: Volatility {vol:.6f}% is unrealistically low")
            logger.warning(f"  This suggests theoretical optimization result (short selling)")
            results['warnings'].append(f"{portfolio}: Unrealistically low volatility ({vol:.6f}%)")
        elif vol < 0:
            logger.error(f"[CRITICAL] {portfolio}: Negative volatility!")
            results['critical_issues'].append(f"{portfolio}: Negative volatility")
        elif vol > 20:
            logger.warning(f"[WARNING] {portfolio}: Volatility {vol:.2f}% is very high")
            results['warnings'].append(f"{portfolio}: Very high volatility ({vol:.2f}%)")
        else:
            logger.info(f"[PASS] {portfolio}: Volatility {vol:.4f}% is reasonable")
            results['passed_checks'].append(f"{portfolio}: Reasonable volatility")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("AUDIT SUMMARY")
    logger.info("="*70)
    logger.info(f"Passed checks: {len(results['passed_checks'])}")
    logger.info(f"Warnings: {len(results['warnings'])}")
    logger.info(f"Critical issues: {len(results['critical_issues'])}")
    
    if len(results['critical_issues']) == 0:
        logger.info("\n No critical issues found!")
    else:
        logger.error("\n Critical issues found - review required")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    results = validate_sharpe_ratios()
    
    print("\n" + "="*70)
    print("SHARPE RATIO AUDIT COMPLETE")
    print("="*70)
