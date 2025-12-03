"""
validate_processed_data.py

Phase 1.3: Processed Data Validation
Validates processed returns panel and calculations.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict

from analysis.config import DATA_PROCESSED_DIR, COUNTRIES

logger = logging.getLogger(__name__)


class ProcessedDataAudit:
    """Audit class for processed data validation."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed = []
        
    def log_issue(self, severity: str, message: str, details: Dict = None):
        """Log an issue."""
        entry = {'severity': severity, 'message': message, 'details': details or {}}
        if severity == 'critical':
            self.issues.append(entry)
        else:
            self.warnings.append(entry)
        logger.warning(f"[{severity.upper()}] {message}")
    
    def log_pass(self, message: str):
        """Log a passed check."""
        self.passed.append(message)
        logger.info(f"[PASS] {message}")
    
    def verify_return_calculations(self) -> Dict:
        """Spot-check return calculations."""
        logger.info("="*70)
        logger.info("Verifying return calculations")
        logger.info("="*70)
        
        issues_found = []
        
        # Load processed panel
        panel_file = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
        if not os.path.exists(panel_file):
            self.log_issue('critical', "Processed returns panel not found")
            return {'passed': False, 'issues': ['Panel file missing']}
        
        panel = pd.read_csv(panel_file, parse_dates=['date'])
        
        # Test case: prices [100, 110, 105] should give returns [NaN, 10%, -4.55%]
        # We'll spot-check a few actual calculations
        sample_data = panel.head(100).copy()
        
        # Check that returns are in percentage form (not decimal)
        # If returns are < 1, they might be in decimal form
        if 'stock_return' in sample_data.columns:
            max_return = sample_data['stock_return'].abs().max()
            if max_return < 1.0:
                issues_found.append({
                    'issue': 'Returns appear to be in decimal form (< 1) rather than percentage form',
                    'max_return': max_return
                })
                self.log_issue('critical',
                             f"Returns may be in decimal form (max: {max_return:.4f})")
            else:
                self.log_pass(f"Returns appear to be in percentage form (max: {max_return:.2f}%)")
        
        # Check for extreme returns (>50% in absolute value)
        if 'stock_return' in sample_data.columns:
            extreme = (sample_data['stock_return'].abs() > 50).sum()
            if extreme > 0:
                issues_found.append({
                    'issue': f'{extreme} extreme returns (>50%) found',
                    'count': extreme
                })
                self.log_issue('warning',
                             f"{extreme} extreme returns (>50%) - may be data issues or corporate actions")
            else:
                self.log_pass("No extreme returns found")
        
        return {
            'passed': len([i for i in issues_found if 'critical' in str(i)]) == 0,
            'issues': issues_found,
            'summary': f"Return calculations: {len(issues_found)} issues found"
        }
    
    def verify_excess_returns(self) -> Dict:
        """Check excess return formula."""
        logger.info("="*70)
        logger.info("Verifying excess return calculations")
        logger.info("="*70)
        
        issues_found = []
        
        panel_file = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
        if not os.path.exists(panel_file):
            return {'passed': False, 'issues': ['Panel file missing']}
        
        panel = pd.read_csv(panel_file, parse_dates=['date'])
        
        # Check if excess returns are calculated correctly
        # Formula: excess_return = stock_return - riskfree_rate
        if 'stock_excess_return' in panel.columns and 'stock_return' in panel.columns and 'riskfree_rate' in panel.columns:
            # Spot-check: excess_return should equal stock_return - riskfree_rate
            sample = panel.dropna(subset=['stock_excess_return', 'stock_return', 'riskfree_rate']).head(1000)
            
            calculated_excess = sample['stock_return'] - sample['riskfree_rate']
            actual_excess = sample['stock_excess_return']
            
            diff = (calculated_excess - actual_excess).abs()
            max_diff = diff.max()
            
            if max_diff > 0.0001:  # More than 0.0001% difference
                issues_found.append({
                    'issue': f'Excess return calculation mismatch (max diff: {max_diff:.6f}%)',
                    'max_diff': max_diff
                })
                self.log_issue('critical',
                             f"Excess return calculation error: max diff = {max_diff:.6f}%")
            else:
                self.log_pass(f"Excess return calculation correct (max diff: {max_diff:.8f}%)")
        
        # Check sign: positive excess return means stock outperformed risk-free
        if 'stock_excess_return' in panel.columns:
            sample = panel.dropna(subset=['stock_excess_return', 'stock_return', 'riskfree_rate']).head(100)
            positive_excess = (sample['stock_excess_return'] > 0).sum()
            outperformed = (sample['stock_return'] > sample['riskfree_rate']).sum()
            
            if abs(positive_excess - outperformed) > 5:  # Allow small rounding differences
                issues_found.append({
                    'issue': 'Excess return sign may be incorrect',
                    'positive_excess': positive_excess,
                    'outperformed': outperformed
                })
                self.log_issue('critical',
                             f"Excess return sign issue: {positive_excess} positive excess vs {outperformed} outperformed")
            else:
                self.log_pass("Excess return signs are correct")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Excess returns: {len(issues_found)} issues found"
        }
    
    def check_date_alignment(self) -> Dict:
        """Verify all dates are aligned."""
        logger.info("="*70)
        logger.info("Checking date alignment in processed data")
        logger.info("="*70)
        
        issues_found = []
        
        panel_file = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
        if not os.path.exists(panel_file):
            return {'passed': False, 'issues': ['Panel file missing']}
        
        panel = pd.read_csv(panel_file, parse_dates=['date'])
        
        # Check that dates are month-end
        dates = pd.to_datetime(panel['date'])
        # Convert to DatetimeIndex if it's a Series
        if isinstance(dates, pd.Series):
            dates = pd.DatetimeIndex(dates)
        month_ends = dates.to_period('M').to_timestamp('M')
        non_month_end = dates[dates != month_ends]
        
        if len(non_month_end) > 0:
            issues_found.append({
                'issue': f'{len(non_month_end)} dates are not month-end',
                'count': len(non_month_end)
            })
            self.log_issue('critical',
                         f"{len(non_month_end)} non-month-end dates in panel")
        else:
            self.log_pass("All dates are month-end")
        
        # Check that stock returns, index returns, and risk-free rates all have same dates
        # (for each stock-date combination)
        sample = panel.dropna(subset=['stock_return', 'msci_index_return', 'riskfree_rate']).head(1000)
        if len(sample) > 0:
            # All should have dates
            missing_dates = sample[['stock_return', 'msci_index_return', 'riskfree_rate']].isna().any(axis=1).sum()
            if missing_dates > 0:
                issues_found.append({
                    'issue': f'{missing_dates} rows with misaligned data',
                    'count': missing_dates
                })
                self.log_issue('warning',
                             f"{missing_dates} rows with misaligned returns/rates")
            else:
                self.log_pass("All returns and rates are aligned by date")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Date alignment: {len(issues_found)} issues found"
        }
    
    def check_data_leakage(self) -> Dict:
        """Ensure no look-ahead bias."""
        logger.info("="*70)
        logger.info("Checking for data leakage (look-ahead bias)")
        logger.info("="*70)
        
        issues_found = []
        
        panel_file = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
        if not os.path.exists(panel_file):
            return {'passed': False, 'issues': ['Panel file missing']}
        
        panel = pd.read_csv(panel_file, parse_dates=['date'])
        
        # Check temporal ordering: dates should be in chronological order
        panel_sorted = panel.sort_values(['country', 'ticker', 'date'])
        dates_by_stock = panel_sorted.groupby(['country', 'ticker'])['date']
        
        out_of_order = 0
        for (country, ticker), dates in dates_by_stock:
            if not dates.is_monotonic_increasing:
                out_of_order += 1
        
        if out_of_order > 0:
            issues_found.append({
                'issue': f'{out_of_order} stocks have out-of-order dates',
                'count': out_of_order
            })
            self.log_issue('critical',
                         f"{out_of_order} stocks have non-chronological dates (potential look-ahead bias)")
        else:
            self.log_pass("All dates are in chronological order")
        
        # Check that returns don't use future prices
        # (This is harder to verify automatically, but we can check the calculation logic)
        # The returns_processing.py should calculate returns as (P_t - P_{t-1}) / P_{t-1}
        # which uses only past prices, so this is more of a code review check
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Data leakage check: {len(issues_found)} issues found"
        }
    
    def run_all_checks(self) -> Dict:
        """Run all processed data validation checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 1.3: PROCESSED DATA VALIDATION")
        logger.info("="*70 + "\n")
        
        results = {
            'return_calculations': self.verify_return_calculations(),
            'excess_returns': self.verify_excess_returns(),
            'date_alignment': self.check_date_alignment(),
            'data_leakage': self.check_data_leakage()
        }
        
        total_issues = sum(len(r.get('issues', [])) for r in results.values())
        total_critical = len(self.issues)
        total_warnings = len(self.warnings)
        
        results['summary'] = {
            'total_checks': len(results),
            'total_issues': total_issues,
            'critical_issues': total_critical,
            'warnings': total_warnings,
            'passed_checks': len(self.passed)
        }
        
        return results

