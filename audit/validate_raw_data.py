"""
validate_raw_data.py

Phase 1.1: Raw Data Validation
Validates raw price data, index data, and risk-free rate files.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from analysis.config import DATA_RAW_DIR, COUNTRIES, ANALYSIS_SETTINGS

logger = logging.getLogger(__name__)


class RawDataAudit:
    """Audit class for raw data validation."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed = []
        
    def log_issue(self, severity: str, message: str, details: Dict = None):
        """Log an issue (critical or warning)."""
        entry = {
            'severity': severity,
            'message': message,
            'details': details or {}
        }
        if severity == 'critical':
            self.issues.append(entry)
        else:
            self.warnings.append(entry)
        logger.warning(f"[{severity.upper()}] {message}")
    
    def log_pass(self, message: str):
        """Log a passed check."""
        self.passed.append(message)
        logger.info(f"[PASS] {message}")
    
    def check_date_alignment(self) -> Dict:
        """Check that all files use month-end dates."""
        logger.info("="*70)
        logger.info("Checking date alignment (month-end dates)")
        logger.info("="*70)
        
        issues_found = []
        
        for country in COUNTRIES.keys():
            # Check stock prices
            stock_file = os.path.join(DATA_RAW_DIR, f"prices_stocks_{country}.csv")
            if os.path.exists(stock_file):
                df = pd.read_csv(stock_file, index_col=0, parse_dates=True)
                dates = pd.to_datetime(df.index)
                
                # Check if dates are month-end
                # Note: to_period uses 'M', not 'ME'
                month_ends = dates.to_period('M').to_timestamp('M')
                if not dates.equals(month_ends):
                    non_month_end = dates[dates != month_ends]
                    if len(non_month_end) > 0:
                        issues_found.append({
                            'file': f"prices_stocks_{country}.csv",
                            'issue': f"{len(non_month_end)} dates are not month-end",
                            'examples': non_month_end[:5].strftime('%Y-%m-%d').tolist()
                        })
                        self.log_issue('critical', 
                                     f"{country} stock prices: {len(non_month_end)} non-month-end dates",
                                     {'file': stock_file, 'dates': non_month_end[:5].strftime('%Y-%m-%d').tolist()})
                else:
                    self.log_pass(f"{country} stock prices: All dates are month-end")
            
            # Check MSCI index
            msci_file = os.path.join(DATA_RAW_DIR, f"prices_indices_msci_{country}.csv")
            if os.path.exists(msci_file):
                df = pd.read_csv(msci_file, index_col=0, parse_dates=True)
                dates = pd.to_datetime(df.index)
                month_ends = dates.to_period('M').to_timestamp('M')
                if not dates.equals(month_ends):
                    non_month_end = dates[dates != month_ends]
                    if len(non_month_end) > 0:
                        issues_found.append({
                            'file': f"prices_indices_msci_{country}.csv",
                            'issue': f"{len(non_month_end)} dates are not month-end"
                        })
                        self.log_issue('critical',
                                     f"{country} MSCI index: {len(non_month_end)} non-month-end dates",
                                     {'file': msci_file})
                else:
                    self.log_pass(f"{country} MSCI index: All dates are month-end")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Date alignment: {len(issues_found)} issues found"
        }
    
    def check_date_ranges(self) -> Dict:
        """Check that all files cover the expected date range."""
        logger.info("="*70)
        logger.info("Checking date ranges")
        logger.info("="*70)
        
        expected_start = pd.to_datetime(ANALYSIS_SETTINGS.start_date)
        expected_end = pd.to_datetime(ANALYSIS_SETTINGS.end_date)
        
        # Allow tolerance: ±1 month for start, ±3 months for end
        start_tolerance = pd.Timedelta(days=35)  # ~1 month
        end_tolerance = pd.Timedelta(days=95)    # ~3 months
        
        issues_found = []
        
        for country in COUNTRIES.keys():
            # Check stock prices
            stock_file = os.path.join(DATA_RAW_DIR, f"prices_stocks_{country}.csv")
            if os.path.exists(stock_file):
                df = pd.read_csv(stock_file, index_col=0, parse_dates=True)
                actual_start = pd.to_datetime(df.index.min())
                actual_end = pd.to_datetime(df.index.max())
                
                # Check start date with tolerance
                start_diff = actual_start - expected_start
                if start_diff > start_tolerance:
                    issues_found.append({
                        'file': f"prices_stocks_{country}.csv",
                        'issue': f"Start date {actual_start} is more than 1 month after expected {expected_start}"
                    })
                    self.log_issue('warning',
                                 f"{country} stock prices: Start date mismatch (gap: {start_diff.days} days)",
                                 {'expected': str(expected_start), 'actual': str(actual_start), 'gap_days': start_diff.days})
                elif start_diff <= start_tolerance:
                    self.log_pass(f"{country} stock prices: Start date OK (actual: {actual_start}, expected: {expected_start})")
                
                # Check end date with tolerance
                end_diff = expected_end - actual_end
                if end_diff > end_tolerance:
                    issues_found.append({
                        'file': f"prices_stocks_{country}.csv",
                        'issue': f"End date {actual_end} is more than 3 months before expected {expected_end}"
                    })
                    self.log_issue('warning',
                                 f"{country} stock prices: End date mismatch (gap: {end_diff.days} days)",
                                 {'expected': str(expected_end), 'actual': str(actual_end), 'gap_days': end_diff.days})
                elif end_diff <= end_tolerance:
                    self.log_pass(f"{country} stock prices: End date OK (actual: {actual_end}, expected: {expected_end})")
                
                # Check if core analysis period is covered (2021-01 to 2025-11 for returns)
                core_start = pd.to_datetime('2021-01-31')
                core_end = pd.to_datetime('2025-11-30')
                if actual_start <= core_start and actual_end >= core_end:
                    self.log_pass(f"{country} stock prices: Covers core analysis period (2021-01 to 2025-11)")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Date ranges: {len(issues_found)} issues found"
        }
    
    def check_price_sanity(self) -> Dict:
        """Check for invalid prices (negative, zero, extreme values)."""
        logger.info("="*70)
        logger.info("Checking price sanity")
        logger.info("="*70)
        
        issues_found = []
        
        for country in COUNTRIES.keys():
            stock_file = os.path.join(DATA_RAW_DIR, f"prices_stocks_{country}.csv")
            if os.path.exists(stock_file):
                df = pd.read_csv(stock_file, index_col=0, parse_dates=True)
                
                # Check for negative prices
                negative_prices = (df < 0).sum().sum()
                if negative_prices > 0:
                    issues_found.append({
                        'file': f"prices_stocks_{country}.csv",
                        'issue': f"{negative_prices} negative price values"
                    })
                    self.log_issue('critical',
                                 f"{country}: {negative_prices} negative prices found")
                
                # Check for zero prices (except possibly delistings)
                zero_prices = (df == 0).sum().sum()
                if zero_prices > 0:
                    issues_found.append({
                        'file': f"prices_stocks_{country}.csv",
                        'issue': f"{zero_prices} zero price values"
                    })
                    self.log_issue('warning',
                                 f"{country}: {zero_prices} zero prices found (may be delistings)")
                
                # Check for extreme month-over-month changes (>50%)
                returns = df.pct_change()
                extreme_changes = (returns.abs() > 0.5).sum().sum()
                if extreme_changes > 0:
                    issues_found.append({
                        'file': f"prices_stocks_{country}.csv",
                        'issue': f"{extreme_changes} extreme price changes (>50%)"
                    })
                    self.log_issue('warning',
                                 f"{country}: {extreme_changes} extreme price changes (may be splits/dividends)")
                
                if negative_prices == 0 and zero_prices == 0:
                    self.log_pass(f"{country}: Price sanity checks passed")
        
        return {
            'passed': len([i for i in issues_found if 'negative' in i['issue']]) == 0,
            'issues': issues_found,
            'summary': f"Price sanity: {len(issues_found)} issues found"
        }
    
    def check_missing_patterns(self) -> Dict:
        """Check for systematic missing data patterns."""
        logger.info("="*70)
        logger.info("Checking missing data patterns")
        logger.info("="*70)
        
        issues_found = []
        
        for country in COUNTRIES.keys():
            stock_file = os.path.join(DATA_RAW_DIR, f"prices_stocks_{country}.csv")
            if os.path.exists(stock_file):
                df = pd.read_csv(stock_file, index_col=0, parse_dates=True)
                
                # Check missing percentage per stock
                missing_pct = df.isna().sum() / len(df) * 100
                high_missing = missing_pct[missing_pct > 10]
                
                if len(high_missing) > 0:
                    issues_found.append({
                        'file': f"prices_stocks_{country}.csv",
                        'issue': f"{len(high_missing)} stocks with >10% missing data",
                        'stocks': high_missing.to_dict()
                    })
                    self.log_issue('warning',
                                 f"{country}: {len(high_missing)} stocks with >10% missing data")
                
                # Check for systematic gaps (same dates missing across many stocks)
                missing_by_date = df.isna().sum(axis=1)
                systematic_gaps = missing_by_date[missing_by_date > len(df.columns) * 0.5]
                
                if len(systematic_gaps) > 0:
                    issues_found.append({
                        'file': f"prices_stocks_{country}.csv",
                        'issue': f"{len(systematic_gaps)} dates with >50% missing stocks",
                        'dates': systematic_gaps.index.strftime('%Y-%m-%d').tolist()
                    })
                    self.log_issue('warning',
                                 f"{country}: {len(systematic_gaps)} dates with systematic gaps")
                
                if len(high_missing) == 0 and len(systematic_gaps) == 0:
                    self.log_pass(f"{country}: No systematic missing data patterns")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Missing patterns: {len(issues_found)} issues found"
        }
    
    def validate_stock_count(self) -> Dict:
        """Validate stock count matches universe definition."""
        logger.info("="*70)
        logger.info("Validating stock counts")
        logger.info("="*70)
        
        issues_found = []
        total_stocks = 0
        
        for country in COUNTRIES.keys():
            stock_file = os.path.join(DATA_RAW_DIR, f"prices_stocks_{country}.csv")
            if os.path.exists(stock_file):
                df = pd.read_csv(stock_file, index_col=0, parse_dates=True)
                n_stocks = len(df.columns)
                total_stocks += n_stocks
                self.log_pass(f"{country}: {n_stocks} stocks")
            else:
                issues_found.append({
                    'country': country,
                    'issue': 'Stock price file not found'
                })
                self.log_issue('critical', f"{country}: Stock price file missing")
        
        # Expected: 249 stocks (from summary)
        if total_stocks != 249:
            issues_found.append({
                'issue': f"Total stock count {total_stocks} does not match expected 249"
            })
            self.log_issue('warning',
                         f"Stock count mismatch: {total_stocks} vs expected 249")
        else:
            self.log_pass(f"Total stock count: {total_stocks} (matches expected)")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'total_stocks': total_stocks,
            'summary': f"Stock count validation: {total_stocks} stocks found"
        }
    
    def run_all_checks(self) -> Dict:
        """Run all raw data validation checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 1.1: RAW DATA VALIDATION")
        logger.info("="*70 + "\n")
        
        results = {
            'date_alignment': self.check_date_alignment(),
            'date_ranges': self.check_date_ranges(),
            'price_sanity': self.check_price_sanity(),
            'missing_patterns': self.check_missing_patterns(),
            'stock_count': self.validate_stock_count()
        }
        
        # Summary
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

