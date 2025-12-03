"""
validate_market_cap_analysis.py

Validates market-capitalization weighted beta analysis.
Checks data quality, calculation correctness, and result reasonableness.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict

from analysis.config import (
    RESULTS_DATA_DIR,
    RESULTS_TABLES_DIR,
    COUNTRIES
)

logger = logging.getLogger(__name__)


class MarketCapAnalysisAudit:
    """Audit class for market-cap weighted beta analysis."""
    
    def __init__(self):
        self.issues = []
        self.passed_checks = []
    
    def log_issue(self, severity: str, message: str, details: dict = None):
        """Log an issue."""
        issue = {
            'severity': severity,
            'message': message,
            'details': details or {}
        }
        self.issues.append(issue)
        if severity == 'critical':
            logger.error(f"[CRITICAL] {message}")
        elif severity == 'warning':
            logger.warning(f"[WARNING] {message}")
        else:
            logger.info(f"[INFO] {message}")
    
    def log_pass(self, message: str):
        """Log a passed check."""
        self.passed_checks.append(message)
        logger.info(f"[PASS] {message}")
    
    def check_market_cap_table_exists(self) -> bool:
        """Check if market-cap weighted betas table exists."""
        logger.info("="*70)
        logger.info("Checking market-cap weighted betas table")
        logger.info("="*70)
        
        table_file = os.path.join(RESULTS_TABLES_DIR, "table7_market_cap_weighted_betas.csv")
        
        if not os.path.exists(table_file):
            self.log_issue('critical', "Market-cap weighted betas table not found")
            return False
        
        try:
            df = pd.read_csv(table_file)
            if len(df) == 0:
                self.log_issue('critical', "Market-cap weighted betas table is empty")
                return False
            
            self.log_pass(f"Market-cap weighted betas table exists with {len(df)} rows")
            return True
        except Exception as e:
            self.log_issue('critical', f"Error reading market-cap table: {e}")
            return False
    
    def validate_market_cap_calculations(self) -> Dict:
        """Validate market-cap weighted beta calculations."""
        logger.info("="*70)
        logger.info("Validating market-cap weighted beta calculations")
        logger.info("="*70)
        
        table_file = os.path.join(RESULTS_TABLES_DIR, "table7_market_cap_weighted_betas.csv")
        
        if not os.path.exists(table_file):
            return {'passed': False, 'issues': self.issues}
        
        df = pd.read_csv(table_file)
        issues_found = []
        
        # Check required columns
        required_cols = ['country', 'equal_weighted_beta', 'market_cap_weighted_beta']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues_found.append({
                'issue': f'Missing required columns: {missing_cols}',
                'severity': 'critical'
            })
            self.log_issue('critical', f"Missing columns: {missing_cols}")
        else:
            self.log_pass("All required columns present")
        
        # Check for NaN values in key columns
        for col in ['equal_weighted_beta', 'market_cap_weighted_beta']:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    issues_found.append({
                        'issue': f'{nan_count} NaN values in {col}',
                        'severity': 'warning' if nan_count < len(df) else 'critical'
                    })
                    self.log_issue('warning', f"{nan_count} NaN values in {col}")
                else:
                    self.log_pass(f"No NaN values in {col}")
        
        # Check beta reasonableness
        if 'equal_weighted_beta' in df.columns:
            ew_betas = df['equal_weighted_beta'].dropna()
            if len(ew_betas) > 0:
                extreme_high = (ew_betas > 3.0).sum()
                extreme_low = (ew_betas < -1.0).sum()
                
                if extreme_high > 0:
                    issues_found.append({
                        'issue': f'{extreme_high} countries with beta > 3.0',
                        'severity': 'warning'
                    })
                    self.log_issue('warning', f"{extreme_high} countries with very high betas")
                
                if extreme_low > 0:
                    issues_found.append({
                        'issue': f'{extreme_low} countries with beta < -1.0',
                        'severity': 'warning'
                    })
                    self.log_issue('warning', f"{extreme_low} countries with negative betas")
                
                if extreme_high == 0 and extreme_low == 0:
                    self.log_pass("All equal-weighted betas in reasonable range")
        
        # Check market-cap weighted betas
        if 'market_cap_weighted_beta' in df.columns:
            mw_betas = df['market_cap_weighted_beta'].dropna()
            if len(mw_betas) > 0:
                extreme_high = (mw_betas > 3.0).sum()
                extreme_low = (mw_betas < -1.0).sum()
                
                if extreme_high > 0:
                    self.log_issue('warning', f"{extreme_high} countries with market-cap weighted beta > 3.0")
                
                if extreme_low > 0:
                    self.log_issue('warning', f"{extreme_low} countries with market-cap weighted beta < -1.0")
        
        # Check differences between EW and MW betas
        if 'equal_weighted_beta' in df.columns and 'market_cap_weighted_beta' in df.columns:
            valid_rows = df[df[['equal_weighted_beta', 'market_cap_weighted_beta']].notna().all(axis=1)]
            if len(valid_rows) > 0:
                differences = (valid_rows['equal_weighted_beta'] - valid_rows['market_cap_weighted_beta']).abs()
                large_diffs = (differences > 0.2).sum()
                
                if large_diffs > 0:
                    issues_found.append({
                        'issue': f'{large_diffs} countries with large EW-MW beta differences (>0.2)',
                        'severity': 'info'
                    })
                    self.log_issue('info', f"{large_diffs} countries with large differences (may indicate weighting matters)")
                else:
                    self.log_pass("EW-MW beta differences are reasonable")
        
        # Check data source
        if 'mcap_data_source' in df.columns:
            sources = df['mcap_data_source'].value_counts()
            estimated_count = (df['mcap_data_source'] == 'estimated').sum() if 'estimated' in df['mcap_data_source'].values else 0
            
            if estimated_count > len(df) * 0.5:
                issues_found.append({
                    'issue': f'More than 50% of market cap data is estimated ({estimated_count}/{len(df)})',
                    'severity': 'warning'
                })
                self.log_issue('warning', f"High proportion of estimated market cap data: {estimated_count}/{len(df)}")
            else:
                self.log_pass(f"Market cap data source: {sources.to_dict()}")
        
        # Check coverage
        if 'n_stocks_with_mcap' in df.columns and 'n_stocks' in df.columns:
            coverage = df['n_stocks_with_mcap'] / df['n_stocks']
            low_coverage = (coverage < 0.8).sum()
            
            if low_coverage > 0:
                issues_found.append({
                    'issue': f'{low_coverage} countries with <80% market cap coverage',
                    'severity': 'warning'
                })
                self.log_issue('warning', f"{low_coverage} countries with low market cap coverage")
            else:
                self.log_pass("Market cap coverage is good (>=80% for all countries)")
        
        self.issues.extend(issues_found)
        
        return {
            'passed': len([i for i in issues_found if i.get('severity') == 'critical']) == 0,
            'issues': issues_found,
            'summary': f"Market-cap analysis validation: {len([i for i in issues_found if i.get('severity') == 'critical'])} critical, {len([i for i in issues_found if i.get('severity') == 'warning'])} warnings"
        }
    
    def validate_consistency_with_capm_results(self) -> Dict:
        """Validate that market-cap weighted betas are consistent with CAPM results."""
        logger.info("="*70)
        logger.info("Validating consistency with CAPM results")
        logger.info("="*70)
        
        # Load CAPM results
        capm_file = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
        mcap_file = os.path.join(RESULTS_TABLES_DIR, "table7_market_cap_weighted_betas.csv")
        
        if not os.path.exists(capm_file):
            self.log_issue('warning', "CAPM results not found, skipping consistency check")
            return {'passed': True, 'issues': []}
        
        if not os.path.exists(mcap_file):
            self.log_issue('warning', "Market-cap table not found, skipping consistency check")
            return {'passed': True, 'issues': []}
        
        capm = pd.read_csv(capm_file)
        mcap = pd.read_csv(mcap_file)
        
        issues_found = []
        
        # Compare equal-weighted betas
        for _, row in mcap.iterrows():
            country = row['country']
            if country == 'Overall':
                continue
            
            # Get CAPM betas for this country
            country_capm = capm[(capm['country'] == country) & (capm['is_valid'] == True)]
            if len(country_capm) == 0:
                continue
            
            capm_ew_beta = country_capm['beta'].mean()
            mcap_ew_beta = row['equal_weighted_beta']
            
            if not pd.isna(capm_ew_beta) and not pd.isna(mcap_ew_beta):
                diff = abs(capm_ew_beta - mcap_ew_beta)
                if diff > 0.01:  # Allow small rounding differences
                    issues_found.append({
                        'issue': f'{country}: EW beta mismatch (CAPM: {capm_ew_beta:.4f}, Market-cap table: {mcap_ew_beta:.4f})',
                        'severity': 'warning'
                    })
                    self.log_issue('warning', f"{country}: EW beta mismatch: {diff:.4f}")
                else:
                    self.log_pass(f"{country}: EW beta consistent with CAPM results")
        
        self.issues.extend(issues_found)
        
        return {
            'passed': len([i for i in issues_found if i.get('severity') == 'critical']) == 0,
            'issues': issues_found,
            'summary': f"Consistency check: {len(issues_found)} issues found"
        }
    
    def run_all_checks(self) -> Dict:
        """Run all market-cap analysis validation checks."""
        logger.info("="*70)
        logger.info("MARKET-CAP ANALYSIS VALIDATION")
        logger.info("="*70)
        
        # Check 1: Table exists
        table_exists = self.check_market_cap_table_exists()
        
        if not table_exists:
            return {
                'passed': False,
                'issues': self.issues,
                'summary': 'Market-cap analysis validation failed: table not found'
            }
        
        # Check 2: Validate calculations
        calc_results = self.validate_market_cap_calculations()
        
        # Check 3: Consistency with CAPM
        consistency_results = self.validate_consistency_with_capm_results()
        
        # Summary
        total_critical = len([i for i in self.issues if i.get('severity') == 'critical'])
        total_warnings = len([i for i in self.issues if i.get('severity') == 'warning'])
        
        return {
            'passed': total_critical == 0,
            'issues': self.issues,
            'passed_checks': self.passed_checks,
            'summary': {
                'total_checks': len(self.passed_checks) + len(self.issues),
                'passed_checks': len(self.passed_checks),
                'critical_issues': total_critical,
                'warnings': total_warnings
            }
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    audit = MarketCapAnalysisAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("MARKET-CAP ANALYSIS VALIDATION COMPLETE")
    print("="*70)
    print(f"Passed: {results['passed']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")

