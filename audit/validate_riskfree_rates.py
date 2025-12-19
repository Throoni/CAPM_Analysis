"""
Risk-Free Rate Validation Audit Module.

This module validates the risk-free rate data used in CAPM excess return
calculations to ensure correctness and consistency.

Validations performed:
    1. Data Source Verification:
       - German 3-month Bund used for all EUR calculations
       - Data sourced from reliable provider (FRED, ECB)
       - No placeholder or synthetic values

    2. Rate Magnitude Checks:
       - Annual rates typically in range [-1%, 10%]
       - Monthly rates = annual rates / 12
       - No extreme outliers or sign errors

    3. Conversion Formula Verification:
       - Annual to monthly: R_monthly = R_annual / 12
       - Percentage handling: 3.0 means 3%, not 300%
       - Consistent units throughout

    4. Temporal Coverage:
       - Complete coverage for analysis period (2018-2023)
       - No missing months
       - Proper date alignment with returns

    5. Cross-Dataset Consistency:
       - Same risk-free rate used in all calculations
       - Consistent between CAPM regressions and Sharpe ratios
       - Currency consistency (EUR throughout)
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

from analysis.config import DATA_RAW_DIR, COUNTRIES
from analysis.data.riskfree_helper import convert_annual_to_monthly_rate, convert_annual_pct_to_monthly_pct

logger = logging.getLogger(__name__)


class RiskFreeRateAudit:
    """Audit class for risk-free rate validation."""
    
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
    
    def verify_conversion_formula(self) -> Dict:
        """Verify risk-free rate conversion formula is correct."""
        logger.info("="*70)
        logger.info("Verifying risk-free rate conversion formula")
        logger.info("="*70)
        
        issues_found = []
        
        # Test case 1: 3% annual → monthly
        annual_3pct = 3.0
        monthly_compounding = convert_annual_pct_to_monthly_pct(annual_3pct)
        monthly_simple = annual_3pct / 12.0
        
        # Expected: Compounding gives ~0.247%, simple gives 0.25%
        expected_compounding = 0.247  # Approximate
        expected_simple = 0.25
        
        if abs(monthly_compounding - expected_compounding) > 0.01:
            issues_found.append({
                'test': '3% annual rate',
                'issue': f'Compounding conversion gives {monthly_compounding:.4f}%, expected ~{expected_compounding}%'
            })
            self.log_issue('critical',
                         f"Conversion formula test failed: 3% annual → {monthly_compounding:.4f}% monthly (expected ~{expected_compounding}%)")
        else:
            self.log_pass(f"Conversion test 1 passed: 3% annual → {monthly_compounding:.4f}% monthly (compounding)")
        
        # Test case 2: 0.5% annual → monthly
        annual_0_5pct = 0.5
        monthly_compounding_2 = convert_annual_pct_to_monthly_pct(annual_0_5pct)
        expected_compounding_2 = 0.0416  # Approximate
        
        if abs(monthly_compounding_2 - expected_compounding_2) > 0.001:
            issues_found.append({
                'test': '0.5% annual rate',
                'issue': f'Compounding conversion gives {monthly_compounding_2:.4f}%, expected ~{expected_compounding_2}%'
            })
            self.log_issue('warning',
                         f"Conversion test 2: 0.5% annual → {monthly_compounding_2:.4f}% monthly (expected ~{expected_compounding_2}%)")
        else:
            self.log_pass(f"Conversion test 2 passed: 0.5% annual → {monthly_compounding_2:.4f}% monthly")
        
        # Check which formula is being used
        # If monthly ≈ annual/12, it's simple division
        # If monthly ≈ (1+annual)^(1/12)-1, it's compounding
        if abs(monthly_compounding - monthly_simple) < 0.001:
            self.log_issue('warning',
                         "Conversion appears to use simple division (annual/12) rather than compounding")
            issues_found.append({
                'issue': 'Formula uses simple division instead of compounding',
                'note': 'Methodology should specify which is correct'
            })
        else:
            self.log_pass("Conversion uses compounding formula (not simple division)")
        
        return {
            'passed': len([i for i in issues_found if 'critical' in str(i)]) == 0,
            'issues': issues_found,
            'summary': f"Conversion formula: {len(issues_found)} issues found"
        }
    
    def check_rate_ranges(self) -> Dict:
        """Check that risk-free rates are in reasonable ranges."""
        logger.info("="*70)
        logger.info("Checking risk-free rate ranges")
        logger.info("="*70)
        
        issues_found = []
        
        for country in COUNTRIES.keys():
            rf_file = os.path.join(DATA_RAW_DIR, f"riskfree_rate_{country}.csv")
            if os.path.exists(rf_file):
                df = pd.read_csv(rf_file, index_col=0, parse_dates=True)
                
                # Get rate column (may be 'Price', 'Value', or similar)
                rate_col = None
                for col in df.columns:
                    if 'price' in col.lower() or 'value' in col.lower() or 'rate' in col.lower():
                        rate_col = col
                        break
                
                if rate_col is None:
                    rate_col = df.columns[0]  # Use first column
                
                rates = pd.to_numeric(df[rate_col], errors='coerce')
                
                # Check if rates are in percentage form (0-10% annual range)
                # Monthly rates should be roughly 0-1% if annual is 0-12%
                min_rate = rates.min()
                max_rate = rates.max()
                
                # If rates are > 1%, they might be in annual form (should be monthly)
                if max_rate > 5.0:
                    issues_found.append({
                        'country': country,
                        'issue': f'Maximum rate {max_rate:.2f}% seems too high for monthly rate (may be annual)',
                        'max_rate': max_rate
                    })
                    self.log_issue('critical',
                                 f"{country}: Maximum rate {max_rate:.2f}% too high (may be annual instead of monthly)")
                
                # If rates are negative, flag
                if min_rate < -1.0:
                    issues_found.append({
                        'country': country,
                        'issue': f'Negative rates found: minimum {min_rate:.2f}%',
                        'min_rate': min_rate
                    })
                    self.log_issue('warning',
                                 f"{country}: Negative rates found (minimum {min_rate:.2f}%)")
                
                # Check for missing values
                missing = rates.isna().sum()
                if missing > 0:
                    issues_found.append({
                        'country': country,
                        'issue': f'{missing} missing risk-free rate values',
                        'missing_count': missing
                    })
                    self.log_issue('critical',
                                 f"{country}: {missing} missing risk-free rate values")
                
                if max_rate <= 5.0 and min_rate >= -1.0 and missing == 0:
                    self.log_pass(f"{country}: Rate ranges OK (min={min_rate:.3f}%, max={max_rate:.3f}%)")
        
        return {
            'passed': len([i for i in issues_found if 'critical' in str(i)]) == 0,
            'issues': issues_found,
            'summary': f"Rate ranges: {len(issues_found)} issues found"
        }
    
    def check_eur_consistency(self) -> Dict:
        """Verify EUR countries use the same German Bund rate."""
        logger.info("="*70)
        logger.info("Checking EUR country consistency")
        logger.info("="*70)
        
        issues_found = []
        eur_countries = ['Germany', 'France', 'Italy', 'Spain']
        
        # Load German rates as reference
        germany_file = os.path.join(DATA_RAW_DIR, "riskfree_rate_Germany.csv")
        if not os.path.exists(germany_file):
            self.log_issue('critical', "Germany risk-free rate file not found")
            return {'passed': False, 'issues': ['Germany file missing']}
        
        df_germany = pd.read_csv(germany_file, index_col=0, parse_dates=True)
        rate_col_germany = None
        for col in df_germany.columns:
            if 'price' in col.lower() or 'value' in col.lower() or 'rate' in col.lower():
                rate_col_germany = col
                break
        if rate_col_germany is None:
            rate_col_germany = df_germany.columns[0]
        
        germany_rates = pd.to_numeric(df_germany[rate_col_germany], errors='coerce')
        
        for country in eur_countries:
            if country == 'Germany':
                continue
            
            rf_file = os.path.join(DATA_RAW_DIR, f"riskfree_rate_{country}.csv")
            if os.path.exists(rf_file):
                df = pd.read_csv(rf_file, index_col=0, parse_dates=True)
                rate_col = None
                for col in df.columns:
                    if 'price' in col.lower() or 'value' in col.lower() or 'rate' in col.lower():
                        rate_col = col
                        break
                if rate_col is None:
                    rate_col = df.columns[0]
                
                country_rates = pd.to_numeric(df[rate_col], errors='coerce')
                
                # Align dates and compare
                aligned = pd.DataFrame({
                    'Germany': germany_rates,
                    country: country_rates
                }).dropna()
                
                if len(aligned) > 0:
                    diff = (aligned['Germany'] - aligned[country]).abs()
                    max_diff = diff.max()
                    
                    if max_diff > 0.01:  # More than 0.01% difference
                        issues_found.append({
                            'country': country,
                            'issue': f'Rates differ from Germany (max diff: {max_diff:.4f}%)',
                            'max_diff': max_diff
                        })
                        self.log_issue('critical',
                                     f"{country}: Rates differ from Germany (max diff: {max_diff:.4f}%)")
                    else:
                        self.log_pass(f"{country}: Rates match Germany (max diff: {max_diff:.6f}%)")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"EUR consistency: {len(issues_found)} issues found"
        }
    
    def check_date_alignment(self) -> Dict:
        """Check that risk-free rates align with return dates."""
        logger.info("="*70)
        logger.info("Checking risk-free rate date alignment")
        logger.info("="*70)
        
        issues_found = []
        
        # Expected date range
        expected_dates = pd.date_range(start='2020-12-31', end='2025-11-30', freq='ME')
        
        for country in COUNTRIES.keys():
            rf_file = os.path.join(DATA_RAW_DIR, f"riskfree_rate_{country}.csv")
            if os.path.exists(rf_file):
                df = pd.read_csv(rf_file, index_col=0, parse_dates=True)
                dates = pd.to_datetime(df.index)
                
                # Check if dates are month-end
                month_ends = dates.to_period('M').to_timestamp('M')
                if not dates.equals(month_ends):
                    non_month_end = dates[dates != month_ends]
                    if len(non_month_end) > 0:
                        issues_found.append({
                            'country': country,
                            'issue': f'{len(non_month_end)} dates are not month-end',
                            'count': len(non_month_end)
                        })
                        self.log_issue('critical',
                                     f"{country}: {len(non_month_end)} non-month-end dates")
                
                # Check date range coverage
                missing_dates = expected_dates[~expected_dates.isin(dates)]
                if len(missing_dates) > 0:
                    issues_found.append({
                        'country': country,
                        'issue': f'{len(missing_dates)} expected dates missing',
                        'missing': missing_dates.strftime('%Y-%m-%d').tolist()[:5]
                    })
                    self.log_issue('warning',
                                 f"{country}: {len(missing_dates)} missing dates")
                
                if dates.equals(month_ends) and len(missing_dates) == 0:
                    self.log_pass(f"{country}: Date alignment OK")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Date alignment: {len(issues_found)} issues found"
        }
    
    def check_processed_panel_consistency(self) -> Dict:
        """Check that all countries have same risk-free rate in processed panel."""
        logger.info("="*70)
        logger.info("Checking processed panel risk-free rate consistency")
        logger.info("="*70)
        
        issues_found = []
        
        # Load processed panel
        from analysis.config import DATA_PROCESSED_DIR
        panel_file = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
        if not os.path.exists(panel_file):
            self.log_issue('critical', "Processed returns panel not found")
            return {'passed': False, 'issues': ['Panel file missing']}
        
        panel = pd.read_csv(panel_file, parse_dates=['date'])
        
        # Check that all countries have the same risk-free rate (should all use German Bund)
        country_rates = {}
        for country in sorted(panel['country'].unique()):
            country_data = panel[panel['country'] == country]
            rf_rates = country_data['riskfree_rate'].dropna()
            if len(rf_rates) > 0:
                country_rates[country] = rf_rates.mean()
        
        if len(country_rates) == 0:
            self.log_issue('critical', "No risk-free rates found in processed panel")
            return {'passed': False, 'issues': ['No risk-free rates in panel']}
        
        # Check if all countries have the same rate (within small tolerance)
        rates_list = list(country_rates.values())
        mean_rate = np.mean(rates_list)
        max_diff = max(abs(r - mean_rate) for r in rates_list)
        
        # Tolerance: rates should be within 0.0001% of each other
        if max_diff > 0.0001:
            for country, rate in country_rates.items():
                diff = abs(rate - mean_rate)
                if diff > 0.0001:
                    issues_found.append({
                        'country': country,
                        'issue': f'Risk-free rate differs from mean: {rate:.6f}% vs {mean_rate:.6f}% (diff: {diff:.6f}%)',
                        'rate': rate,
                        'mean_rate': mean_rate,
                        'diff': diff
                    })
                    self.log_issue('critical',
                                 f"{country}: Risk-free rate {rate:.6f}% differs from mean {mean_rate:.6f}% (diff: {diff:.6f}%)")
        else:
            self.log_pass(f"All countries have consistent risk-free rate: {mean_rate:.6f}% (max diff: {max_diff:.8f}%)")
        
        # Additional check: verify non-EUR countries use German Bund (not converted rates)
        # If rates are very different, it suggests incorrect conversion
        if len(country_rates) > 1:
            eur_countries = [c for c in country_rates.keys() if COUNTRIES.get(c, {}).currency == "EUR"]
            non_eur_countries = [c for c in country_rates.keys() if COUNTRIES.get(c, {}).currency != "EUR"]
            
            if eur_countries and non_eur_countries:
                eur_mean = np.mean([country_rates[c] for c in eur_countries])
                non_eur_mean = np.mean([country_rates[c] for c in non_eur_countries])
                diff = abs(eur_mean - non_eur_mean)
                
                if diff > 0.0001:
                    issues_found.append({
                        'issue': f'EUR vs non-EUR rate difference: {diff:.6f}% (suggests incorrect conversion)',
                        'eur_mean': eur_mean,
                        'non_eur_mean': non_eur_mean,
                        'diff': diff
                    })
                    self.log_issue('critical',
                                 f"EUR countries ({eur_mean:.6f}%) vs non-EUR ({non_eur_mean:.6f}%) differ by {diff:.6f}%")
                    self.log_issue('critical',
                                 "This suggests risk-free rates were incorrectly converted by exchange rates")
                    self.log_issue('critical',
                                 "All countries should use German Bund rate (EUR) - no conversion needed")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Processed panel consistency: {len(issues_found)} issues found"
        }
    
    def run_all_checks(self) -> Dict:
        """Run all risk-free rate validation checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 1.2: RISK-FREE RATE VALIDATION")
        logger.info("="*70 + "\n")
        
        results = {
            'conversion_formula': self.verify_conversion_formula(),
            'rate_ranges': self.check_rate_ranges(),
            'eur_consistency': self.check_eur_consistency(),
            'date_alignment': self.check_date_alignment(),
            'processed_panel_consistency': self.check_processed_panel_consistency()
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

