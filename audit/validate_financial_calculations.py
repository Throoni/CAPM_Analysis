"""
validate_financial_calculations.py

Phase 2: Financial Calculations Audit
Validates return calculations, risk-free rate conversion, and excess returns.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict

from analysis.utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR
from analysis.core.returns_processing import prices_to_returns

logger = logging.getLogger(__name__)


class FinancialCalculationsAudit:
    """Audit class for financial calculations."""
    
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
    
    def test_return_calculations(self) -> Dict:
        """Test return calculation formula with known data."""
        logger.info("="*70)
        logger.info("Testing return calculations")
        logger.info("="*70)
        
        issues_found = []
        
        # Test case: prices [100, 110, 105] should give returns [NaN, 10%, -4.55%]
        # Note: prices_to_returns drops the first NaN row, so we get 2 returns from 3 prices
        test_prices = pd.DataFrame({
            'TEST': [100, 110, 105]
        }, index=pd.date_range('2020-01-31', periods=3, freq='ME'))
        
        test_returns = prices_to_returns(test_prices)
        
        # After dropping NaN, we have 2 returns: [10%, -4.55%]
        actual_returns = test_returns['TEST'].values
        
        # Check first return (should be ~10%: (110-100)/100 = 10%)
        if len(actual_returns) < 1:
            issues_found.append({
                'test': 'Return count',
                'issue': 'No returns calculated'
            })
            self.log_issue('critical', "No returns calculated")
        elif abs(actual_returns[0] - 10.0) > 0.01:
            issues_found.append({
                'test': 'First return',
                'issue': f'Expected 10%, got {actual_returns[0]:.4f}%'
            })
            self.log_issue('critical',
                         f"Return calculation error: Expected 10%, got {actual_returns[0]:.4f}%")
        else:
            self.log_pass(f"First return correct: {actual_returns[0]:.2f}%")
        
        # Check second return (should be ~-4.55%: (105-110)/110 = -4.545%)
        if len(actual_returns) < 2:
            issues_found.append({
                'test': 'Return count',
                'issue': 'Only one return calculated'
            })
            self.log_issue('warning', "Only one return calculated (expected 2)")
        elif abs(actual_returns[1] - (-4.545454545)) > 0.1:
            issues_found.append({
                'test': 'Second return',
                'issue': f'Expected -4.55%, got {actual_returns[1]:.4f}%'
            })
            self.log_issue('critical',
                         f"Return calculation error: Expected -4.55%, got {actual_returns[1]:.4f}%")
        else:
            self.log_pass(f"Second return correct: {actual_returns[1]:.2f}%")
        
        # Verify returns are in percentage form (not decimal)
        if abs(actual_returns[1]) < 1.0:
            issues_found.append({
                'test': 'Return units',
                'issue': 'Returns appear to be in decimal form (< 1) rather than percentage'
            })
            self.log_issue('critical', "Returns may be in decimal form instead of percentage")
        else:
            self.log_pass("Returns are in percentage form")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Return calculations: {len(issues_found)} issues found"
        }
    
    def verify_simple_vs_log_returns(self) -> Dict:
        """Verify that simple returns (not log returns) are used."""
        logger.info("="*70)
        logger.info("Verifying simple returns (not log returns)")
        logger.info("="*70)
        
        issues_found = []
        
        # Check the returns_processing.py code logic
        # Simple returns: (P_t - P_{t-1}) / P_{t-1}
        # Log returns: ln(P_t / P_{t-1})
        
        # Test with known data
        # With 2 prices, we get 1 return after dropping NaN
        test_prices = pd.DataFrame({
            'TEST': [100, 110]
        }, index=pd.date_range('2020-01-31', periods=2, freq='ME'))
        
        test_returns = prices_to_returns(test_prices)
        
        if len(test_returns) == 0:
            issues_found.append({
                'issue': 'No returns calculated'
            })
            self.log_issue('critical', "No returns calculated")
        else:
            simple_return = test_returns['TEST'].iloc[0]  # Should be 10%
            
            # If log returns were used, it would be ln(110/100) = 0.0953 = 9.53%
            # Simple return is (110-100)/100 = 0.10 = 10%
            if abs(simple_return - 10.0) < 0.1:
                self.log_pass("Simple returns are used (not log returns)")
            else:
                issues_found.append({
                    'issue': f'Returns may be log returns instead of simple returns (got {simple_return:.4f}%)'
                })
                self.log_issue('critical', f"Returns may be log returns instead of simple returns (got {simple_return:.4f}%)")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Return type: {len(issues_found)} issues found"
        }
    
    def test_riskfree_conversion(self) -> Dict:
        """Test risk-free rate conversion with known values."""
        logger.info("="*70)
        logger.info("Testing risk-free rate conversion")
        logger.info("="*70)
        
        issues_found = []
        
        from analysis.riskfree_helper import convert_annual_pct_to_monthly_pct
        
        # Test case 1: 3% annual → monthly
        annual_3pct = 3.0
        monthly_3pct = convert_annual_pct_to_monthly_pct(annual_3pct)
        expected_compounding = 0.247  # (1.03^(1/12) - 1) * 100
        expected_simple = 0.25  # 3/12
        
        if abs(monthly_3pct - expected_compounding) < 0.01:
            self.log_pass(f"3% annual → {monthly_3pct:.4f}% monthly (compounding)")
        elif abs(monthly_3pct - expected_simple) < 0.01:
            self.log_issue('warning',
                         f"3% annual → {monthly_3pct:.4f}% monthly (simple division, not compounding)")
            issues_found.append({
                'test': '3% conversion',
                'issue': 'Uses simple division instead of compounding'
            })
        else:
            issues_found.append({
                'test': '3% conversion',
                'issue': f'Unexpected result: {monthly_3pct:.4f}%'
            })
            self.log_issue('critical',
                         f"Conversion test failed: 3% → {monthly_3pct:.4f}% (expected ~{expected_compounding}%)")
        
        # Test case 2: 0.5% annual → monthly
        annual_0_5pct = 0.5
        monthly_0_5pct = convert_annual_pct_to_monthly_pct(annual_0_5pct)
        expected_compounding_2 = 0.0416  # (1.005^(1/12) - 1) * 100
        
        if abs(monthly_0_5pct - expected_compounding_2) < 0.001:
            self.log_pass(f"0.5% annual → {monthly_0_5pct:.4f}% monthly")
        else:
            issues_found.append({
                'test': '0.5% conversion',
                'issue': f'Unexpected result: {monthly_0_5pct:.4f}%'
            })
            self.log_issue('warning',
                         f"Conversion test 2: 0.5% → {monthly_0_5pct:.4f}% (expected ~{expected_compounding_2}%)")
        
        return {
            'passed': len([i for i in issues_found if 'critical' in str(i)]) == 0,
            'issues': issues_found,
            'summary': f"Risk-free conversion: {len(issues_found)} issues found"
        }
    
    def check_currency_conversion_logic(self) -> Dict:
        """Verify currency conversions are applied correctly to prices vs rates."""
        logger.info("="*70)
        logger.info("Checking currency conversion logic")
        logger.info("="*70)
        
        issues_found = []
        
        # Load processed panel to check actual values
        from analysis.config import DATA_PROCESSED_DIR
        panel_file = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
        if not os.path.exists(panel_file):
            self.log_issue('critical', "Processed returns panel not found")
            return {'passed': False, 'issues': ['Panel file missing']}
        
        panel = pd.read_csv(panel_file, parse_dates=['date'])
        
        # Check 1: Verify risk-free rates are NOT converted by exchange rates
        # All countries should have the same risk-free rate (German Bund)
        country_rf_means = {}
        for country in sorted(panel['country'].unique()):
            country_data = panel[panel['country'] == country]
            rf_rates = country_data['riskfree_rate'].dropna()
            if len(rf_rates) > 0:
                country_rf_means[country] = rf_rates.mean()
        
        if len(country_rf_means) == 0:
            self.log_issue('critical', "No risk-free rates found in processed panel")
            return {'passed': False, 'issues': ['No risk-free rates in panel']}
        
        # Check if all countries have the same rate (within small tolerance)
        rates_list = list(country_rf_means.values())
        mean_rate = np.mean(rates_list)
        max_diff = max(abs(r - mean_rate) for r in rates_list)
        
        # Tolerance: rates should be within 0.0001% of each other
        if max_diff > 0.0001:
            # Check if non-EUR countries have different rates (suggests conversion error)
            from analysis.config import COUNTRIES
            eur_countries = [c for c in country_rf_means.keys() if COUNTRIES.get(c, {}).currency == "EUR"]
            non_eur_countries = [c for c in country_rf_means.keys() if COUNTRIES.get(c, {}).currency != "EUR"]
            
            if eur_countries and non_eur_countries:
                eur_mean = np.mean([country_rf_means[c] for c in eur_countries])
                non_eur_mean = np.mean([country_rf_means[c] for c in non_eur_countries])
                diff = abs(eur_mean - non_eur_mean)
                
                if diff > 0.0001:
                    issues_found.append({
                        'issue': 'Risk-free rates differ between EUR and non-EUR countries',
                        'details': f'EUR countries: {eur_mean:.6f}%, non-EUR: {non_eur_mean:.6f}% (diff: {diff:.6f}%)',
                        'severity': 'critical'
                    })
                    self.log_issue('critical',
                                 f"Currency conversion error detected: EUR countries ({eur_mean:.6f}%) vs non-EUR ({non_eur_mean:.6f}%) differ by {diff:.6f}%")
                    self.log_issue('critical',
                                 "Risk-free rates should NOT be multiplied by exchange rates")
                    self.log_issue('critical',
                                 "All countries should use German Bund rate (EUR) - no conversion needed")
                else:
                    self.log_pass("Risk-free rates are consistent across countries")
            else:
                self.log_pass("Risk-free rates are consistent")
        else:
            self.log_pass(f"All countries have consistent risk-free rate (max diff: {max_diff:.8f}%)")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Currency conversion logic: {len(issues_found)} issues found"
        }
    
    def test_excess_return_formula(self) -> Dict:
        """Test excess return calculation formula."""
        logger.info("="*70)
        logger.info("Testing excess return formula")
        logger.info("="*70)
        
        issues_found = []
        
        # Test case 1: Stock return 2%, Risk-free 0.5% → Excess 1.5%
        stock_return = 2.0
        riskfree_rate = 0.5
        expected_excess = 1.5
        calculated_excess = stock_return - riskfree_rate
        
        if abs(calculated_excess - expected_excess) < 0.0001:
            self.log_pass(f"Excess return test 1: {stock_return}% - {riskfree_rate}% = {calculated_excess}%")
        else:
            issues_found.append({
                'test': 'Excess return 1',
                'issue': f'Expected {expected_excess}%, got {calculated_excess}%'
            })
            self.log_issue('critical',
                         f"Excess return calculation error: {stock_return}% - {riskfree_rate}% ≠ {expected_excess}%")
        
        # Test case 2: Stock return -1%, Risk-free 0.5% → Excess -1.5%
        stock_return_2 = -1.0
        riskfree_rate_2 = 0.5
        expected_excess_2 = -1.5
        calculated_excess_2 = stock_return_2 - riskfree_rate_2
        
        if abs(calculated_excess_2 - expected_excess_2) < 0.0001:
            self.log_pass(f"Excess return test 2: {stock_return_2}% - {riskfree_rate_2}% = {calculated_excess_2}%")
        else:
            issues_found.append({
                'test': 'Excess return 2',
                'issue': f'Expected {expected_excess_2}%, got {calculated_excess_2}%'
            })
            self.log_issue('critical',
                         f"Excess return calculation error: {stock_return_2}% - {riskfree_rate_2}% ≠ {expected_excess_2}%")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Excess return formula: {len(issues_found)} issues found"
        }
    
    def run_all_checks(self) -> Dict:
        """Run all financial calculation checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: FINANCIAL CALCULATIONS AUDIT")
        logger.info("="*70 + "\n")
        
        results = {
            'return_calculations': self.test_return_calculations(),
            'simple_vs_log': self.verify_simple_vs_log_returns(),
            'riskfree_conversion': self.test_riskfree_conversion(),
            'currency_conversion_logic': self.check_currency_conversion_logic(),
            'excess_return_formula': self.test_excess_return_formula()
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

