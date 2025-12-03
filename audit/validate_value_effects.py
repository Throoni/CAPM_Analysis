"""
validate_value_effects.py

Validates value effects analysis (book-to-market ratios and alpha relationships).
Checks data quality, portfolio formation, and statistical test correctness.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict

from analysis.config import (
    RESULTS_DATA_DIR,
    RESULTS_REPORTS_DIR,
    RESULTS_PLOTS_DIR
)

logger = logging.getLogger(__name__)


class ValueEffectsAudit:
    """Audit class for value effects analysis."""
    
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
    
    def check_files_exist(self) -> bool:
        """Check if value effects result files exist."""
        logger.info("="*70)
        logger.info("Checking value effects files")
        logger.info("="*70)
        
        required_files = [
            os.path.join(RESULTS_REPORTS_DIR, "value_effects_portfolios.csv"),
            os.path.join(RESULTS_REPORTS_DIR, "value_effects_test_results.csv"),
            os.path.join(RESULTS_PLOTS_DIR, "value_effect_analysis.png")
        ]
        
        all_exist = True
        for file_path in required_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                self.log_pass(f"File exists: {os.path.basename(file_path)} ({file_size:,} bytes)")
            else:
                self.log_issue('critical', f"Required file not found: {os.path.basename(file_path)}")
                all_exist = False
        
        return all_exist
    
    def validate_portfolio_formation(self) -> Dict:
        """Validate value/growth portfolio formation."""
        logger.info("="*70)
        logger.info("Validating portfolio formation")
        logger.info("="*70)
        
        portfolios_file = os.path.join(RESULTS_REPORTS_DIR, "value_effects_portfolios.csv")
        
        if not os.path.exists(portfolios_file):
            return {'passed': False, 'issues': self.issues}
        
        df = pd.read_csv(portfolios_file)
        issues_found = []
        
        # Check required columns
        required_cols = ['portfolio', 'avg_book_to_market', 'avg_alpha']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues_found.append({
                'issue': f'Missing required columns: {missing_cols}',
                'severity': 'critical'
            })
            self.log_issue('critical', f"Missing columns: {missing_cols}")
        else:
            self.log_pass("All required columns present")
        
        # Check portfolio ordering (should be sorted by B/M: P1=growth, P5=value)
        if 'portfolio' in df.columns and 'avg_book_to_market' in df.columns:
            df_sorted = df.sort_values('avg_book_to_market')
            portfolios = df_sorted['portfolio'].values
            
            # Check if portfolios are in expected order
            expected_order = ['P1', 'P2', 'P3', 'P4', 'P5']
            if len(portfolios) == len(expected_order):
                if list(portfolios) != expected_order:
                    issues_found.append({
                        'issue': f'Portfolios not in expected B/M order: {list(portfolios)}',
                        'severity': 'warning'
                    })
                    self.log_issue('warning', "Portfolios may not be correctly sorted by B/M")
                else:
                    self.log_pass("Portfolios correctly sorted by B/M (P1=growth, P5=value)")
            
            # Check B/M increases from P1 to P5
            bm_values = df_sorted['avg_book_to_market'].values
            if len(bm_values) > 1:
                is_increasing = all(bm_values[i] <= bm_values[i+1] for i in range(len(bm_values)-1))
                if not is_increasing:
                    issues_found.append({
                        'issue': 'Book-to-market ratios do not increase from P1 to P5',
                        'severity': 'critical'
                    })
                    self.log_issue('critical', "B/M ratios not increasing (portfolio formation error)")
                else:
                    self.log_pass("B/M ratios increase from growth (P1) to value (P5)")
        
        # Check portfolio sizes (should be roughly equal)
        if 'n_stocks' in df.columns:
            n_stocks = df['n_stocks'].values
            if len(n_stocks) > 0:
                min_stocks = n_stocks.min()
                max_stocks = n_stocks.max()
                size_diff = max_stocks - min_stocks
                
                # Allow up to 2 stock difference (for quintiles with 219 stocks: 43-44 stocks each)
                if size_diff > 2:
                    issues_found.append({
                        'issue': f'Portfolio sizes vary significantly: min={min_stocks}, max={max_stocks} (diff={size_diff})',
                        'severity': 'warning'
                    })
                    self.log_issue('warning', f"Portfolio size imbalance: {size_diff} stocks difference")
                else:
                    self.log_pass(f"Portfolio sizes are balanced (min={min_stocks}, max={max_stocks})")
        
        # Check B/M reasonableness
        if 'avg_book_to_market' in df.columns:
            bm_values = df['avg_book_to_market'].dropna()
            if len(bm_values) > 0:
                # B/M ratios should be positive
                negative_bm = (bm_values < 0).sum()
                if negative_bm > 0:
                    issues_found.append({
                        'issue': f'{negative_bm} portfolios with negative B/M ratios',
                        'severity': 'critical'
                    })
                    self.log_issue('critical', f"{negative_bm} portfolios with negative B/M")
                else:
                    self.log_pass("All B/M ratios are positive")
                
                # B/M ratios should be reasonable (typically 0.1 to 5.0)
                extreme_bm = ((bm_values < 0.01) | (bm_values > 10.0)).sum()
                if extreme_bm > 0:
                    issues_found.append({
                        'issue': f'{extreme_bm} portfolios with extreme B/M ratios (<0.01 or >10.0)',
                        'severity': 'warning'
                    })
                    self.log_issue('warning', f"{extreme_bm} portfolios with extreme B/M ratios")
        
        self.issues.extend(issues_found)
        
        return {
            'passed': len([i for i in issues_found if i.get('severity') == 'critical']) == 0,
            'issues': issues_found,
            'summary': f"Portfolio formation validation: {len([i for i in issues_found if i.get('severity') == 'critical'])} critical, {len([i for i in issues_found if i.get('severity') == 'warning'])} warnings"
        }
    
    def validate_statistical_tests(self) -> Dict:
        """Validate statistical test results."""
        logger.info("="*70)
        logger.info("Validating statistical tests")
        logger.info("="*70)
        
        test_file = os.path.join(RESULTS_REPORTS_DIR, "value_effects_test_results.csv")
        
        if not os.path.exists(test_file):
            self.log_issue('warning', "Value effects test results file not found")
            return {'passed': True, 'issues': []}
        
        df = pd.read_csv(test_file)
        issues_found = []
        
        # Check required columns
        required_cols = ['correlation', 'regression_slope', 'regression_pvalue', 'alpha_spread']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues_found.append({
                'issue': f'Missing columns: {missing_cols}',
                'severity': 'critical'
            })
            self.log_issue('critical', f"Missing columns: {missing_cols}")
        else:
            self.log_pass("All required test result columns present")
        
        # Validate correlation
        if 'correlation' in df.columns:
            corr = df['correlation'].iloc[0]
            if not pd.isna(corr):
                # Correlation should be between -1 and 1
                if abs(corr) > 1.0:
                    issues_found.append({
                        'issue': f'Correlation out of range: {corr:.4f} (should be [-1, 1])',
                        'severity': 'critical'
                    })
                    self.log_issue('critical', f"Correlation out of range: {corr:.4f}")
                else:
                    self.log_pass(f"Correlation in valid range: {corr:.4f}")
        
        # Validate p-value
        if 'regression_pvalue' in df.columns:
            pvalue = df['regression_pvalue'].iloc[0]
            if not pd.isna(pvalue):
                # P-value should be between 0 and 1
                if pvalue < 0 or pvalue > 1:
                    issues_found.append({
                        'issue': f'P-value out of range: {pvalue:.4f} (should be [0, 1])',
                        'severity': 'critical'
                    })
                    self.log_issue('critical', f"P-value out of range: {pvalue:.4f}")
                else:
                    self.log_pass(f"P-value in valid range: {pvalue:.4f}")
        
        # Validate alpha spread calculation
        if 'alpha_spread' in df.columns and 'value_alpha' in df.columns and 'growth_alpha' in df.columns:
            spread = df['alpha_spread'].iloc[0]
            value_alpha = df['value_alpha'].iloc[0]
            growth_alpha = df['growth_alpha'].iloc[0]
            
            if not pd.isna(spread) and not pd.isna(value_alpha) and not pd.isna(growth_alpha):
                expected_spread = value_alpha - growth_alpha
                if abs(expected_spread - spread) > 0.0001:
                    issues_found.append({
                        'issue': f'Alpha spread mismatch: expected {expected_spread:.4f}, got {spread:.4f}',
                        'severity': 'warning'
                    })
                    self.log_issue('warning', "Alpha spread calculation may be incorrect")
                else:
                    self.log_pass(f"Alpha spread correct: {spread:.4f}%")
        
        self.issues.extend(issues_found)
        
        return {
            'passed': len([i for i in issues_found if i.get('severity') == 'critical']) == 0,
            'issues': issues_found,
            'summary': f"Statistical tests validation: {len([i for i in issues_found if i.get('severity') == 'critical'])} critical, {len([i for i in issues_found if i.get('severity') == 'warning'])} warnings"
        }
    
    def validate_consistency_with_capm(self) -> Dict:
        """Validate that value effects alphas are consistent with CAPM results."""
        logger.info("="*70)
        logger.info("Validating consistency with CAPM alphas")
        logger.info("="*70)
        
        capm_file = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
        portfolios_file = os.path.join(RESULTS_REPORTS_DIR, "value_effects_portfolios.csv")
        
        if not os.path.exists(capm_file) or not os.path.exists(portfolios_file):
            self.log_issue('info', "Cannot validate consistency: files not found")
            return {'passed': True, 'issues': []}
        
        capm = pd.read_csv(capm_file)
        portfolios = pd.read_csv(portfolios_file)
        
        issues_found = []
        
        # Get overall average alpha from CAPM
        valid_capm = capm[capm['is_valid'] == True]
        overall_alpha = valid_capm['alpha'].mean()
        
        # Get weighted average alpha from portfolios
        if 'avg_alpha' in portfolios.columns and 'n_stocks' in portfolios.columns:
            portfolio_alphas = portfolios['avg_alpha'].values
            portfolio_sizes = portfolios['n_stocks'].values
            
            # Weighted average
            weighted_alpha = np.average(portfolio_alphas, weights=portfolio_sizes)
            
            # Compare (allow small differences due to portfolio formation)
            diff = abs(overall_alpha - weighted_alpha)
            if diff > 0.1:  # Allow 0.1% difference
                issues_found.append({
                    'issue': f'Alpha mismatch: CAPM overall={overall_alpha:.4f}%, Portfolio weighted={weighted_alpha:.4f}% (diff={diff:.4f}%)',
                    'severity': 'warning'
                })
                self.log_issue('warning', f"Alpha consistency check: difference of {diff:.4f}%")
            else:
                self.log_pass(f"Alphas consistent: CAPM={overall_alpha:.4f}%, Portfolio={weighted_alpha:.4f}%")
        
        self.issues.extend(issues_found)
        
        return {
            'passed': len([i for i in issues_found if i.get('severity') == 'critical']) == 0,
            'issues': issues_found,
            'summary': f"Consistency validation: {len(issues_found)} issues found"
        }
    
    def run_all_checks(self) -> Dict:
        """Run all value effects validation checks."""
        logger.info("="*70)
        logger.info("VALUE EFFECTS VALIDATION")
        logger.info("="*70)
        
        # Check 1: Files exist
        files_exist = self.check_files_exist()
        
        if not files_exist:
            return {
                'passed': False,
                'issues': self.issues,
                'summary': 'Value effects validation failed: required files not found'
            }
        
        # Check 2: Validate portfolio formation
        portfolio_results = self.validate_portfolio_formation()
        
        # Check 3: Validate statistical tests
        test_results = self.validate_statistical_tests()
        
        # Check 4: Consistency with CAPM
        consistency_results = self.validate_consistency_with_capm()
        
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
    
    audit = ValueEffectsAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("VALUE EFFECTS VALIDATION COMPLETE")
    print("="*70)
    print(f"Passed: {results['passed']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")

