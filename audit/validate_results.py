"""
validate_results.py

Phase 5: Results Validation Audit
Validates beta values, R-squared, Fama-MacBeth results, and portfolio results.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict

from analysis.config import RESULTS_DATA_DIR, RESULTS_REPORTS_DIR

logger = logging.getLogger(__name__)


class ResultsAudit:
    """Audit class for results validation."""
    
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
    
    def validate_beta_values(self) -> Dict:
        """Validate beta values are reasonable."""
        logger.info("="*70)
        logger.info("Validating beta values")
        logger.info("="*70)
        
        issues_found = []
        
        capm_results_file = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
        if not os.path.exists(capm_results_file):
            return {'passed': False, 'issues': ['CAPM results file missing']}
        
        capm_results = pd.read_csv(capm_results_file)
        
        if 'beta' not in capm_results.columns:
            return {'passed': False, 'issues': ['Beta column missing']}
        
        betas = capm_results['beta'].dropna()
        
        # Check reasonable ranges: most betas between 0.3 and 1.5
        in_range = ((betas >= 0.3) & (betas <= 1.5)).sum()
        pct_in_range = in_range / len(betas) * 100
        
        if pct_in_range < 80:
            issues_found.append({
                'issue': f'Only {pct_in_range:.1f}% of betas in reasonable range (0.3-1.5)',
                'pct': pct_in_range
            })
            self.log_issue('warning',
                         f"Only {pct_in_range:.1f}% of betas in reasonable range")
        else:
            self.log_pass(f"{pct_in_range:.1f}% of betas in reasonable range (0.3-1.5)")
        
        # Check for extreme values
        extreme_high = (betas > 3.0).sum()
        extreme_low = (betas < -1.0).sum()
        
        if extreme_high > 0:
            issues_found.append({
                'issue': f'{extreme_high} betas > 3.0 (extreme)',
                'count': extreme_high
            })
            self.log_issue('warning', f"{extreme_high} extreme high betas (>3.0)")
        
        if extreme_low > 0:
            issues_found.append({
                'issue': f'{extreme_low} betas < -1.0 (extreme)',
                'count': extreme_low
            })
            self.log_issue('warning', f"{extreme_low} extreme low betas (<-1.0)")
        
        # Check distribution
        mean_beta = betas.mean()
        median_beta = betas.median()
        
        if abs(mean_beta - 0.7) > 0.3:
            issues_found.append({
                'issue': f'Mean beta {mean_beta:.3f} differs from expected ~0.7',
                'mean': mean_beta
            })
            self.log_issue('warning',
                         f"Mean beta {mean_beta:.3f} differs from expected ~0.7")
        else:
            self.log_pass(f"Mean beta: {mean_beta:.3f} (expected ~0.7)")
        
        # Check statistical significance
        if 'pvalue_beta' in capm_results.columns:
            significant = (capm_results['pvalue_beta'] < 0.05).sum()
            pct_significant = significant / len(capm_results) * 100
            
            if pct_significant < 90:
                issues_found.append({
                    'issue': f'Only {pct_significant:.1f}% of betas are significant (p<0.05)',
                    'pct': pct_significant
                })
                self.log_issue('warning',
                             f"Only {pct_significant:.1f}% of betas are significant")
            else:
                self.log_pass(f"{pct_significant:.1f}% of betas are significant (p<0.05)")
        
        return {
            'passed': len([i for i in issues_found if 'critical' in str(i)]) == 0,
            'issues': issues_found,
            'summary': f"Beta validation: {len(issues_found)} issues found"
        }
    
    def validate_r_squared(self) -> Dict:
        """Validate R-squared values."""
        logger.info("="*70)
        logger.info("Validating R-squared values")
        logger.info("="*70)
        
        issues_found = []
        
        capm_results_file = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
        if not os.path.exists(capm_results_file):
            return {'passed': False, 'issues': ['CAPM results file missing']}
        
        capm_results = pd.read_csv(capm_results_file)
        
        if 'r_squared' not in capm_results.columns:
            return {'passed': False, 'issues': ['R-squared column missing']}
        
        r2 = capm_results['r_squared'].dropna()
        
        # Check reasonable ranges: R² between 0.05 and 0.50
        in_range = ((r2 >= 0.05) & (r2 <= 0.50)).sum()
        pct_in_range = in_range / len(r2) * 100
        
        if pct_in_range < 80:
            issues_found.append({
                'issue': f'Only {pct_in_range:.1f}% of R² in reasonable range (0.05-0.50)',
                'pct': pct_in_range
            })
            self.log_issue('warning',
                         f"Only {pct_in_range:.1f}% of R² in reasonable range")
        else:
            self.log_pass(f"{pct_in_range:.1f}% of R² in reasonable range (0.05-0.50)")
        
        # Check average R²
        mean_r2 = r2.mean()
        if abs(mean_r2 - 0.24) > 0.1:
            issues_found.append({
                'issue': f'Mean R² {mean_r2:.3f} differs from expected ~0.24',
                'mean': mean_r2
            })
            self.log_issue('warning',
                         f"Mean R² {mean_r2:.3f} differs from expected ~0.24")
        else:
            self.log_pass(f"Mean R²: {mean_r2:.3f} (expected ~0.24)")
        
        # Check for very low R² (may indicate data issues)
        very_low = (r2 < 0.05).sum()
        if very_low > len(r2) * 0.1:
            issues_found.append({
                'issue': f'{very_low} stocks with R² < 0.05 (may indicate data issues)',
                'count': very_low
            })
            self.log_issue('warning',
                         f"{very_low} stocks with very low R² (<0.05)")
        else:
            self.log_pass(f"{very_low} stocks with very low R² (<0.05)")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"R-squared validation: {len(issues_found)} issues found"
        }
    
    def validate_fama_macbeth_results(self) -> Dict:
        """Validate Fama-MacBeth results."""
        logger.info("="*70)
        logger.info("Validating Fama-MacBeth results")
        logger.info("="*70)
        
        issues_found = []
        
        fm_summary_file = os.path.join(RESULTS_REPORTS_DIR, "fama_macbeth_summary.csv")
        if not os.path.exists(fm_summary_file):
            return {'passed': False, 'issues': ['Fama-MacBeth summary file missing']}
        
        fm_summary = pd.read_csv(fm_summary_file)
        
        # Extract gamma_1 and gamma_0
        # Try various possible column names
        gamma1_col = None
        gamma0_col = None
        t1_col = None
        t0_col = None
        p1_col = None
        
        for col in fm_summary.columns:
            col_lower = col.lower()
            # Gamma 1
            if ('gamma_1' in col_lower or 'γ₁' in col or 'gamma1' in col_lower) and ('mean' in col_lower or 'avg' in col_lower):
                gamma1_col = col
            # Gamma 0
            if ('gamma_0' in col_lower or 'γ₀' in col or 'gamma0' in col_lower) and ('mean' in col_lower or 'avg' in col_lower):
                gamma0_col = col
            # T-statistics
            if ('t(' in col or 'tstat' in col_lower or 't_stat' in col_lower) and ('gamma_1' in col_lower or 'gamma1' in col_lower):
                t1_col = col
            if ('t(' in col or 'tstat' in col_lower or 't_stat' in col_lower) and ('gamma_0' in col_lower or 'gamma0' in col_lower):
                t0_col = col
            # P-values
            if ('p-value' in col_lower or 'pvalue' in col_lower or 'p_value' in col_lower) and ('gamma_1' in col_lower or 'gamma1' in col_lower):
                p1_col = col
        
        if gamma1_col and gamma0_col:
            gamma1 = fm_summary[gamma1_col].iloc[0]
            gamma0 = fm_summary[gamma0_col].iloc[0]
            
            # Check gamma_1 sign and magnitude
            if gamma1 < 0:
                self.log_pass(f"gamma_1 is negative: {gamma1:.4f} (consistent with CAPM rejection)")
            else:
                self.log_issue('warning',
                             f"gamma_1 is positive: {gamma1:.4f} (unusual if CAPM is rejected)")
            
            # Check t-statistic
            if t1_col:
                t1 = fm_summary[t1_col].iloc[0]
                if abs(t1) < 1.96:  # Not significant at 5% level
                    self.log_pass(f"gamma_1 t-statistic: {t1:.4f} (not significant, consistent with CAPM rejection)")
                else:
                    issues_found.append({
                        'issue': f'gamma_1 is significant (t={t1:.4f}), but CAPM is rejected',
                        't_stat': t1
                    })
                    self.log_issue('warning',
                                 f"gamma_1 is significant (t={t1:.4f})")
            
            # Check p-value
            if p1_col:
                p1 = fm_summary[p1_col].iloc[0]
                if p1 > 0.05:
                    self.log_pass(f"gamma_1 p-value: {p1:.4f} (not significant)")
                else:
                    issues_found.append({
                        'issue': f'gamma_1 p-value {p1:.4f} is significant',
                        'p_value': p1
                    })
                    self.log_issue('warning',
                                 f"gamma_1 p-value {p1:.4f} is significant")
            
            # Check gamma_0 magnitude
            if abs(gamma0) > 2.0:  # > 2% monthly is high
                issues_found.append({
                    'issue': f'gamma_0 {gamma0:.4f}% is high (verify risk-free rate)',
                    'gamma0': gamma0
                })
                self.log_issue('warning',
                             f"gamma_0 {gamma0:.4f}% is high (may indicate risk-free rate issue)")
            else:
                self.log_pass(f"gamma_0: {gamma0:.4f}% (reasonable)")
        else:
            issues_found.append({
                'issue': 'Fama-MacBeth coefficients not found'
            })
            self.log_issue('critical', "Fama-MacBeth coefficients not found")
        
        return {
            'passed': len([i for i in issues_found if 'critical' in str(i)]) == 0,
            'issues': issues_found,
            'summary': f"Fama-MacBeth validation: {len(issues_found)} issues found"
        }
    
    def validate_portfolio_results(self) -> Dict:
        """Validate beta-sorted portfolio results."""
        logger.info("="*70)
        logger.info("Validating portfolio results")
        logger.info("="*70)
        
        issues_found = []
        
        portfolio_file = os.path.join(RESULTS_REPORTS_DIR, "beta_sorted_portfolios.csv")
        if not os.path.exists(portfolio_file):
            return {'passed': False, 'issues': ['Portfolio results file missing']}
        
        portfolios = pd.read_csv(portfolio_file)
        
        # Find beta and return columns (may have different names)
        beta_col = None
        return_col = None
        
        for col in portfolios.columns:
            col_lower = col.lower()
            # Beta column: look for 'beta' but not 'portfolio_beta' in the check
            # Actually 'portfolio_beta' is fine, we want that
            if 'beta' in col_lower and beta_col is None:
                # Prefer 'portfolio_beta' or columns with 'beta' but not 'min' or 'max'
                if 'portfolio_beta' in col_lower or ('beta' in col_lower and 'min' not in col_lower and 'max' not in col_lower):
                    beta_col = col
            # Return column: look for 'avg_return' specifically
            if 'avg_return' in col_lower or (col_lower == 'avg_return'):
                return_col = col
        
        # Fallback if not found
        if beta_col is None:
            if 'portfolio_beta' in portfolios.columns:
                beta_col = 'portfolio_beta'
        if return_col is None:
            if 'avg_return' in portfolios.columns:
                return_col = 'avg_return'
        
        # Check portfolio construction
        if beta_col and return_col:
            # Portfolios should be sorted by beta
            betas = portfolios[beta_col].values
            if not np.all(betas[:-1] <= betas[1:]):  # Check if sorted
                issues_found.append({
                    'issue': 'Portfolios are not sorted by beta'
                })
                self.log_issue('critical', "Portfolios are not sorted by beta")
            else:
                self.log_pass("Portfolios are correctly sorted by beta")
            
            # Check for negative slope (higher beta → lower return)
            returns = portfolios[return_col].values
            # Calculate slope
            if len(betas) > 1:
                slope = np.polyfit(betas, returns, 1)[0]
                if slope < 0:
                    self.log_pass(f"Portfolio slope is negative: {slope:.4f} (consistent with CAPM rejection)")
                else:
                    issues_found.append({
                        'issue': f'Portfolio slope is positive: {slope:.4f} (unexpected if CAPM rejected)',
                        'slope': slope
                    })
                    self.log_issue('warning',
                                 f"Portfolio slope is positive: {slope:.4f} (contradicts Fama-MacBeth negative gamma_1)")
        else:
            issues_found.append({
                'issue': 'Portfolio file missing required columns'
            })
            self.log_issue('critical', "Portfolio file missing required columns")
        
        return {
            'passed': len([i for i in issues_found if 'critical' in str(i)]) == 0,
            'issues': issues_found,
            'summary': f"Portfolio validation: {len(issues_found)} issues found"
        }
    
    def run_all_checks(self) -> Dict:
        """Run all results validation checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 5: RESULTS VALIDATION AUDIT")
        logger.info("="*70 + "\n")
        
        results = {
            'beta_values': self.validate_beta_values(),
            'r_squared': self.validate_r_squared(),
            'fama_macbeth': self.validate_fama_macbeth_results(),
            'portfolios': self.validate_portfolio_results()
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

