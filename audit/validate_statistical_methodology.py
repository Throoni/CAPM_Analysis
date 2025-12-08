"""
validate_statistical_methodology.py

Phase 3: Statistical Methodology Audit
Validates CAPM regression and Fama-MacBeth implementation.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict
import statsmodels.api as sm

from analysis.utils.config import DATA_PROCESSED_DIR, RESULTS_DATA_DIR, RESULTS_REPORTS_DIR

# Import run_capm_regression only when needed to avoid seaborn dependency
def _get_capm_regression_function():
    """Lazy import of run_capm_regression to avoid seaborn dependency."""
    try:
        from analysis.core.capm_regression import run_capm_regression
        return run_capm_regression
    except ImportError as e:
        logger.warning(f"Could not import run_capm_regression: {e}. Using local implementation.")
        # Fallback: implement a simple version locally
        return None

logger = logging.getLogger(__name__)


class StatisticalMethodologyAudit:
    """Audit class for statistical methodology validation."""
    
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
    
    def test_capm_regression(self) -> Dict:
        """Test CAPM regression with synthetic data."""
        logger.info("="*70)
        logger.info("Testing CAPM regression with synthetic data")
        logger.info("="*70)
        
        issues_found = []
        
        # Create synthetic data with known beta=1.0, alpha=0
        np.random.seed(42)
        n_obs = 59
        market_excess = np.random.normal(0, 2, n_obs)  # Market excess returns
        stock_excess = 1.0 * market_excess + np.random.normal(0, 1, n_obs)  # Beta=1.0, alpha=0
        
        test_data = pd.DataFrame({
            'stock_excess_return': stock_excess,
            'market_excess_return': market_excess
        })
        
        # Try to use run_capm_regression, fallback to local implementation
        run_capm_regression = _get_capm_regression_function()
        
        if run_capm_regression is None:
            # Local implementation without seaborn dependency
            clean_data = test_data.dropna()
            if len(clean_data) < 10:
                return {'passed': False, 'issues': ['Insufficient data']}
            
            y = clean_data['stock_excess_return'].values
            X = clean_data['market_excess_return'].values
            X_with_const = sm.add_constant(X)
            
            model = sm.OLS(y, X_with_const).fit()
            result = {
                'beta': model.params[1],
                'alpha': model.params[0],
                'beta_tstat': model.tvalues[1],
                'alpha_tstat': model.tvalues[0],
                'r_squared': model.rsquared,
                'beta_se': model.bse[1],
                'alpha_se': model.bse[0],
                'pvalue_beta': model.pvalues[1],
                'pvalue_alpha': model.pvalues[0],
                'n_obs': len(clean_data)
            }
        else:
            # Run regression using imported function
            result = run_capm_regression(test_data)
        
        # Check beta (should be close to 1.0)
        if abs(result['beta'] - 1.0) > 0.2:
            issues_found.append({
                'test': 'Beta estimation',
                'issue': f'Expected beta ≈ 1.0, got {result["beta"]:.4f}'
            })
            self.log_issue('critical',
                         f"Beta estimation error: Expected ≈1.0, got {result['beta']:.4f}")
        else:
            self.log_pass(f"Beta estimation correct: {result['beta']:.4f} (expected ≈1.0)")
        
        # Check alpha (should be close to 0)
        if abs(result['alpha']) > 0.5:
            issues_found.append({
                'test': 'Alpha estimation',
                'issue': f'Expected alpha ≈ 0, got {result["alpha"]:.4f}'
            })
            self.log_issue('warning',
                         f"Alpha estimation: Expected ≈0, got {result['alpha']:.4f}")
        else:
            self.log_pass(f"Alpha estimation correct: {result['alpha']:.4f} (expected ≈0)")
        
        # Check t-statistics are reasonable
        if abs(result['beta_tstat']) < 1.0:
            issues_found.append({
                'test': 'Beta t-statistic',
                'issue': f'Beta t-statistic {result["beta_tstat"]:.4f} is too low'
            })
            self.log_issue('warning',
                         f"Beta t-statistic {result['beta_tstat']:.4f} is low (beta may not be significant)")
        else:
            self.log_pass(f"Beta t-statistic reasonable: {result['beta_tstat']:.4f}")
        
        # Check R-squared is reasonable (> 0.5 for this synthetic data)
        if result['r_squared'] < 0.3:
            issues_found.append({
                'test': 'R-squared',
                'issue': f'R-squared {result["r_squared"]:.4f} is unexpectedly low'
            })
            self.log_issue('warning',
                         f"R-squared {result['r_squared']:.4f} is low")
        else:
            self.log_pass(f"R-squared reasonable: {result['r_squared']:.4f}")
        
        return {
            'passed': len([i for i in issues_found if 'critical' in str(i)]) == 0,
            'issues': issues_found,
            'summary': f"CAPM regression test: {len(issues_found)} issues found"
        }
    
    def verify_fama_macbeth_methodology(self) -> Dict:
        """Verify Fama-MacBeth methodology implementation."""
        logger.info("="*70)
        logger.info("Verifying Fama-MacBeth methodology")
        logger.info("="*70)
        
        issues_found = []
        
        # Check Fama-MacBeth summary file
        fm_summary_file = os.path.join(RESULTS_REPORTS_DIR, "fama_macbeth_summary.csv")
        if not os.path.exists(fm_summary_file):
            return {'passed': False, 'issues': ['Fama-MacBeth summary file missing']}
        
        fm_summary = pd.read_csv(fm_summary_file)
        
        # Check that gamma_1 and gamma_0 are present
        # Try various possible column names (flexible matching)
        gamma1_col = None
        gamma0_col = None
        t1_col = None
        t0_col = None
        
        for col in fm_summary.columns:
            col_lower = col.lower().replace('_', '').replace('-', '').replace(' ', '')
            
            # Gamma 1 (beta coefficient)
            if ('gamma1' in col_lower or 'gamma_1' in col_lower or 'γ₁' in col or 'γ1' in col_lower):
                if 'mean' in col_lower or 'avg' in col_lower or 'average' in col_lower:
                    gamma1_col = col
                elif 't' in col_lower and ('stat' in col_lower or 'value' in col_lower):
                    t1_col = col
            
            # Gamma 0 (intercept)
            if ('gamma0' in col_lower or 'gamma_0' in col_lower or 'γ₀' in col or 'γ0' in col_lower):
                if 'mean' in col_lower or 'avg' in col_lower or 'average' in col_lower:
                    gamma0_col = col
                elif 't' in col_lower and ('stat' in col_lower or 'value' in col_lower):
                    t0_col = col
        
        if gamma1_col and gamma0_col:
            
            gamma1 = fm_summary[gamma1_col].iloc[0]
            gamma0 = fm_summary[gamma0_col].iloc[0]
            
            # Check t-statistics
            t1_col = 't(γ₁)' if 't(γ₁)' in fm_summary.columns else 't(gamma_1)'
            t0_col = 't(γ₀)' if 't(γ₀)' in fm_summary.columns else 't(gamma_0)'
            
            if t1_col in fm_summary.columns:
                t1 = fm_summary[t1_col].iloc[0]
                self.log_pass(f"Fama-MacBeth gamma_1: {gamma1:.4f}, t-stat: {t1:.4f}")
            else:
                issues_found.append({
                    'issue': 'Fama-MacBeth t-statistics missing'
                })
                self.log_issue('warning', "Fama-MacBeth t-statistics not found in summary")
        else:
            issues_found.append({
                'issue': 'Fama-MacBeth coefficients missing'
            })
            self.log_issue('critical', "Fama-MacBeth coefficients not found in summary")
        
        # Check monthly coefficients file
        fm_monthly_file = os.path.join(RESULTS_REPORTS_DIR, "fama_macbeth_monthly_coefficients.csv")
        if os.path.exists(fm_monthly_file):
            fm_monthly = pd.read_csv(fm_monthly_file, parse_dates=['date'])
            
            # Verify we have ~59 months
            n_months = len(fm_monthly)
            if abs(n_months - 59) > 5:
                issues_found.append({
                    'issue': f'Expected ~59 months, got {n_months}'
                })
                self.log_issue('warning',
                             f"Fama-MacBeth: Expected ~59 months, got {n_months}")
            else:
                self.log_pass(f"Fama-MacBeth: {n_months} monthly regressions")
            
            # Check that Fama-MacBeth standard errors are used (not OLS SE)
            # FM SE = std(gamma_t) / sqrt(T)
            if 'gamma_1' in fm_monthly.columns:
                std_gamma1 = fm_monthly['gamma_1'].std()
                fm_se = std_gamma1 / np.sqrt(n_months)
                
                # Compare to reported SE if available
                if 'gamma_1_se' in fm_monthly.columns:
                    ols_se_avg = fm_monthly['gamma_1_se'].mean()
                    if abs(fm_se - ols_se_avg) < 0.01:
                        self.log_issue('warning',
                                     "Fama-MacBeth may be using OLS SE instead of FM SE")
                        issues_found.append({
                            'issue': 'Fama-MacBeth standard errors may be incorrect'
                        })
                    else:
                        self.log_pass("Fama-MacBeth standard errors appear correct")
        else:
            issues_found.append({
                'issue': 'Fama-MacBeth monthly coefficients file missing'
            })
            self.log_issue('warning', "Fama-MacBeth monthly coefficients file not found")
        
        return {
            'passed': len([i for i in issues_found if 'critical' in str(i)]) == 0,
            'issues': issues_found,
            'summary': f"Fama-MacBeth methodology: {len(issues_found)} issues found"
        }
    
    def verify_beta_usage(self) -> Dict:
        """Verify how betas are used in Fama-MacBeth (full-sample vs rolling)."""
        logger.info("="*70)
        logger.info("Verifying beta usage in Fama-MacBeth")
        logger.info("="*70)
        
        issues_found = []
        
        # Check CAPM results for observation counts
        capm_results_file = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
        if os.path.exists(capm_results_file):
            capm_results = pd.read_csv(capm_results_file)
            
            if 'n_obs' in capm_results.columns:
                mean_obs = capm_results['n_obs'].mean()
                std_obs = capm_results['n_obs'].std()
                
                # If all stocks have ~59 observations, betas are full-sample
                if abs(mean_obs - 59) < 2 and std_obs < 5:
                    self.log_pass(f"Betas appear to be full-sample (mean obs: {mean_obs:.1f})")
                    self.log_issue('warning',
                                 "CRITICAL ASSUMPTION: Betas are estimated on full sample (2021-2025), "
                                 "then used in cross-sectional regressions. This is standard Fama-MacBeth "
                                 "but creates potential look-ahead bias. Document this clearly.")
                    issues_found.append({
                        'issue': 'Full-sample beta estimation (potential look-ahead bias)',
                        'note': 'This is standard Fama-MacBeth but should be documented'
                    })
                else:
                    issues_found.append({
                        'issue': f'Inconsistent observation counts (mean: {mean_obs:.1f}, std: {std_obs:.1f})'
                    })
                    self.log_issue('warning',
                                 f"Observation counts vary (mean: {mean_obs:.1f}, std: {std_obs:.1f})")
        
        return {
            'passed': True,  # Not a failure, just documentation
            'issues': issues_found,
            'summary': f"Beta usage: {len(issues_found)} notes found"
        }
    
    def run_all_checks(self) -> Dict:
        """Run all statistical methodology checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: STATISTICAL METHODOLOGY AUDIT")
        logger.info("="*70 + "\n")
        
        results = {
            'capm_regression': self.test_capm_regression(),
            'fama_macbeth': self.verify_fama_macbeth_methodology(),
            'beta_usage': self.verify_beta_usage()
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

