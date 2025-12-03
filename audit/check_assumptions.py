"""
check_assumptions.py

Phase 4.3: Assumption Violations
Checks CAPM and statistical assumptions.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
try:
    from statsmodels.stats.diagnostic import jarque_bera
except ImportError:
    # Fallback: use scipy.stats.jarque_bera if statsmodels version doesn't have it
    try:
        from scipy.stats import jarque_bera
    except ImportError:
        # If neither works, define a simple wrapper
        def jarque_bera(residuals):
            from scipy import stats
            return stats.jarque_bera(residuals)

from analysis.config import DATA_PROCESSED_DIR, RESULTS_DATA_DIR

logger = logging.getLogger(__name__)


class AssumptionsAudit:
    """Audit class for assumption checks."""
    
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
    
    def check_linearity(self) -> Dict:
        """Check linearity assumption (returns vs beta)."""
        logger.info("="*70)
        logger.info("Checking linearity assumption")
        logger.info("="*70)
        
        issues_found = []
        
        # Load CAPM results and panel data
        capm_results_file = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
        panel_file = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
        
        if not os.path.exists(capm_results_file) or not os.path.exists(panel_file):
            return {'passed': False, 'issues': ['Required files missing']}
        
        capm_results = pd.read_csv(capm_results_file)
        panel = pd.read_csv(panel_file, parse_dates=['date'])
        
        # Merge to get average returns and betas
        if 'ticker' in capm_results.columns and 'beta' in capm_results.columns:
            # Calculate average returns per stock
            avg_returns = panel.groupby(['country', 'ticker'])['stock_return'].mean().reset_index()
            avg_returns.columns = ['country', 'ticker', 'avg_return']
            
            # Merge with betas
            merged = avg_returns.merge(
                capm_results[['country', 'ticker', 'beta']],
                on=['country', 'ticker'],
                how='inner'
            )
            
            if len(merged) > 10:
                # Check correlation between beta and average return
                correlation = merged['beta'].corr(merged['avg_return'])
                
                if abs(correlation) < 0.1:
                    issues_found.append({
                        'issue': f'Weak linear relationship (correlation: {correlation:.3f})',
                        'correlation': correlation
                    })
                    self.log_issue('warning',
                                 f"Linearity assumption: Weak correlation ({correlation:.3f}) between beta and returns")
                else:
                    self.log_pass(f"Linearity check: Correlation = {correlation:.3f}")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Linearity: {len(issues_found)} issues found"
        }
    
    def check_homoscedasticity(self) -> Dict:
        """Check homoscedasticity (constant variance of residuals)."""
        logger.info("="*70)
        logger.info("Checking homoscedasticity assumption")
        logger.info("="*70)
        
        issues_found = []
        
        # Sample a few stocks and check residual plots
        capm_results_file = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
        panel_file = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
        
        if not os.path.exists(capm_results_file) or not os.path.exists(panel_file):
            return {'passed': False, 'issues': ['Required files missing']}
        
        capm_results = pd.read_csv(capm_results_file)
        panel = pd.read_csv(panel_file, parse_dates=['date'])
        
        # Check residuals for a sample of stocks
        sample_stocks = capm_results.head(10)
        heteroscedastic = 0
        
        for _, stock in sample_stocks.iterrows():
            country = stock['country']
            ticker = stock['ticker']
            
            stock_data = panel[(panel['country'] == country) & (panel['ticker'] == ticker)]
            stock_data = stock_data.dropna(subset=['stock_excess_return', 'market_excess_return'])
            
            if len(stock_data) > 10:
                y = stock_data['stock_excess_return'].values
                X = stock_data['market_excess_return'].values
                X = sm.add_constant(X)
                
                model = sm.OLS(y, X).fit()
                residuals = model.resid
                
                # Simple test: check if variance changes across time
                # Split residuals in half and compare variances
                n = len(residuals)
                var_first = residuals[:n//2].var()
                var_second = residuals[n//2:].var()
                
                if var_first > 0 and var_second > 0:
                    ratio = max(var_first, var_second) / min(var_first, var_second)
                    if ratio > 2.0:  # Variance differs by more than 2x
                        heteroscedastic += 1
        
        if heteroscedastic > 5:
            issues_found.append({
                'issue': f'{heteroscedastic}/10 sample stocks show heteroscedasticity',
                'count': heteroscedastic
            })
            self.log_issue('warning',
                         f"Heteroscedasticity detected in {heteroscedastic}/10 sample stocks")
        else:
            self.log_pass(f"Homoscedasticity check: {heteroscedastic}/10 stocks show issues")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Homoscedasticity: {len(issues_found)} issues found"
        }
    
    def check_autocorrelation(self) -> Dict:
        """Check independence assumption (no autocorrelation)."""
        logger.info("="*70)
        logger.info("Checking autocorrelation (independence)")
        logger.info("="*70)
        
        issues_found = []
        
        panel_file = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
        if not os.path.exists(panel_file):
            return {'passed': False, 'issues': ['Panel file missing']}
        
        panel = pd.read_csv(panel_file, parse_dates=['date'])
        
        # Check autocorrelation for a sample of stocks
        sample_stocks = panel.groupby(['country', 'ticker']).head(1).head(10)
        autocorrelated = 0
        
        for _, row in sample_stocks.iterrows():
            country = row['country']
            ticker = row['ticker']
            
            stock_data = panel[(panel['country'] == country) & (panel['ticker'] == ticker)]
            stock_data = stock_data.sort_values('date')
            stock_data = stock_data.dropna(subset=['stock_excess_return', 'market_excess_return'])
            
            if len(stock_data) > 20:
                y = stock_data['stock_excess_return'].values
                X = stock_data['market_excess_return'].values
                X = sm.add_constant(X)
                
                model = sm.OLS(y, X).fit()
                residuals = model.resid
                
                # Durbin-Watson test
                dw_stat = durbin_watson(residuals)
                # DW near 2 indicates no autocorrelation
                # DW < 1.5 or > 2.5 indicates autocorrelation
                if dw_stat < 1.5 or dw_stat > 2.5:
                    autocorrelated += 1
        
        if autocorrelated > 5:
            issues_found.append({
                'issue': f'{autocorrelated}/10 sample stocks show autocorrelation',
                'count': autocorrelated
            })
            self.log_issue('warning',
                         f"Autocorrelation detected in {autocorrelated}/10 sample stocks")
        else:
            self.log_pass(f"Autocorrelation check: {autocorrelated}/10 stocks show issues")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Autocorrelation: {len(issues_found)} issues found"
        }
    
    def check_normality(self) -> Dict:
        """Check normality of residuals."""
        logger.info("="*70)
        logger.info("Checking normality of residuals")
        logger.info("="*70)
        
        issues_found = []
        
        panel_file = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
        if not os.path.exists(panel_file):
            return {'passed': False, 'issues': ['Panel file missing']}
        
        panel = pd.read_csv(panel_file, parse_dates=['date'])
        
        # Check normality for a sample of stocks
        sample_stocks = panel.groupby(['country', 'ticker']).head(1).head(10)
        non_normal = 0
        
        for _, row in sample_stocks.iterrows():
            country = row['country']
            ticker = row['ticker']
            
            stock_data = panel[(panel['country'] == country) & (panel['ticker'] == ticker)]
            stock_data = stock_data.sort_values('date')
            stock_data = stock_data.dropna(subset=['stock_excess_return', 'market_excess_return'])
            
            if len(stock_data) > 20:
                y = stock_data['stock_excess_return'].values
                X = stock_data['market_excess_return'].values
                X = sm.add_constant(X)
                
                model = sm.OLS(y, X).fit()
                residuals = model.resid
                
                # Jarque-Bera test for normality
                jb_stat, jb_pvalue = jarque_bera(residuals)
                if jb_pvalue < 0.05:  # Reject normality at 5% level
                    non_normal += 1
        
        if non_normal > 7:
            issues_found.append({
                'issue': f'{non_normal}/10 sample stocks show non-normal residuals',
                'count': non_normal
            })
            self.log_issue('warning',
                         f"Non-normality detected in {non_normal}/10 sample stocks")
        else:
            self.log_pass(f"Normality check: {non_normal}/10 stocks show non-normal residuals")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Normality: {len(issues_found)} issues found"
        }
    
    def document_capm_assumptions(self) -> Dict:
        """Document which CAPM assumptions are violated."""
        logger.info("="*70)
        logger.info("Documenting CAPM assumption violations")
        logger.info("="*70)
        
        # This is a documentation check
        assumptions = {
            'homogeneous_expectations': 'VIOLATED - Investors have different expectations',
            'risk_free_borrowing_lending': 'VIOLATED - Borrowing rates > lending rates',
            'no_taxes': 'VIOLATED - Taxes exist',
            'no_transaction_costs': 'VIOLATED - Transaction costs exist',
            'market_efficiency': 'PARTIALLY VIOLATED - Markets may not be fully efficient',
            'perfect_markets': 'VIOLATED - Markets have frictions'
        }
        
        self.log_pass("CAPM assumptions documented (all standard assumptions are violated in practice)")
        
        return {
            'passed': True,
            'assumptions': assumptions,
            'summary': "CAPM assumptions documented"
        }
    
    def run_all_checks(self) -> Dict:
        """Run all assumption checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 4.3: ASSUMPTION VIOLATIONS")
        logger.info("="*70 + "\n")
        
        results = {
            'linearity': self.check_linearity(),
            'homoscedasticity': self.check_homoscedasticity(),
            'autocorrelation': self.check_autocorrelation(),
            'normality': self.check_normality(),
            'capm_assumptions': self.document_capm_assumptions()
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

