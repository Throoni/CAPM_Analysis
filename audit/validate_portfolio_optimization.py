"""
validate_portfolio_optimization.py

Validates portfolio optimization analysis (efficient frontier, minimum-variance portfolio, tangency portfolio).
Checks calculation correctness, constraint satisfaction, and result reasonableness.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict

from analysis.config import (
    RESULTS_REPORTS_DIR,
    RESULTS_PLOTS_DIR
)

logger = logging.getLogger(__name__)


class PortfolioOptimizationAudit:
    """Audit class for portfolio optimization analysis."""
    
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
        """Check if portfolio optimization result files exist."""
        logger.info("="*70)
        logger.info("Checking portfolio optimization files")
        logger.info("="*70)
        
        required_files = [
            os.path.join(RESULTS_REPORTS_DIR, "portfolio_optimization_results.csv"),
            os.path.join(RESULTS_REPORTS_DIR, "diversification_benefits.csv"),
            os.path.join(RESULTS_PLOTS_DIR, "efficient_frontier.png")
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
    
    def validate_portfolio_results(self) -> Dict:
        """Validate portfolio optimization results."""
        logger.info("="*70)
        logger.info("Validating portfolio optimization results")
        logger.info("="*70)
        
        results_file = os.path.join(RESULTS_REPORTS_DIR, "portfolio_optimization_results.csv")
        
        if not os.path.exists(results_file):
            return {'passed': False, 'issues': self.issues}
        
        df = pd.read_csv(results_file)
        issues_found = []
        
        # Check required portfolios
        required_portfolios = ['Minimum-Variance', 'Optimal Risky (Tangency)', 'Equal-Weighted']
        missing = [p for p in required_portfolios if p not in df['portfolio'].values]
        if missing:
            issues_found.append({
                'issue': f'Missing portfolios: {missing}',
                'severity': 'critical'
            })
            self.log_issue('critical', f"Missing portfolios: {missing}")
        else:
            self.log_pass("All required portfolios present")
        
        # Check for required columns
        required_cols = ['portfolio', 'expected_return', 'volatility', 'sharpe_ratio']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues_found.append({
                'issue': f'Missing columns: {missing_cols}',
                'severity': 'critical'
            })
            self.log_issue('critical', f"Missing columns: {missing_cols}")
        else:
            self.log_pass("All required columns present")
        
        # Validate minimum-variance portfolio
        min_var = df[df['portfolio'] == 'Minimum-Variance']
        if len(min_var) > 0:
            mv_return = min_var['expected_return'].iloc[0]
            mv_vol = min_var['volatility'].iloc[0]
            mv_sharpe = min_var['sharpe_ratio'].iloc[0]
            
            # Check volatility is positive
            if mv_vol <= 0:
                issues_found.append({
                    'issue': 'Minimum-variance portfolio has non-positive volatility',
                    'severity': 'critical'
                })
                self.log_issue('critical', "Minimum-variance portfolio volatility <= 0")
            else:
                self.log_pass(f"Minimum-variance portfolio: Return={mv_return:.4f}%, Vol={mv_vol:.4f}%, Sharpe={mv_sharpe:.4f}")
            
            # Check Sharpe ratio calculation
            if not pd.isna(mv_return) and not pd.isna(mv_vol) and mv_vol > 0:
                expected_sharpe = mv_return / mv_vol
                if abs(expected_sharpe - mv_sharpe) > 0.01:
                    issues_found.append({
                        'issue': f'Minimum-variance Sharpe ratio mismatch: expected {expected_sharpe:.4f}, got {mv_sharpe:.4f}',
                        'severity': 'warning'
                    })
                    self.log_issue('warning', "Sharpe ratio calculation may be incorrect")
        
        # Validate tangency portfolio
        tangency = df[df['portfolio'] == 'Optimal Risky (Tangency)']
        if len(tangency) > 0:
            tan_return = tangency['expected_return'].iloc[0]
            tan_vol = tangency['volatility'].iloc[0]
            tan_sharpe = tangency['sharpe_ratio'].iloc[0]
            
            # Tangency should have highest Sharpe ratio
            all_sharpes = df['sharpe_ratio'].dropna()
            if len(all_sharpes) > 0:
                max_sharpe = all_sharpes.max()
                if abs(tan_sharpe - max_sharpe) > 0.01:
                    issues_found.append({
                        'issue': f'Tangency portfolio does not have highest Sharpe ratio: {tan_sharpe:.4f} vs max {max_sharpe:.4f}',
                        'severity': 'warning'
                    })
                    self.log_issue('warning', "Tangency portfolio may not be optimal")
                else:
                    self.log_pass(f"Tangency portfolio has highest Sharpe ratio: {tan_sharpe:.4f}")
            
            # Tangency should have higher return than minimum-variance
            if len(min_var) > 0:
                mv_return = min_var['expected_return'].iloc[0]
                if tan_return < mv_return:
                    issues_found.append({
                        'issue': 'Tangency portfolio has lower return than minimum-variance (unusual)',
                        'severity': 'warning'
                    })
                    self.log_issue('warning', "Tangency return < Minimum-variance return")
                else:
                    self.log_pass("Tangency return >= Minimum-variance return (as expected)")
        
        # Check reasonableness of values
        returns = df['expected_return'].dropna()
        volatilities = df['volatility'].dropna()
        
        if len(returns) > 0:
            # Monthly returns should be reasonable (typically -5% to +5%)
            extreme_returns = ((returns < -10) | (returns > 10)).sum()
            if extreme_returns > 0:
                issues_found.append({
                    'issue': f'{extreme_returns} portfolios with extreme returns (>10% or <-10% monthly)',
                    'severity': 'warning'
                })
                self.log_issue('warning', f"{extreme_returns} portfolios with extreme returns")
            else:
                self.log_pass("All portfolio returns in reasonable range")
        
        if len(volatilities) > 0:
            # Monthly volatilities should be positive and reasonable
            negative_vol = (volatilities <= 0).sum()
            if negative_vol > 0:
                issues_found.append({
                    'issue': f'{negative_vol} portfolios with non-positive volatility',
                    'severity': 'critical'
                })
                self.log_issue('critical', f"{negative_vol} portfolios with non-positive volatility")
            else:
                self.log_pass("All portfolio volatilities are positive")
        
        self.issues.extend(issues_found)
        
        return {
            'passed': len([i for i in issues_found if i.get('severity') == 'critical']) == 0,
            'issues': issues_found,
            'summary': f"Portfolio optimization validation: {len([i for i in issues_found if i.get('severity') == 'critical'])} critical, {len([i for i in issues_found if i.get('severity') == 'warning'])} warnings"
        }
    
    def validate_diversification_benefits(self) -> Dict:
        """Validate diversification benefits calculations."""
        logger.info("="*70)
        logger.info("Validating diversification benefits")
        logger.info("="*70)
        
        div_file = os.path.join(RESULTS_REPORTS_DIR, "diversification_benefits.csv")
        
        if not os.path.exists(div_file):
            self.log_issue('warning', "Diversification benefits file not found")
            return {'passed': True, 'issues': []}
        
        df = pd.read_csv(div_file)
        issues_found = []
        
        # Check required columns
        required_cols = ['avg_stock_volatility', 'portfolio_volatility', 'diversification_ratio', 'variance_reduction_pct']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues_found.append({
                'issue': f'Missing columns: {missing_cols}',
                'severity': 'critical'
            })
            self.log_issue('critical', f"Missing columns: {missing_cols}")
        else:
            self.log_pass("All required columns present")
        
        # Validate diversification ratio
        if 'diversification_ratio' in df.columns and 'avg_stock_volatility' in df.columns and 'portfolio_volatility' in df.columns:
            div_ratio = df['diversification_ratio'].iloc[0]
            avg_vol = df['avg_stock_volatility'].iloc[0]
            port_vol = df['portfolio_volatility'].iloc[0]
            
            if not pd.isna(div_ratio) and not pd.isna(avg_vol) and not pd.isna(port_vol) and port_vol > 0:
                expected_ratio = avg_vol / port_vol
                if abs(expected_ratio - div_ratio) > 0.01:
                    issues_found.append({
                        'issue': f'Diversification ratio mismatch: expected {expected_ratio:.4f}, got {div_ratio:.4f}',
                        'severity': 'warning'
                    })
                    self.log_issue('warning', "Diversification ratio calculation may be incorrect")
                else:
                    self.log_pass(f"Diversification ratio correct: {div_ratio:.4f}")
            
            # Diversification ratio should be > 1 (portfolio less volatile than average stock)
            if div_ratio < 1.0:
                issues_found.append({
                    'issue': f'Diversification ratio < 1.0 ({div_ratio:.4f}), portfolio is more volatile than average stock (unusual)',
                    'severity': 'warning'
                })
                self.log_issue('warning', "Diversification ratio < 1.0 (unusual)")
            elif div_ratio > 1.0:
                self.log_pass(f"Diversification ratio > 1.0 ({div_ratio:.4f}), indicating diversification benefits")
        
        # Validate variance reduction
        if 'variance_reduction_pct' in df.columns:
            var_reduction = df['variance_reduction_pct'].iloc[0]
            
            # Variance reduction should be positive (portfolio has less variance)
            if var_reduction < 0:
                issues_found.append({
                    'issue': f'Negative variance reduction ({var_reduction:.2f}%), portfolio has more variance than average stock',
                    'severity': 'warning'
                })
                self.log_issue('warning', "Negative variance reduction (unusual)")
            elif var_reduction > 0:
                self.log_pass(f"Variance reduction: {var_reduction:.2f}% (positive, as expected)")
        
        self.issues.extend(issues_found)
        
        return {
            'passed': len([i for i in issues_found if i.get('severity') == 'critical']) == 0,
            'issues': issues_found,
            'summary': f"Diversification validation: {len([i for i in issues_found if i.get('severity') == 'critical'])} critical, {len([i for i in issues_found if i.get('severity') == 'warning'])} warnings"
        }
    
    def validate_efficient_frontier_plot(self) -> Dict:
        """Validate efficient frontier plot exists and is reasonable."""
        logger.info("="*70)
        logger.info("Validating efficient frontier plot")
        logger.info("="*70)
        
        plot_file = os.path.join(RESULTS_PLOTS_DIR, "efficient_frontier.png")
        
        if not os.path.exists(plot_file):
            self.log_issue('critical', "Efficient frontier plot not found")
            return {'passed': False, 'issues': [{'severity': 'critical', 'message': 'Plot file not found'}]}
        
        file_size = os.path.getsize(plot_file)
        
        # Check file size (should be reasonable, not empty or corrupted)
        if file_size < 1000:  # Less than 1KB is suspicious
            self.log_issue('warning', f"Efficient frontier plot file is very small ({file_size} bytes), may be corrupted")
        else:
            self.log_pass(f"Efficient frontier plot exists ({file_size:,} bytes)")
        
        return {
            'passed': True,
            'issues': [],
            'summary': 'Efficient frontier plot validation complete'
        }
    
    def run_all_checks(self) -> Dict:
        """Run all portfolio optimization validation checks."""
        logger.info("="*70)
        logger.info("PORTFOLIO OPTIMIZATION VALIDATION")
        logger.info("="*70)
        
        # Check 1: Files exist
        files_exist = self.check_files_exist()
        
        if not files_exist:
            return {
                'passed': False,
                'issues': self.issues,
                'summary': 'Portfolio optimization validation failed: required files not found'
            }
        
        # Check 2: Validate portfolio results
        portfolio_results = self.validate_portfolio_results()
        
        # Check 3: Validate diversification benefits
        div_results = self.validate_diversification_benefits()
        
        # Check 4: Validate plot
        plot_results = self.validate_efficient_frontier_plot()
        
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
    
    audit = PortfolioOptimizationAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION VALIDATION COMPLETE")
    print("="*70)
    print(f"Passed: {results['passed']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")

