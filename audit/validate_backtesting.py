"""
validate_backtesting.py

Phase 10.7: Backtesting Framework Audit

Validates backtesting:
- Historical simulation
- Strategy evaluation
- Performance metrics
- Drawdown analysis
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class BacktestingAudit:
    """Audit class for backtesting validation."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed = []
        self.project_root = Path(__file__).parent.parent
        
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
    
    def check_historical_simulation(self) -> Dict:
        """Check for historical simulation."""
        logger.info("Checking for historical simulation...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for historical simulation patterns
        hist_patterns = ['historical', 'backtest', 'back.test', 'simulate', 'walk.forward']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in hist_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No historical simulation found")
            issues_found.append({
                'issue': 'No historical simulation (recommended for strategy validation)'
            })
        else:
            self.log_pass(f"Found historical simulation in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_strategy_evaluation(self) -> Dict:
        """Check for strategy evaluation."""
        logger.info("Checking for strategy evaluation...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for strategy evaluation patterns
        strategy_patterns = ['strategy', 'portfolio', 'return', 'sharpe', 'performance']
        found_patterns = []
        
        # Check portfolio optimization and recommendation files
        portfolio_files = [
            'analysis/portfolio_optimization.py',
            'analysis/portfolio_recommendation.py'
        ]
        
        for pf_file in portfolio_files:
            full_path = self.project_root / pf_file
            if full_path.exists():
                found_patterns.append({
                    'file': pf_file,
                    'pattern': 'portfolio_strategy'
                })
                self.log_pass(f"Found strategy evaluation: {pf_file}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No strategy evaluation found")
            issues_found.append({
                'issue': 'No strategy evaluation (recommended)'
            })
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_performance_metrics(self) -> Dict:
        """Check for performance metrics calculation."""
        logger.info("Checking for performance metrics...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for performance metric patterns
        metric_patterns = ['sharpe', 'return', 'volatility', 'std', 'mean', 'alpha', 'beta']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                # Count how many metrics are mentioned
                metrics_found = sum(1 for pattern in metric_patterns if pattern in content)
                if metrics_found >= 3:  # At least 3 different metrics
                    found_patterns.append({
                        'file': str(py_file.relative_to(self.project_root)),
                        'metrics_count': metrics_found
                    })
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "Limited performance metrics found")
            issues_found.append({
                'issue': 'Limited performance metrics (recommended: Sharpe, return, volatility)'
            })
        else:
            self.log_pass(f"Found performance metrics in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_drawdown_analysis(self) -> Dict:
        """Check for drawdown analysis."""
        logger.info("Checking for drawdown analysis...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for drawdown patterns
        drawdown_patterns = ['drawdown', 'max.drawdown', 'peak.to.trough']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in drawdown_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No drawdown analysis found")
            issues_found.append({
                'issue': 'No drawdown analysis (optional, but recommended for risk assessment)'
            })
        else:
            self.log_pass(f"Found drawdown analysis in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all backtesting checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 10.7: BACKTESTING FRAMEWORK")
        logger.info("="*70 + "\n")
        
        results = {
            'historical_simulation': self.check_historical_simulation(),
            'strategy_evaluation': self.check_strategy_evaluation(),
            'performance_metrics': self.check_performance_metrics(),
            'drawdown_analysis': self.check_drawdown_analysis()
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    audit = BacktestingAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("BACKTESTING AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

