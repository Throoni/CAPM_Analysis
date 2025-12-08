"""
validate_sensitivity.py

Phase 10.5: Sensitivity Analysis Audit

Validates sensitivity analysis:
- Parameter variation tests
- Scenario analysis
- Monte Carlo sampling
- Sensitivity visualization
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SensitivityAudit:
    """Audit class for sensitivity analysis validation."""
    
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
    
    def check_parameter_variation(self) -> Dict:
        """Check for parameter variation testing."""
        logger.info("Checking for parameter variation testing...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for parameter variation patterns
        variation_patterns = ['parameter.variation', 'vary', 'range', 'sweep', 'grid.search']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in variation_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No parameter variation testing found")
            issues_found.append({
                'issue': 'No parameter variation testing (recommended)'
            })
        else:
            self.log_pass(f"Found parameter variation in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_scenario_analysis(self) -> Dict:
        """Check for scenario analysis."""
        logger.info("Checking for scenario analysis...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for scenario patterns
        scenario_patterns = ['scenario', 'what.if', 'stress.test', 'stress_test']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in scenario_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No scenario analysis found")
            issues_found.append({
                'issue': 'No scenario analysis (recommended)'
            })
        else:
            self.log_pass(f"Found scenario analysis in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_monte_carlo(self) -> Dict:
        """Check for Monte Carlo sampling."""
        logger.info("Checking for Monte Carlo sampling...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for Monte Carlo patterns
        mc_patterns = ['monte.carlo', 'monte_carlo', 'simulation', 'bootstrap', 'resample']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in mc_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No Monte Carlo sampling found")
            issues_found.append({
                'issue': 'No Monte Carlo sampling (recommended for uncertainty quantification)'
            })
        else:
            self.log_pass(f"Found Monte Carlo in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_sensitivity_visualization(self) -> Dict:
        """Check for sensitivity visualization."""
        logger.info("Checking for sensitivity visualization...")
        
        issues_found = []
        results_dir = self.project_root / "results" / "plots"
        
        if not results_dir.exists():
            self.log_issue('warning', "results/plots/ directory not found")
            return {
                'passed': False,
                'issues': [{'issue': 'results/plots/ directory not found'}]
            }
        
        # Look for sensitivity-related plots
        sensitivity_plots = [
            'sensitivity',
            'tornado',
            'spider',
            'parameter'
        ]
        
        found_plots = []
        for plot_file in results_dir.glob("*.png"):
            plot_name_lower = plot_file.name.lower()
            for keyword in sensitivity_plots:
                if keyword in plot_name_lower:
                    found_plots.append(plot_file.name)
                    break
        
        if len(found_plots) == 0:
            self.log_issue('warning', "No sensitivity plots found")
            issues_found.append({
                'issue': 'No sensitivity visualization plots (optional)'
            })
        else:
            self.log_pass(f"Found {len(found_plots)} sensitivity plot(s)")
        
        return {
            'passed': len(found_plots) > 0,
            'plots_found': found_plots,
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all sensitivity analysis checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 10.5: SENSITIVITY ANALYSIS")
        logger.info("="*70 + "\n")
        
        results = {
            'parameter_variation': self.check_parameter_variation(),
            'scenario_analysis': self.check_scenario_analysis(),
            'monte_carlo': self.check_monte_carlo(),
            'visualization': self.check_sensitivity_visualization()
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
    
    audit = SensitivityAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

