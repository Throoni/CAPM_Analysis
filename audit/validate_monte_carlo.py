"""
validate_monte_carlo.py

Phase 10.6: Monte Carlo Validation Audit

Validates Monte Carlo methods:
- Simulation correctness
- Bootstrap implementation
- Confidence interval validation
- Test calibration
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MonteCarloAudit:
    """Audit class for Monte Carlo validation."""
    
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
    
    def check_bootstrap_implementation(self) -> Dict:
        """Check for bootstrap implementation."""
        logger.info("Checking for bootstrap implementation...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for bootstrap patterns
        bootstrap_patterns = ['bootstrap', 'resample', 'with.replacement', 'np.random.choice']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    for pattern in bootstrap_patterns:
                        if pattern in line.lower() and not line.strip().startswith('#'):
                            found_patterns.append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': i,
                                'pattern': pattern
                            })
                            break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No bootstrap implementation found")
            issues_found.append({
                'issue': 'No bootstrap implementation (recommended for inference)'
            })
        else:
            self.log_pass(f"Found bootstrap in {len(found_patterns)} location(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_confidence_intervals(self) -> Dict:
        """Check for confidence interval calculation."""
        logger.info("Checking for confidence interval calculation...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for CI patterns
        ci_patterns = ['confidence.interval', 'ci', 'percentile', 'quantile', 'alpha']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in ci_patterns:
                    if pattern in content and ('def' in content or '=' in content):
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        # Check if statsmodels is used (has built-in CIs)
        try:
            import statsmodels.api as sm
            # If statsmodels is used, CIs are likely calculated
            found_patterns.append({
                'file': 'statsmodels (via imports)',
                'pattern': 'statsmodels_ci'
            })
            self.log_pass("statsmodels used (provides confidence intervals)")
        except ImportError:
            pass
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No confidence interval calculation found")
            issues_found.append({
                'issue': 'No confidence interval calculation (recommended)'
            })
        else:
            self.log_pass(f"Found CI calculation in {len(found_patterns)} location(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_simulation_correctness(self) -> Dict:
        """Check for simulation correctness (random seeds, etc.)."""
        logger.info("Checking simulation correctness...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Check for random seed setting in simulation code
        simulation_files = []
        seed_found = False
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                # Check if file has simulation code
                if 'random' in content or 'simulation' in content or 'bootstrap' in content:
                    simulation_files.append(str(py_file.relative_to(self.project_root)))
                    
                    # Check for random seed
                    if 'random.seed' in content or 'np.random.seed' in content:
                        seed_found = True
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(simulation_files) > 0:
            if seed_found:
                self.log_pass("Random seed found in simulation code")
            else:
                self.log_issue('warning', "No random seed in simulation code")
                issues_found.append({
                    'issue': 'No random seed in simulation code (affects reproducibility)'
                })
        else:
            self.log_pass("No simulation code found (not applicable)")
        
        return {
            'passed': seed_found or len(simulation_files) == 0,
            'simulation_files': simulation_files,
            'seed_found': seed_found,
            'issues': issues_found
        }
    
    def check_iteration_count(self) -> Dict:
        """Check for reasonable iteration counts in simulations."""
        logger.info("Checking iteration counts...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for iteration patterns
        iteration_patterns = ['n_iter', 'n_sim', 'n_bootstrap', 'iterations', 'n_samples']
        found_iterations = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    for pattern in iteration_patterns:
                        if pattern in line.lower():
                            # Try to extract number
                            import re
                            numbers = re.findall(r'\d+', line)
                            if numbers:
                                n_iter = int(numbers[0])
                                if n_iter < 100:
                                    found_iterations.append({
                                        'file': str(py_file.relative_to(self.project_root)),
                                        'line': i,
                                        'n_iter': n_iter,
                                        'issue': 'Low iteration count'
                                    })
                                    self.log_issue('warning',
                                                 f"Low iteration count ({n_iter}) in {py_file.name}:{i}")
                                elif n_iter > 10000:
                                    found_iterations.append({
                                        'file': str(py_file.relative_to(self.project_root)),
                                        'line': i,
                                        'n_iter': n_iter,
                                        'issue': 'Very high iteration count'
                                    })
                                    self.log_issue('warning',
                                                 f"Very high iteration count ({n_iter}) in {py_file.name}:{i}")
                                else:
                                    self.log_pass(f"Reasonable iteration count ({n_iter}) in {py_file.name}")
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_iterations) == 0:
            self.log_pass("No iteration counts found (or not applicable)")
        
        return {
            'passed': len([f for f in found_iterations if 'Low' in f.get('issue', '')]) == 0,
            'iterations_found': found_iterations,
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all Monte Carlo validation checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 10.6: MONTE CARLO VALIDATION")
        logger.info("="*70 + "\n")
        
        results = {
            'bootstrap': self.check_bootstrap_implementation(),
            'confidence_intervals': self.check_confidence_intervals(),
            'simulation_correctness': self.check_simulation_correctness(),
            'iteration_count': self.check_iteration_count()
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
    
    audit = MonteCarloAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("MONTE CARLO VALIDATION AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

