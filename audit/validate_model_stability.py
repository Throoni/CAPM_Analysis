"""
validate_model_stability.py

Phase 10.4: Model Stability Analysis Audit

Validates model stability:
- Time stability tests
- Parameter sensitivity
- Robustness checks
- Structural break detection
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelStabilityAudit:
    """Audit class for model stability validation."""
    
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
    
    def check_time_stability(self) -> Dict:
        """Check for time stability testing."""
        logger.info("Checking for time stability testing...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for time stability patterns
        stability_patterns = ['time.stability', 'rolling', 'window', 'subperiod', 'stability']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in stability_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        # Check robustness_checks.py specifically
        robustness_file = analysis_dir / "robustness_checks.py"
        if robustness_file.exists():
            with open(robustness_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                if 'subperiod' in content:
                    found_patterns.append({
                        'file': 'analysis/robustness_checks.py',
                        'pattern': 'subperiod'
                    })
                    self.log_pass("Found subperiod analysis in robustness_checks.py")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No time stability testing found")
            issues_found.append({
                'issue': 'No time stability testing (recommended)'
            })
        else:
            self.log_pass(f"Found time stability testing in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_parameter_sensitivity(self) -> Dict:
        """Check for parameter sensitivity analysis."""
        logger.info("Checking for parameter sensitivity analysis...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for sensitivity patterns
        sensitivity_patterns = ['sensitivity', 'parameter.variation', 'scenario', 'what.if']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in sensitivity_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No parameter sensitivity analysis found")
            issues_found.append({
                'issue': 'No parameter sensitivity analysis (recommended)'
            })
        else:
            self.log_pass(f"Found sensitivity analysis in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_robustness_checks(self) -> Dict:
        """Check for robustness checks implementation."""
        logger.info("Checking for robustness checks...")
        
        issues_found = []
        robustness_file = self.project_root / "analysis" / "robustness_checks.py"
        
        if not robustness_file.exists():
            self.log_issue('warning', "robustness_checks.py not found")
            issues_found.append({
                'issue': 'Missing robustness_checks.py'
            })
            return {
                'passed': False,
                'issues': issues_found
            }
        
        self.log_pass("robustness_checks.py exists")
        
        # Check for key robustness functions
        with open(robustness_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        robustness_functions = [
            'subperiod',
            'country',
            'portfolio',
            'clean_sample'
        ]
        
        found_functions = []
        for func in robustness_functions:
            if func in content.lower():
                found_functions.append(func)
        
        if len(found_functions) >= 2:
            self.log_pass(f"Found {len(found_functions)} robustness check types")
        else:
            self.log_issue('warning', "Limited robustness checks found")
            issues_found.append({
                'issue': f'Only {len(found_functions)} robustness check types found'
            })
        
        return {
            'passed': len(found_functions) >= 2,
            'robustness_functions': found_functions,
            'issues': issues_found
        }
    
    def check_structural_breaks(self) -> Dict:
        """Check for structural break detection."""
        logger.info("Checking for structural break detection...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for structural break patterns
        break_patterns = ['structural.break', 'breakpoint', 'chow.test', 'regime', 'crisis']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in break_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No structural break detection found")
            issues_found.append({
                'issue': 'No structural break detection (optional, but recommended)'
            })
        else:
            self.log_pass(f"Found structural break detection in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all model stability checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 10.4: MODEL STABILITY ANALYSIS")
        logger.info("="*70 + "\n")
        
        results = {
            'time_stability': self.check_time_stability(),
            'parameter_sensitivity': self.check_parameter_sensitivity(),
            'robustness_checks': self.check_robustness_checks(),
            'structural_breaks': self.check_structural_breaks()
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
    
    audit = ModelStabilityAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("MODEL STABILITY AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

