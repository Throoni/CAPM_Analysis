"""
validate_out_of_sample.py

Phase 10.3: Out-of-Sample Validation Audit

Validates out-of-sample testing:
- Holdout set verification
- Prediction accuracy checks
- Forecast evaluation
- Model selection validation
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class OutOfSampleAudit:
    """Audit class for out-of-sample validation."""
    
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
    
    def check_holdout_set(self) -> Dict:
        """Check if holdout set is reserved."""
        logger.info("Checking for holdout set...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for holdout-related patterns
        holdout_patterns = ['holdout', 'test_set', 'validation_set', 'out.of.sample', 'oos']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in holdout_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No holdout set implementation found")
            issues_found.append({
                'issue': 'No holdout set (recommended for out-of-sample validation)'
            })
        else:
            self.log_pass(f"Found holdout-related code in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_prediction_accuracy(self) -> Dict:
        """Check for prediction accuracy evaluation."""
        logger.info("Checking for prediction accuracy evaluation...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for accuracy-related patterns
        accuracy_patterns = ['accuracy', 'mse', 'rmse', 'mae', 'prediction_error', 'forecast_error']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in accuracy_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No prediction accuracy evaluation found")
            issues_found.append({
                'issue': 'No prediction accuracy metrics (recommended)'
            })
        else:
            self.log_pass(f"Found accuracy evaluation in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_forecast_evaluation(self) -> Dict:
        """Check for forecast evaluation."""
        logger.info("Checking for forecast evaluation...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for forecast-related patterns
        forecast_patterns = ['forecast', 'predict', 'out.of.sample', 'oos', 'future_return']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in forecast_patterns:
                    if pattern in content and 'def' in content:  # Must be a function
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No forecast evaluation found")
            issues_found.append({
                'issue': 'No forecast evaluation (recommended for model validation)'
            })
        else:
            self.log_pass(f"Found forecast evaluation in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_model_selection(self) -> Dict:
        """Check for model selection procedures."""
        logger.info("Checking for model selection procedures...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for model selection patterns
        selection_patterns = ['model_selection', 'aic', 'bic', 'information_criterion', 'best_model']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in selection_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No model selection procedures found")
            issues_found.append({
                'issue': 'No model selection (recommended for comparing models)'
            })
        else:
            self.log_pass(f"Found model selection in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all out-of-sample validation checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 10.3: OUT-OF-SAMPLE VALIDATION")
        logger.info("="*70 + "\n")
        
        results = {
            'holdout_set': self.check_holdout_set(),
            'prediction_accuracy': self.check_prediction_accuracy(),
            'forecast_evaluation': self.check_forecast_evaluation(),
            'model_selection': self.check_model_selection()
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
    
    audit = OutOfSampleAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("OUT-OF-SAMPLE VALIDATION AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

