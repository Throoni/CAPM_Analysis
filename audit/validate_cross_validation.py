"""
validate_cross_validation.py

Phase 10.2: Cross-Validation Framework Audit

Validates cross-validation:
- K-fold validation checks
- Time-series CV validation
- Stability testing
- Overfitting detection
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class CrossValidationAudit:
    """Audit class for cross-validation validation."""
    
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
    
    def check_cv_implementation(self) -> Dict:
        """Check if cross-validation is implemented."""
        logger.info("Checking for cross-validation implementation...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for CV-related code
        cv_keywords = ['cross_val', 'kfold', 'k-fold', 'walk.forward', 'walk_forward', 'train_test_split']
        cv_files = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                found_keywords = [kw for kw in cv_keywords if kw in content.lower()]
                if found_keywords:
                    cv_files.append({
                        'file': str(py_file.relative_to(self.project_root)),
                        'keywords': found_keywords
                    })
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(cv_files) == 0:
            self.log_issue('warning', "No cross-validation implementation found")
            issues_found.append({
                'issue': 'No cross-validation implementation (recommended for model validation)'
            })
        else:
            self.log_pass(f"Found CV implementation in {len(cv_files)} file(s)")
            for cv_file in cv_files[:3]:
                logger.info(f"  - {cv_file['file']}: {', '.join(cv_file['keywords'])}")
        
        return {
            'passed': len(cv_files) > 0,
            'cv_files': cv_files,
            'issues': issues_found
        }
    
    def check_train_test_split(self) -> Dict:
        """Check if train/test split is used."""
        logger.info("Checking for train/test split...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for train/test split patterns
        split_patterns = ['train_test_split', 'train_size', 'test_size', 'holdout']
        found_splits = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    for pattern in split_patterns:
                        if pattern in line.lower() and not line.strip().startswith('#'):
                            found_splits.append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': i,
                                'pattern': pattern
                            })
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_splits) == 0:
            self.log_issue('warning', "No train/test split found")
            issues_found.append({
                'issue': 'No train/test split (recommended for out-of-sample validation)'
            })
        else:
            self.log_pass(f"Found train/test split in {len(found_splits)} location(s)")
        
        return {
            'passed': len(found_splits) > 0,
            'splits_found': found_splits,
            'issues': issues_found
        }
    
    def check_overfitting_detection(self) -> Dict:
        """Check for overfitting detection mechanisms."""
        logger.info("Checking for overfitting detection...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for overfitting-related patterns
        overfitting_patterns = ['overfit', 'over.fit', 'regularization', 'penalty', 'validation', 'holdout']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in overfitting_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break  # Only count each file once
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No overfitting detection mechanisms found")
            issues_found.append({
                'issue': 'No overfitting detection (recommended for model validation)'
            })
        else:
            self.log_pass(f"Found overfitting-related code in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_model_stability(self) -> Dict:
        """Check for model stability testing."""
        logger.info("Checking for model stability testing...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for stability-related patterns
        stability_patterns = ['stability', 'robust', 'sensitivity', 'bootstrap', 'resample']
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
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No model stability testing found")
            issues_found.append({
                'issue': 'No model stability testing (recommended)'
            })
        else:
            self.log_pass(f"Found stability-related code in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all cross-validation checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 10.2: CROSS-VALIDATION FRAMEWORK")
        logger.info("="*70 + "\n")
        
        results = {
            'cv_implementation': self.check_cv_implementation(),
            'train_test_split': self.check_train_test_split(),
            'overfitting_detection': self.check_overfitting_detection(),
            'model_stability': self.check_model_stability()
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
    
    audit = CrossValidationAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("CROSS-VALIDATION AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

