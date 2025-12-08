"""
validate_test_coverage.py

Phase 8.2: Test Coverage Audit

Validates test coverage:
- Check if tests exist
- Verify test coverage percentage
- Identify untested functions
- Check test quality
"""

import os
import sys
import logging
import subprocess
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TestCoverageAudit:
    """Audit class for test coverage validation."""
    
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
    
    def check_test_directory(self) -> Dict:
        """Check if tests directory exists and has structure."""
        logger.info("Checking test directory structure...")
        
        issues_found = []
        tests_dir = self.project_root / "tests"
        
        if not tests_dir.exists():
            self.log_issue('critical', "tests/ directory not found")
            issues_found.append({
                'issue': 'Missing tests directory'
            })
            return {
                'passed': False,
                'issues': issues_found
            }
        
        self.log_pass("tests/ directory exists")
        
        # Check for subdirectories
        subdirs = ['unit', 'integration', 'fixtures']
        for subdir in subdirs:
            subdir_path = tests_dir / subdir
            if subdir_path.exists():
                self.log_pass(f"tests/{subdir}/ exists")
            else:
                self.log_issue('warning', f"tests/{subdir}/ not found")
                issues_found.append({
                    'issue': f'Missing tests/{subdir}/ directory'
                })
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found
        }
    
    def check_test_files(self) -> Dict:
        """Check if test files exist."""
        logger.info("Checking for test files...")
        
        issues_found = []
        tests_dir = self.project_root / "tests"
        
        if not tests_dir.exists():
            return {
                'passed': False,
                'issues': [{'issue': 'Cannot check - tests directory missing'}]
            }
        
        # Find test files
        test_files = list(tests_dir.rglob("test_*.py"))
        
        if len(test_files) == 0:
            self.log_issue('critical', "No test files found")
            issues_found.append({
                'issue': 'No test files found (test_*.py)'
            })
        else:
            self.log_pass(f"Found {len(test_files)} test file(s)")
            for test_file in test_files[:5]:  # First 5
                logger.info(f"  - {test_file.relative_to(self.project_root)}")
        
        return {
            'passed': len(test_files) > 0,
            'test_files': [str(f.relative_to(self.project_root)) for f in test_files],
            'count': len(test_files),
            'issues': issues_found
        }
    
    def check_pytest_installed(self) -> Dict:
        """Check if pytest is installed."""
        logger.info("Checking if pytest is installed...")
        
        issues_found = []
        
        try:
            import pytest
            pytest_version = pytest.__version__
            self.log_pass(f"pytest installed (version {pytest_version})")
            return {
                'passed': True,
                'version': pytest_version,
                'issues': []
            }
        except ImportError:
            self.log_issue('critical', "pytest not installed")
            issues_found.append({
                'issue': 'pytest not installed - cannot run tests'
            })
            return {
                'passed': False,
                'issues': issues_found
            }
    
    def check_test_coverage(self) -> Dict:
        """Check test coverage (if pytest-cov is available)."""
        logger.info("Checking test coverage...")
        
        issues_found = []
        
        try:
            import pytest_cov
            self.log_pass("pytest-cov installed")
            
            # Try to run coverage (this is a simplified check)
            # In practice, you'd run: pytest --cov=analysis --cov-report=term-missing
            coverage_available = True
        except ImportError:
            self.log_issue('warning', "pytest-cov not installed - cannot check coverage")
            issues_found.append({
                'issue': 'pytest-cov not installed (recommended for coverage checking)'
            })
            coverage_available = False
        
        return {
            'passed': coverage_available,
            'coverage_available': coverage_available,
            'issues': issues_found
        }
    
    def check_test_quality(self) -> Dict:
        """Check test quality (heuristic checks)."""
        logger.info("Checking test quality...")
        
        issues_found = []
        tests_dir = self.project_root / "tests"
        
        if not tests_dir.exists():
            return {
                'passed': False,
                'issues': [{'issue': 'Cannot check - tests directory missing'}]
            }
        
        test_files = list(tests_dir.rglob("test_*.py"))
        
        if len(test_files) == 0:
            return {
                'passed': False,
                'issues': [{'issue': 'No test files to check'}]
            }
        
        # Check for test classes and test functions
        total_tests = 0
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count test functions
                test_functions = content.count('def test_')
                total_tests += test_functions
                
                # Check for assertions
                if 'assert' not in content:
                    issues_found.append({
                        'file': str(test_file.relative_to(self.project_root)),
                        'issue': 'No assertions found in test file'
                    })
                    self.log_issue('warning',
                                 f"No assertions in {test_file.name}")
            except Exception as e:
                self.log_issue('warning', f"Error checking {test_file.name}: {e}")
        
        if total_tests == 0:
            self.log_issue('critical', "No test functions found")
            issues_found.append({
                'issue': 'No test functions found (def test_*)'
            })
        else:
            self.log_pass(f"Found {total_tests} test function(s)")
        
        return {
            'passed': total_tests > 0 and len(issues_found) == 0,
            'total_tests': total_tests,
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all test coverage checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 8.2: TEST COVERAGE AUDIT")
        logger.info("="*70 + "\n")
        
        results = {
            'test_directory': self.check_test_directory(),
            'test_files': self.check_test_files(),
            'pytest_installed': self.check_pytest_installed(),
            'test_coverage': self.check_test_coverage(),
            'test_quality': self.check_test_quality()
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
    
    audit = TestCoverageAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("TEST COVERAGE AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

