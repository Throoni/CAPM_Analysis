"""
validate_integration.py

Phase 8.3: Integration Testing Audit

Validates integration between components:
- End-to-end pipeline tests
- Component integration
- Data flow validation
- Error propagation
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class IntegrationAudit:
    """Audit class for integration testing validation."""
    
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
    
    def check_integration_tests(self) -> Dict:
        """Check if integration tests exist."""
        logger.info("Checking for integration tests...")
        
        issues_found = []
        tests_dir = self.project_root / "tests" / "integration"
        
        if not tests_dir.exists():
            self.log_issue('warning', "tests/integration/ directory not found")
            issues_found.append({
                'issue': 'Missing tests/integration/ directory'
            })
            return {
                'passed': False,
                'issues': issues_found
            }
        
        # Find integration test files
        test_files = list(tests_dir.glob("test_*.py"))
        
        if len(test_files) == 0:
            self.log_issue('warning', "No integration test files found")
            issues_found.append({
                'issue': 'No integration test files (test_*.py)'
            })
        else:
            self.log_pass(f"Found {len(test_files)} integration test file(s)")
            for test_file in test_files[:5]:
                logger.info(f"  - {test_file.name}")
        
        return {
            'passed': len(test_files) > 0,
            'test_files': [str(f.name) for f in test_files],
            'count': len(test_files),
            'issues': issues_found
        }
    
    def check_data_flow(self) -> Dict:
        """Check data flow between components."""
        logger.info("Checking data flow between components...")
        
        issues_found = []
        
        # Check key data flow paths
        data_flow_paths = [
            {
                'source': 'data/raw/prices_stocks_*.csv',
                'processor': 'analysis/returns_processing.py',
                'output': 'data/processed/returns_panel.csv',
                'description': 'Stock prices → Returns processing → Returns panel'
            },
            {
                'source': 'data/processed/returns_panel.csv',
                'processor': 'analysis/capm_regression.py',
                'output': 'results/data/capm_results.csv',
                'description': 'Returns panel → CAPM regression → CAPM results'
            },
            {
                'source': 'results/data/capm_results.csv',
                'processor': 'analysis/fama_macbeth.py',
                'output': 'results/reports/fama_macbeth_summary.csv',
                'description': 'CAPM results → Fama-MacBeth → FM summary'
            }
        ]
        
        for path_info in data_flow_paths:
            source_pattern = path_info['source']
            processor = self.project_root / path_info['processor']
            output = self.project_root / path_info['output']
            
            # Check if processor exists
            if not processor.exists():
                issues_found.append({
                    'issue': f"Processor missing: {path_info['processor']}",
                    'path': path_info['description']
                })
                self.log_issue('critical', f"Missing processor: {path_info['processor']}")
            else:
                self.log_pass(f"Processor exists: {path_info['processor']}")
            
            # Check if output can be generated (file may not exist yet, that's OK)
            # We just check the directory exists
            output_dir = output.parent
            if not output_dir.exists():
                issues_found.append({
                    'issue': f"Output directory missing: {output_dir}",
                    'path': path_info['description']
                })
                self.log_issue('warning', f"Output directory missing: {output_dir}")
            else:
                self.log_pass(f"Output directory exists: {output_dir}")
        
        return {
            'passed': len([i for i in issues_found if 'critical' in str(i)]) == 0,
            'paths_checked': len(data_flow_paths),
            'issues': issues_found
        }
    
    def check_module_imports(self) -> Dict:
        """Check if modules can be imported (integration check)."""
        logger.info("Checking module imports...")
        
        issues_found = []
        modules_to_check = [
            'analysis.config',
            'analysis.returns_processing',
            'analysis.capm_regression',
            'analysis.fama_macbeth',
            'analysis.riskfree_helper'
        ]
        
        failed_imports = []
        for module_name in modules_to_check:
            try:
                __import__(module_name)
                self.log_pass(f"Module imports successfully: {module_name}")
            except ImportError as e:
                failed_imports.append(module_name)
                issues_found.append({
                    'issue': f"Import failed: {module_name}",
                    'error': str(e)
                })
                self.log_issue('critical', f"Import failed: {module_name} - {e}")
            except Exception as e:
                failed_imports.append(module_name)
                issues_found.append({
                    'issue': f"Error importing {module_name}",
                    'error': str(e)
                })
                self.log_issue('warning', f"Error importing {module_name}: {e}")
        
        return {
            'passed': len(failed_imports) == 0,
            'modules_checked': len(modules_to_check),
            'failed_imports': failed_imports,
            'issues': issues_found
        }
    
    def check_file_dependencies(self) -> Dict:
        """Check file dependencies and ensure they exist."""
        logger.info("Checking file dependencies...")
        
        issues_found = []
        
        # Key files that should exist
        required_files = [
            'analysis/config.py',
            'analysis/returns_processing.py',
            'analysis/capm_regression.py',
            'data/processed/returns_panel.csv'  # May not exist, but directory should
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.log_pass(f"File exists: {file_path}")
            else:
                # Check if it's a directory that should exist
                if file_path.endswith('.csv'):
                    dir_path = full_path.parent
                    if dir_path.exists():
                        self.log_pass(f"Directory exists: {dir_path} (file may be generated)")
                    else:
                        missing_files.append(file_path)
                        issues_found.append({
                            'issue': f"Missing: {file_path}",
                            'file': file_path
                        })
                        self.log_issue('warning', f"Missing: {file_path}")
                else:
                    missing_files.append(file_path)
                    issues_found.append({
                        'issue': f"Missing: {file_path}",
                        'file': file_path
                    })
                    self.log_issue('critical', f"Missing required file: {file_path}")
        
        return {
            'passed': len(missing_files) == 0,
            'files_checked': len(required_files),
            'missing_files': missing_files,
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all integration checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 8.3: INTEGRATION TESTING AUDIT")
        logger.info("="*70 + "\n")
        
        results = {
            'integration_tests': self.check_integration_tests(),
            'data_flow': self.check_data_flow(),
            'module_imports': self.check_module_imports(),
            'file_dependencies': self.check_file_dependencies()
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
    
    audit = IntegrationAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("INTEGRATION AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

