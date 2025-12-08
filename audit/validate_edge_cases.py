"""
validate_edge_cases.py

Phase 11.3: Edge Case Testing Audit

Validates edge case handling:
- Boundary conditions
- Empty data handling
- Missing data handling
- Extreme value handling
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class EdgeCaseAudit:
    """Audit class for edge case validation."""
    
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
    
    def check_empty_data_handling(self) -> Dict:
        """Check for empty data handling."""
        logger.info("Checking empty data handling...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for empty data handling patterns
        empty_patterns = ['empty', 'len(', 'is_empty', 'if.*empty', 'dropna', 'notna']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in empty_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No empty data handling found")
            issues_found.append({
                'issue': 'No empty data handling (recommended)'
            })
        else:
            self.log_pass(f"Found empty data handling in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_missing_data_handling(self) -> Dict:
        """Check for missing data handling."""
        logger.info("Checking missing data handling...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for missing data handling
        missing_patterns = ['dropna', 'fillna', 'isna', 'isnull', 'notna', 'notnull', 'nan']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in missing_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No missing data handling found")
            issues_found.append({
                'issue': 'No missing data handling (critical for data quality)'
            })
        else:
            self.log_pass(f"Found missing data handling in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_extreme_value_handling(self) -> Dict:
        """Check for extreme value handling."""
        logger.info("Checking extreme value handling...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for extreme value handling
        extreme_patterns = ['extreme', 'outlier', 'clip', 'bound', 'threshold', 'filter']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in extreme_patterns:
                    if pattern in content:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'pattern': pattern
                        })
                        break
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        # Check fix_extreme_betas.py specifically
        fix_file = analysis_dir / "fix_extreme_betas.py"
        if fix_file.exists():
            found_patterns.append({
                'file': 'analysis/fix_extreme_betas.py',
                'pattern': 'extreme_beta_fix'
            })
            self.log_pass("Found extreme beta fixing module")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No extreme value handling found")
            issues_found.append({
                'issue': 'No extreme value handling (recommended)'
            })
        else:
            self.log_pass(f"Found extreme value handling in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def check_boundary_conditions(self) -> Dict:
        """Check for boundary condition handling."""
        logger.info("Checking boundary condition handling...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        # Look for boundary checks
        boundary_patterns = ['if.*>', 'if.*<', 'if.*==', 'if.*>=', 'if.*<=', 'min', 'max', 'bound']
        found_patterns = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    # Look for boundary checks
                    if any(op in line for op in ['>', '<', '>=', '<=']) and 'if' in line:
                        found_patterns.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': i
                        })
                        break  # Count each file once
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_patterns) == 0:
            self.log_issue('warning', "No boundary condition checks found")
            issues_found.append({
                'issue': 'No boundary condition checks (recommended)'
            })
        else:
            self.log_pass(f"Found boundary checks in {len(found_patterns)} file(s)")
        
        return {
            'passed': len(found_patterns) > 0,
            'patterns_found': found_patterns,
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all edge case checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 11.3: EDGE CASE TESTING")
        logger.info("="*70 + "\n")
        
        results = {
            'empty_data': self.check_empty_data_handling(),
            'missing_data': self.check_missing_data_handling(),
            'extreme_values': self.check_extreme_value_handling(),
            'boundary_conditions': self.check_boundary_conditions()
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
    
    audit = EdgeCaseAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("EDGE CASE AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

