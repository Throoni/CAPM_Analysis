"""
validate_reproducibility.py

Phase 9.2: Computational Reproducibility Audit

Validates that results are reproducible:
- Random seed verification
- Deterministic execution checks
- Environment documentation
- Platform independence tests
"""

import os
import sys
import logging
import subprocess
import platform
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ReproducibilityAudit:
    """Audit class for reproducibility validation."""
    
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
    
    def check_random_seeds(self) -> Dict:
        """Check if random seeds are set in code."""
        logger.info("Checking for random seed usage...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        seed_patterns = ['np.random.seed', 'random.seed', 'tf.random.set_seed']
        found_seeds = []
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    for pattern in seed_patterns:
                        if pattern in line:
                            found_seeds.append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': i,
                                'pattern': pattern
                            })
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(found_seeds) == 0:
            self.log_issue('warning', "No random seeds found in code")
            issues_found.append({
                'issue': 'No random seeds set - results may not be reproducible'
            })
        else:
            self.log_pass(f"Found {len(found_seeds)} random seed setting(s)")
        
        return {
            'passed': len(found_seeds) > 0,
            'seeds_found': found_seeds,
            'issues': issues_found
        }
    
    def check_environment_file(self) -> Dict:
        """Check if environment/requirements files exist."""
        logger.info("Checking environment documentation...")
        
        issues_found = []
        files_checked = []
        
        # Check for requirements files
        req_files = ['requirements.txt', 'requirements_lock.txt', 'environment.yml', 'pyproject.toml']
        for req_file in req_files:
            file_path = self.project_root / req_file
            if file_path.exists():
                files_checked.append(req_file)
                self.log_pass(f"Found: {req_file}")
            else:
                if req_file == 'requirements.txt':
                    self.log_issue('critical', f"Missing: {req_file}")
                    issues_found.append({
                        'issue': f'Missing {req_file}',
                        'file': req_file
                    })
                elif req_file == 'requirements_lock.txt':
                    self.log_issue('warning', f"Missing: {req_file} (recommended for reproducibility)")
                    issues_found.append({
                        'issue': f'Missing {req_file} (recommended)',
                        'file': req_file
                    })
        
        return {
            'passed': 'requirements.txt' in files_checked,
            'files_found': files_checked,
            'issues': issues_found
        }
    
    def check_python_version(self) -> Dict:
        """Check Python version and document it."""
        logger.info("Checking Python version...")
        
        python_version = platform.python_version()
        python_impl = platform.python_implementation()
        
        self.log_pass(f"Python {python_version} ({python_impl})")
        
        # Check if version is documented
        readme_path = self.project_root / "README.md"
        version_documented = False
        
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
                if python_version.split('.')[0] + '.' + python_version.split('.')[1] in readme_content:
                    version_documented = True
        
        if not version_documented:
            self.log_issue('warning', "Python version not documented in README")
        
        return {
            'passed': True,
            'python_version': python_version,
            'python_impl': python_impl,
            'version_documented': version_documented
        }
    
    def check_deterministic_operations(self) -> Dict:
        """Check for potentially non-deterministic operations."""
        logger.info("Checking for non-deterministic operations...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        non_deterministic_patterns = [
            ('datetime.now()', 'Uses current time - may not be reproducible'),
            ('time.time()', 'Uses current time - may not be reproducible'),
            ('uuid', 'UUID generation - not reproducible'),
        ]
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    for pattern, description in non_deterministic_patterns:
                        if pattern in line and not line.strip().startswith('#'):
                            issues_found.append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': i,
                                'issue': description,
                                'pattern': pattern
                            })
                            self.log_issue('warning',
                                         f"{description} in {py_file.name}:{i}")
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(issues_found) == 0:
            self.log_pass("No obvious non-deterministic operations found")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found
        }
    
    def check_file_paths(self) -> Dict:
        """Check for hardcoded absolute paths."""
        logger.info("Checking for hardcoded paths...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        audit_dir = self.project_root / "audit"
        
        # Common absolute path patterns
        path_patterns = ['/Users/', '/home/', 'C:\\', 'D:\\']
        
        for directory in [analysis_dir, audit_dir]:
            if not directory.exists():
                continue
                
            for py_file in directory.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                    for i, line in enumerate(lines, 1):
                        for pattern in path_patterns:
                            if pattern in line and not line.strip().startswith('#'):
                                issues_found.append({
                                    'file': str(py_file.relative_to(self.project_root)),
                                    'line': i,
                                    'issue': 'Hardcoded absolute path',
                                    'pattern': pattern
                                })
                                self.log_issue('warning',
                                             f"Hardcoded path in {py_file.name}:{i}")
                except Exception as e:
                    self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(issues_found) == 0:
            self.log_pass("No hardcoded absolute paths found")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all reproducibility checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 9.2: COMPUTATIONAL REPRODUCIBILITY")
        logger.info("="*70 + "\n")
        
        results = {
            'random_seeds': self.check_random_seeds(),
            'environment_files': self.check_environment_file(),
            'python_version': self.check_python_version(),
            'deterministic_ops': self.check_deterministic_operations(),
            'file_paths': self.check_file_paths()
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
    
    audit = ReproducibilityAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("REPRODUCIBILITY AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

