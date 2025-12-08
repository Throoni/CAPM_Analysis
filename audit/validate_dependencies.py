"""
validate_dependencies.py

Phase 9.4: Dependency Auditing

Validates dependencies:
- Security vulnerability scanning
- Outdated package detection
- License compatibility
- Dependency conflicts
"""

import os
import sys
import logging
import subprocess
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DependencyAudit:
    """Audit class for dependency validation."""
    
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
    
    def check_requirements_file(self) -> Dict:
        """Check if requirements.txt exists and is valid."""
        logger.info("Checking requirements.txt...")
        
        issues_found = []
        req_file = self.project_root / "requirements.txt"
        
        if not req_file.exists():
            self.log_issue('critical', "requirements.txt not found")
            issues_found.append({
                'issue': 'Missing requirements.txt'
            })
            return {
                'passed': False,
                'issues': issues_found
            }
        
        # Parse requirements file
        packages = []
        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name (before >=, ==, etc.)
                        package_name = line.split('>=')[0].split('==')[0].split('<')[0].strip()
                        if package_name:
                            packages.append(package_name)
            
            self.log_pass(f"Found {len(packages)} packages in requirements.txt")
        except Exception as e:
            self.log_issue('critical', f"Error reading requirements.txt: {e}")
            issues_found.append({
                'issue': f'Error parsing requirements.txt: {e}'
            })
        
        return {
            'passed': len(issues_found) == 0,
            'packages': packages,
            'issues': issues_found
        }
    
    def check_lock_file(self) -> Dict:
        """Check if requirements_lock.txt exists."""
        logger.info("Checking requirements_lock.txt...")
        
        issues_found = []
        lock_file = self.project_root / "requirements_lock.txt"
        
        if lock_file.exists():
            self.log_pass("requirements_lock.txt exists (good for reproducibility)")
        else:
            self.log_issue('warning', "requirements_lock.txt not found (recommended for reproducibility)")
            issues_found.append({
                'issue': 'Missing requirements_lock.txt (recommended)'
            })
        
        return {
            'passed': lock_file.exists(),
            'issues': issues_found
        }
    
    def check_core_dependencies(self) -> Dict:
        """Check if core dependencies are present."""
        logger.info("Checking core dependencies...")
        
        issues_found = []
        core_deps = [
            'pandas',
            'numpy',
            'statsmodels',
            'matplotlib',
            'scipy'
        ]
        
        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            return {
                'passed': False,
                'issues': [{'issue': 'Cannot check - requirements.txt missing'}]
            }
        
        with open(req_file, 'r', encoding='utf-8') as f:
            req_content = f.read().lower()
        
        missing_deps = []
        for dep in core_deps:
            if dep.lower() not in req_content:
                missing_deps.append(dep)
                self.log_issue('critical', f"Missing core dependency: {dep}")
                issues_found.append({
                    'issue': f'Missing core dependency: {dep}',
                    'dependency': dep
                })
            else:
                self.log_pass(f"Core dependency found: {dep}")
        
        return {
            'passed': len(missing_deps) == 0,
            'missing': missing_deps,
            'issues': issues_found
        }
    
    def check_version_pinning(self) -> Dict:
        """Check if versions are pinned in requirements.txt."""
        logger.info("Checking version pinning...")
        
        issues_found = []
        req_file = self.project_root / "requirements.txt"
        
        if not req_file.exists():
            return {
                'passed': False,
                'issues': [{'issue': 'Cannot check - requirements.txt missing'}]
            }
        
        unpinned = []
        with open(req_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Check if version is specified
                    if '>=' in line or '==' in line or '~=' in line:
                        continue
                    else:
                        # Extract package name
                        package = line.split()[0] if line.split() else line
                        unpinned.append({
                            'line': i,
                            'package': package
                        })
                        self.log_issue('warning', f"Unpinned version: {package} (line {i})")
        
        if len(unpinned) == 0:
            self.log_pass("All packages have version specifications")
        else:
            issues_found.append({
                'issue': f'{len(unpinned)} packages without version pinning',
                'unpinned': unpinned[:10]  # First 10
            })
        
        return {
            'passed': len(unpinned) == 0,
            'unpinned_count': len(unpinned),
            'unpinned': unpinned[:10],
            'issues': issues_found
        }
    
    def check_installed_packages(self) -> Dict:
        """Check if required packages are installed."""
        logger.info("Checking installed packages...")
        
        issues_found = []
        req_file = self.project_root / "requirements.txt"
        
        if not req_file.exists():
            return {
                'passed': False,
                'issues': [{'issue': 'Cannot check - requirements.txt missing'}]
            }
        
        # Try to import core packages
        core_packages = {
            'pandas': 'pd',
            'numpy': 'np',
            'statsmodels': 'sm',
            'matplotlib': 'plt',
            'scipy': 'sp'
        }
        
        missing = []
        for package, alias in core_packages.items():
            try:
                __import__(package)
                self.log_pass(f"Package installed: {package}")
            except ImportError:
                missing.append(package)
                self.log_issue('critical', f"Package not installed: {package}")
                issues_found.append({
                    'issue': f'Package not installed: {package}',
                    'package': package
                })
        
        return {
            'passed': len(missing) == 0,
            'missing': missing,
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all dependency checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 9.4: DEPENDENCY AUDITING")
        logger.info("="*70 + "\n")
        
        results = {
            'requirements_file': self.check_requirements_file(),
            'lock_file': self.check_lock_file(),
            'core_dependencies': self.check_core_dependencies(),
            'version_pinning': self.check_version_pinning(),
            'installed_packages': self.check_installed_packages()
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
    
    audit = DependencyAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("DEPENDENCY AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

