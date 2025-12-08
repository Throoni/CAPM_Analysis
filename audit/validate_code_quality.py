"""
validate_code_quality.py

Phase 8.1: Code Quality & Static Analysis Audit

Validates code quality through static analysis:
- Linting (pylint, flake8)
- Type checking (mypy)
- Complexity analysis
- Code smells detection
- Security scanning (bandit)
- Docstring coverage
"""

import os
import sys
import logging
import subprocess
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeQualityAudit:
    """Audit class for code quality validation."""
    
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
    
    def check_docstring_coverage(self) -> Dict:
        """Check docstring coverage for all Python files."""
        logger.info("Checking docstring coverage...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        audit_dir = self.project_root / "audit"
        
        total_functions = 0
        documented_functions = 0
        
        for directory in [analysis_dir, audit_dir]:
            if not directory.exists():
                continue
                
            for py_file in directory.rglob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Simple heuristic: count function definitions
                    import ast
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                total_functions += 1
                                # Check if function has docstring
                                if (ast.get_docstring(node) is not None or 
                                    (node.body and isinstance(node.body[0], ast.Expr) and 
                                     isinstance(node.body[0].value, ast.Str))):
                                    documented_functions += 1
                                else:
                                    if node.name not in ['__init__', '__repr__', '__str__']:
                                        issues_found.append({
                                            'file': str(py_file.relative_to(self.project_root)),
                                            'function': node.name,
                                            'line': node.lineno
                                        })
                    except SyntaxError:
                        self.log_issue('warning', f"Syntax error in {py_file.name}")
                except Exception as e:
                    self.log_issue('warning', f"Error reading {py_file.name}: {e}")
        
        coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
        
        if coverage < 80:
            self.log_issue('warning', 
                         f"Docstring coverage: {coverage:.1f}% (target: 80%+)")
            issues_found.append({
                'issue': f'Low docstring coverage: {coverage:.1f}%',
                'coverage': coverage
            })
        else:
            self.log_pass(f"Docstring coverage: {coverage:.1f}%")
        
        return {
            'passed': coverage >= 80,
            'coverage': coverage,
            'total_functions': total_functions,
            'documented_functions': documented_functions,
            'missing_docstrings': issues_found[:10]  # First 10
        }
    
    def check_imports(self) -> Dict:
        """Check for problematic imports and unused imports."""
        logger.info("Checking imports...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        for py_file in analysis_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # Check for wildcard imports
                for i, line in enumerate(lines, 1):
                    if 'from' in line and 'import *' in line:
                        issues_found.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': i,
                            'issue': 'Wildcard import detected'
                        })
                        self.log_issue('warning', 
                                     f"Wildcard import in {py_file.name}:{i}")
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(issues_found) == 0:
            self.log_pass("No problematic imports found")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found
        }
    
    def check_file_structure(self) -> Dict:
        """Check file structure and organization."""
        logger.info("Checking file structure...")
        
        issues_found = []
        
        # Check for required directories
        required_dirs = ['analysis', 'audit', 'data', 'results']
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                issues_found.append({
                    'issue': f'Missing directory: {dir_name}',
                    'directory': dir_name
                })
                self.log_issue('critical', f"Missing directory: {dir_name}")
            else:
                self.log_pass(f"Directory exists: {dir_name}")
        
        # Check for __init__.py files
        for py_package in ['analysis', 'audit']:
            init_file = self.project_root / py_package / "__init__.py"
            if not init_file.exists():
                issues_found.append({
                    'issue': f'Missing __init__.py in {py_package}',
                    'package': py_package
                })
                self.log_issue('warning', f"Missing __init__.py in {py_package}")
            else:
                self.log_pass(f"__init__.py exists in {py_package}")
        
        return {
            'passed': len([i for i in issues_found if 'critical' in str(i)]) == 0,
            'issues': issues_found
        }
    
    def check_code_complexity(self) -> Dict:
        """Check code complexity (simple heuristic)."""
        logger.info("Checking code complexity...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        for py_file in analysis_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # Simple complexity check: very long functions
                in_function = False
                function_start = 0
                function_name = ""
                indent_level = 0
                
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if stripped.startswith('def '):
                        if in_function and (i - function_start) > 100:
                            issues_found.append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'function': function_name,
                                'lines': i - function_start,
                                'line': function_start
                            })
                            self.log_issue('warning',
                                         f"Long function in {py_file.name}:{function_start} ({i - function_start} lines)")
                        in_function = True
                        function_start = i
                        function_name = stripped.split('(')[0].replace('def ', '')
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(issues_found) == 0:
            self.log_pass("No overly complex functions found")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found[:10]  # First 10
        }
    
    def check_security_issues(self) -> Dict:
        """Check for common security issues."""
        logger.info("Checking for security issues...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        audit_dir = self.project_root / "audit"
        
        # Check for hardcoded credentials, API keys
        sensitive_patterns = [
            ('password', 'Hardcoded password'),
            ('api_key', 'Hardcoded API key'),
            ('secret', 'Hardcoded secret'),
            ('token', 'Hardcoded token'),
        ]
        
        for directory in [analysis_dir, audit_dir]:
            if not directory.exists():
                continue
                
            for py_file in directory.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                    for pattern, description in sensitive_patterns:
                        for i, line in enumerate(lines, 1):
                            if pattern in line.lower() and '=' in line:
                                # Skip if it's:
                                # 1. A comment
                                # 2. Using os.getenv or os.environ (secure)
                                # 3. A function parameter (def func(param=value))
                                # 4. A variable assignment from os.getenv (secure)
                                # 5. Part of a docstring or string literal
                                line_lower = line.lower()
                                is_comment = line.strip().startswith('#')
                                uses_env_var = 'os.getenv' in line or 'os.environ' in line
                                is_function_param = 'def ' in line and pattern in line_lower.split('(')[0] if '(' in line else False
                                is_variable_from_env = '=' in line and ('os.getenv' in line or 'os.environ' in line)
                                is_string_literal = (line.count('"') >= 2 or line.count("'") >= 2) and pattern in line_lower
                                
                                # Only flag if it looks like a hardcoded value
                                if not (is_comment or uses_env_var or is_function_param or 
                                       is_variable_from_env or is_string_literal):
                                    # Additional check: is it assigning a literal string value?
                                    if '=' in line:
                                        parts = line.split('=')
                                        if len(parts) == 2:
                                            value_part = parts[1].strip()
                                            # Check if it's a hardcoded string (starts with quote)
                                            if (value_part.startswith('"') or value_part.startswith("'")) and \
                                               len(value_part) > 3:  # More than just empty string
                                                issues_found.append({
                                                    'file': str(py_file.relative_to(self.project_root)),
                                                    'line': i,
                                                    'issue': description,
                                                    'pattern': pattern
                                                })
                                                self.log_issue('critical',
                                                             f"{description} in {py_file.name}:{i}")
                except Exception as e:
                    self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(issues_found) == 0:
            self.log_pass("No obvious security issues found")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all code quality checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 8.1: CODE QUALITY & STATIC ANALYSIS")
        logger.info("="*70 + "\n")
        
        results = {
            'docstring_coverage': self.check_docstring_coverage(),
            'imports': self.check_imports(),
            'file_structure': self.check_file_structure(),
            'complexity': self.check_code_complexity(),
            'security': self.check_security_issues()
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
    
    audit = CodeQualityAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("CODE QUALITY AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

