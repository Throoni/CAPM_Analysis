"""
validate_error_handling.py

Phase 11.2: Error Handling Validation Audit

Validates error handling:
- Exception handling coverage
- Error message quality
- Logging completeness
- Recovery mechanisms
"""

import os
import sys
import logging
import ast
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorHandlingAudit:
    """Audit class for error handling validation."""
    
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
    
    def check_exception_handling(self) -> Dict:
        """Check for exception handling."""
        logger.info("Checking exception handling...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        total_functions = 0
        functions_with_try = 0
        functions_without_try = []
        
        for py_file in analysis_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name.startswith('_') and node.name != '__init__':
                            continue
                            
                        total_functions += 1
                        
                        # Check if function has try-except
                        has_try = any(isinstance(child, ast.Try) for child in ast.walk(node))
                        
                        if has_try:
                            functions_with_try += 1
                        else:
                            # Check if function does file I/O or external calls
                            function_code = ast.get_source_segment(content, node)
                            if function_code:
                                risky_patterns = ['open(', 'pd.read', 'requests.', 'yf.', '__import__']
                                if any(pattern in function_code for pattern in risky_patterns):
                                    functions_without_try.append({
                                        'file': str(py_file.relative_to(self.project_root)),
                                        'function': node.name,
                                        'line': node.lineno
                                    })
            except SyntaxError:
                self.log_issue('warning', f"Syntax error in {py_file.name}")
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        coverage = (functions_with_try / total_functions * 100) if total_functions > 0 else 0
        
        if len(functions_without_try) > 0:
            self.log_issue('warning', f"{len(functions_without_try)} functions with risky operations lack try-except")
            for func in functions_without_try[:5]:
                issues_found.append({
                    'issue': f"Function {func['function']} in {func['file']} lacks exception handling",
                    'file': func['file'],
                    'function': func['function']
                })
        else:
            self.log_pass("Functions with risky operations have exception handling")
        
        return {
            'passed': len(functions_without_try) == 0,
            'coverage': coverage,
            'functions_with_try': functions_with_try,
            'functions_without_try': functions_without_try[:10],
            'issues': issues_found
        }
    
    def check_error_messages(self) -> Dict:
        """Check quality of error messages."""
        logger.info("Checking error message quality...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        error_patterns = ['raise', 'except', 'error', 'exception']
        functions_with_errors = 0
        functions_with_good_errors = 0
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    if 'raise' in line.lower() and 'Exception' in line:
                        functions_with_errors += 1
                        # Check if error message is informative
                        if ('"' in line or "'" in line) and len(line) > 30:
                            functions_with_good_errors += 1
                        else:
                            issues_found.append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': i,
                                'issue': 'Error message may be too short or generic'
                            })
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if functions_with_errors > 0:
            quality = (functions_with_good_errors / functions_with_errors * 100)
            if quality < 80:
                self.log_issue('warning', f"Error message quality: {quality:.1f}%")
            else:
                self.log_pass(f"Error message quality: {quality:.1f}%")
        
        return {
            'passed': len(issues_found) == 0,
            'error_functions': functions_with_errors,
            'good_error_functions': functions_with_good_errors,
            'issues': issues_found
        }
    
    def check_logging(self) -> Dict:
        """Check for logging usage."""
        logger.info("Checking logging usage...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        files_with_logging = 0
        files_without_logging = []
        
        for py_file in analysis_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'logger' in content.lower() or 'logging' in content.lower():
                    files_with_logging += 1
                else:
                    # Check if file has substantial code
                    if len(content) > 200:  # Substantial file
                        files_without_logging.append(str(py_file.relative_to(self.project_root)))
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if len(files_without_logging) > 0:
            self.log_issue('warning', f"{len(files_without_logging)} files without logging")
            issues_found.append({
                'issue': f'{len(files_without_logging)} files lack logging',
                'files': files_without_logging[:10]
            })
        else:
            self.log_pass(f"All substantial files use logging ({files_with_logging} files)")
        
        return {
            'passed': len(files_without_logging) == 0,
            'files_with_logging': files_with_logging,
            'files_without_logging': files_without_logging,
            'issues': issues_found
        }
    
    def check_recovery_mechanisms(self) -> Dict:
        """Check for error recovery mechanisms."""
        logger.info("Checking error recovery mechanisms...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        recovery_patterns = ['finally', 'else:', 'continue', 'pass', 'return', 'default']
        files_with_recovery = 0
        
        for py_file in analysis_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'try' in content.lower():
                    # Check for recovery patterns
                    has_recovery = any(pattern in content for pattern in recovery_patterns)
                    if has_recovery:
                        files_with_recovery += 1
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        if files_with_recovery > 0:
            self.log_pass(f"Found recovery mechanisms in {files_with_recovery} file(s)")
        else:
            self.log_issue('warning', "Limited error recovery mechanisms found")
            issues_found.append({
                'issue': 'Limited error recovery (recommended: finally blocks, default values)'
            })
        
        return {
            'passed': files_with_recovery > 0,
            'files_with_recovery': files_with_recovery,
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all error handling checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 11.2: ERROR HANDLING VALIDATION")
        logger.info("="*70 + "\n")
        
        results = {
            'exception_handling': self.check_exception_handling(),
            'error_messages': self.check_error_messages(),
            'logging': self.check_logging(),
            'recovery_mechanisms': self.check_recovery_mechanisms()
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
    
    audit = ErrorHandlingAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("ERROR HANDLING AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

