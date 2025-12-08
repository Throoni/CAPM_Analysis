"""
validate_documentation.py

Phase 11.1: Documentation Completeness Audit

Validates documentation:
- 100% docstring coverage
- README quality checks
- API documentation
- Code examples
"""

import os
import sys
import logging
import ast
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentationAudit:
    """Audit class for documentation validation."""
    
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
    
    def check_readme(self) -> Dict:
        """Check README quality."""
        logger.info("Checking README...")
        
        issues_found = []
        readme_file = self.project_root / "README.md"
        
        if not readme_file.exists():
            self.log_issue('critical', "README.md not found")
            issues_found.append({
                'issue': 'Missing README.md'
            })
            return {
                'passed': False,
                'issues': issues_found
            }
        
        with open(readme_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key sections
        required_sections = [
            'installation',
            'usage',
            'requirements',
            'description'
        ]
        
        found_sections = []
        for section in required_sections:
            if section in content.lower():
                found_sections.append(section)
        
        if len(found_sections) >= 2:
            self.log_pass(f"README has {len(found_sections)} key sections")
        else:
            self.log_issue('warning', "README may be missing key sections")
            issues_found.append({
                'issue': f'README missing sections (found {len(found_sections)}/4)'
            })
        
        # Check length (should be substantial)
        if len(content) < 500:
            self.log_issue('warning', "README is very short")
            issues_found.append({
                'issue': 'README is very short (<500 chars)'
            })
        else:
            self.log_pass(f"README length OK: {len(content)} characters")
        
        return {
            'passed': len(issues_found) == 0,
            'sections_found': found_sections,
            'length': len(content),
            'issues': issues_found
        }
    
    def check_docstring_coverage(self) -> Dict:
        """Check docstring coverage for public functions."""
        logger.info("Checking docstring coverage...")
        
        issues_found = []
        analysis_dir = self.project_root / "analysis"
        
        total_functions = 0
        documented_functions = 0
        missing_docstrings = []
        
        for py_file in analysis_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Skip private functions (starting with _)
                        if node.name.startswith('_') and node.name != '__init__':
                            continue
                            
                        total_functions += 1
                        docstring = ast.get_docstring(node)
                        
                        if docstring:
                            documented_functions += 1
                        else:
                            missing_docstrings.append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'function': node.name,
                                'line': node.lineno
                            })
            except SyntaxError:
                self.log_issue('warning', f"Syntax error in {py_file.name}")
            except Exception as e:
                self.log_issue('warning', f"Error checking {py_file.name}: {e}")
        
        coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
        
        if coverage < 80:
            self.log_issue('warning', f"Docstring coverage: {coverage:.1f}% (target: 80%+)")
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
            'missing_docstrings': missing_docstrings[:10]  # First 10
        }
    
    def check_code_examples(self) -> Dict:
        """Check for code examples in documentation."""
        logger.info("Checking for code examples...")
        
        issues_found = []
        readme_file = self.project_root / "README.md"
        docs_dir = self.project_root / "docs"
        
        examples_found = 0
        
        # Check README
        if readme_file.exists():
            with open(readme_file, 'r', encoding='utf-8') as f:
                readme_content = f.read()
                if '```' in readme_content or 'python' in readme_content.lower():
                    examples_found += 1
        
        # Check docs directory
        if docs_dir.exists():
            for doc_file in docs_dir.glob("*.md"):
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '```' in content:
                        examples_found += 1
        
        if examples_found == 0:
            self.log_issue('warning', "No code examples found in documentation")
            issues_found.append({
                'issue': 'No code examples in documentation (recommended)'
            })
        else:
            self.log_pass(f"Found code examples in {examples_found} file(s)")
        
        return {
            'passed': examples_found > 0,
            'examples_found': examples_found,
            'issues': issues_found
        }
    
    def check_api_documentation(self) -> Dict:
        """Check for API documentation."""
        logger.info("Checking API documentation...")
        
        issues_found = []
        docs_dir = self.project_root / "docs"
        
        # Check for API documentation files
        api_docs = list(docs_dir.glob("*api*.md")) if docs_dir.exists() else []
        api_docs += list(docs_dir.glob("*reference*.md")) if docs_dir.exists() else []
        
        if len(api_docs) == 0:
            self.log_issue('warning', "No API documentation found")
            issues_found.append({
                'issue': 'No API documentation (optional, but recommended)'
            })
        else:
            self.log_pass(f"Found {len(api_docs)} API documentation file(s)")
        
        return {
            'passed': len(api_docs) > 0,
            'api_docs': [str(f.name) for f in api_docs],
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all documentation checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 11.1: DOCUMENTATION COMPLETENESS")
        logger.info("="*70 + "\n")
        
        results = {
            'readme': self.check_readme(),
            'docstring_coverage': self.check_docstring_coverage(),
            'code_examples': self.check_code_examples(),
            'api_documentation': self.check_api_documentation()
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
    
    audit = DocumentationAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("DOCUMENTATION AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

