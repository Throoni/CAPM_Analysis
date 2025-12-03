"""
validate_portfolio_recommendation.py

Validates portfolio recommendation analysis.
Checks recommendation consistency with findings, justification completeness, and report structure.
"""

import os
import logging
import pandas as pd
from typing import Dict

from analysis.config import RESULTS_REPORTS_DIR

logger = logging.getLogger(__name__)


class PortfolioRecommendationAudit:
    """Audit class for portfolio recommendation analysis."""
    
    def __init__(self):
        self.issues = []
        self.passed_checks = []
    
    def log_issue(self, severity: str, message: str, details: dict = None):
        """Log an issue."""
        issue = {
            'severity': severity,
            'message': message,
            'details': details or {}
        }
        self.issues.append(issue)
        if severity == 'critical':
            logger.error(f"[CRITICAL] {message}")
        elif severity == 'warning':
            logger.warning(f"[WARNING] {message}")
        else:
            logger.info(f"[INFO] {message}")
    
    def log_pass(self, message: str):
        """Log a passed check."""
        self.passed_checks.append(message)
        logger.info(f"[PASS] {message}")
    
    def check_recommendation_file_exists(self) -> bool:
        """Check if portfolio recommendation report exists."""
        logger.info("="*70)
        logger.info("Checking portfolio recommendation file")
        logger.info("="*70)
        
        report_file = os.path.join(RESULTS_REPORTS_DIR, "Portfolio_Recommendation.md")
        
        if not os.path.exists(report_file):
            self.log_issue('critical', "Portfolio recommendation report not found")
            return False
        
        file_size = os.path.getsize(report_file)
        if file_size < 1000:  # Less than 1KB is suspicious
            self.log_issue('warning', f"Recommendation report is very small ({file_size} bytes)")
        else:
            self.log_pass(f"Portfolio recommendation report exists ({file_size:,} bytes)")
        
        return True
    
    def validate_recommendation_structure(self) -> Dict:
        """Validate recommendation report structure and completeness."""
        logger.info("="*70)
        logger.info("Validating recommendation structure")
        logger.info("="*70)
        
        report_file = os.path.join(RESULTS_REPORTS_DIR, "Portfolio_Recommendation.md")
        
        if not os.path.exists(report_file):
            return {'passed': False, 'issues': self.issues}
        
        with open(report_file, 'r') as f:
            content = f.read()
        
        issues_found = []
        
        # Check for required sections
        required_sections = [
            'Executive Summary',
            'Key Findings Summary',
            'Recommendation:',
            'Justification',
            'Implementation Strategy',
            'Risks and Considerations'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            issues_found.append({
                'issue': f'Missing sections: {missing_sections}',
                'severity': 'warning'
            })
            self.log_issue('warning', f"Missing sections: {missing_sections}")
        else:
            self.log_pass("All required sections present")
        
        # Check for recommendation (Active, Passive, or Hybrid)
        has_recommendation = any(keyword in content for keyword in ['Active', 'Passive', 'Hybrid'])
        if not has_recommendation:
            issues_found.append({
                'issue': 'No clear recommendation (Active/Passive/Hybrid) found',
                'severity': 'critical'
            })
            self.log_issue('critical', "No clear recommendation found")
        else:
            self.log_pass("Recommendation strategy identified")
        
        # Check for justification
        if 'Justification' in content or 'justification' in content.lower():
            self.log_pass("Justification section present")
        else:
            issues_found.append({
                'issue': 'Justification section missing or unclear',
                'severity': 'warning'
            })
            self.log_issue('warning', "Justification may be incomplete")
        
        # Check for implementation strategy
        if 'Implementation' in content or 'implementation' in content.lower():
            self.log_pass("Implementation strategy present")
        else:
            issues_found.append({
                'issue': 'Implementation strategy missing',
                'severity': 'warning'
            })
            self.log_issue('warning', "Implementation strategy may be missing")
        
        # Check for risks discussion
        if 'Risk' in content or 'risk' in content.lower():
            self.log_pass("Risks and considerations present")
        else:
            issues_found.append({
                'issue': 'Risks and considerations missing',
                'severity': 'warning'
            })
            self.log_issue('warning', "Risks discussion may be missing")
        
        self.issues.extend(issues_found)
        
        return {
            'passed': len([i for i in issues_found if i.get('severity') == 'critical']) == 0,
            'issues': issues_found,
            'summary': f"Recommendation structure validation: {len([i for i in issues_found if i.get('severity') == 'critical'])} critical, {len([i for i in issues_found if i.get('severity') == 'warning'])} warnings"
        }
    
    def validate_recommendation_consistency(self) -> Dict:
        """Validate that recommendation is consistent with empirical findings."""
        logger.info("="*70)
        logger.info("Validating recommendation consistency with findings")
        logger.info("="*70)
        
        report_file = os.path.join(RESULTS_REPORTS_DIR, "Portfolio_Recommendation.md")
        fm_file = os.path.join(RESULTS_REPORTS_DIR, "fama_macbeth_summary.csv")
        
        if not os.path.exists(report_file):
            return {'passed': True, 'issues': []}
        
        with open(report_file, 'r') as f:
            content = f.read()
        
        issues_found = []
        
        # Check if recommendation mentions CAPM rejection
        if 'CAPM' in content and ('reject' in content.lower() or 'fail' in content.lower()):
            self.log_pass("Recommendation acknowledges CAPM rejection")
        else:
            issues_found.append({
                'issue': 'Recommendation may not acknowledge CAPM rejection',
                'severity': 'warning'
            })
            self.log_issue('warning', "Recommendation should acknowledge CAPM findings")
        
        # Check if recommendation mentions negative beta-return relationship
        if 'negative' in content.lower() and ('beta' in content.lower() or 'return' in content.lower()):
            self.log_pass("Recommendation mentions negative beta-return relationship")
        else:
            issues_found.append({
                'issue': 'Recommendation may not mention negative beta-return relationship',
                'severity': 'info'
            })
            self.log_issue('info', "Recommendation could mention negative beta-return finding")
        
        # Check if recommendation is consistent with findings
        # If CAPM is rejected and beta-return is negative, active management makes sense
        if 'Active' in content:
            # Check if it references the right evidence
            has_evidence = any(keyword in content.lower() for keyword in [
                'inefficiency', 'mispricing', 'capm reject', 'negative beta'
            ])
            if has_evidence:
                self.log_pass("Active recommendation is supported by evidence")
            else:
                issues_found.append({
                    'issue': 'Active recommendation may lack supporting evidence in text',
                    'severity': 'warning'
                })
                self.log_issue('warning', "Active recommendation should reference supporting evidence")
        
        self.issues.extend(issues_found)
        
        return {
            'passed': len([i for i in issues_found if i.get('severity') == 'critical']) == 0,
            'issues': issues_found,
            'summary': f"Consistency validation: {len(issues_found)} issues found"
        }
    
    def run_all_checks(self) -> Dict:
        """Run all portfolio recommendation validation checks."""
        logger.info("="*70)
        logger.info("PORTFOLIO RECOMMENDATION VALIDATION")
        logger.info("="*70)
        
        # Check 1: File exists
        file_exists = self.check_recommendation_file_exists()
        
        if not file_exists:
            return {
                'passed': False,
                'issues': self.issues,
                'summary': 'Portfolio recommendation validation failed: file not found'
            }
        
        # Check 2: Validate structure
        structure_results = self.validate_recommendation_structure()
        
        # Check 3: Validate consistency
        consistency_results = self.validate_recommendation_consistency()
        
        # Summary
        total_critical = len([i for i in self.issues if i.get('severity') == 'critical'])
        total_warnings = len([i for i in self.issues if i.get('severity') == 'warning'])
        
        return {
            'passed': total_critical == 0,
            'issues': self.issues,
            'passed_checks': self.passed_checks,
            'summary': {
                'total_checks': len(self.passed_checks) + len(self.issues),
                'passed_checks': len(self.passed_checks),
                'critical_issues': total_critical,
                'warnings': total_warnings
            }
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    audit = PortfolioRecommendationAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("PORTFOLIO RECOMMENDATION VALIDATION COMPLETE")
    print("="*70)
    print(f"Passed: {results['passed']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")

