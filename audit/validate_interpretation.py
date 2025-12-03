"""
validate_interpretation.py

Phase 6: Interpretation & Reporting Audit
Validates that interpretations match results and tables/figures are accurate.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict

from analysis.config import RESULTS_REPORTS_DIR, RESULTS_PLOTS_DIR

logger = logging.getLogger(__name__)


class InterpretationAudit:
    """Audit class for interpretation and reporting validation."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed = []
        
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
    
    def check_interpretation_consistency(self) -> Dict:
        """Check that interpretations match results."""
        logger.info("="*70)
        logger.info("Checking interpretation consistency")
        logger.info("="*70)
        
        issues_found = []
        
        # Check economic interpretation file
        interpretation_file = os.path.join(RESULTS_REPORTS_DIR, "economic_interpretation.txt")
        if os.path.exists(interpretation_file):
            with open(interpretation_file, 'r') as f:
                interpretation_text = f.read()
            
            # Check that key findings are mentioned
            key_findings = [
                'gamma_1',
                'Fama-MacBeth',
                'CAPM',
                'beta',
                'reject'
            ]
            
            for finding in key_findings:
                if finding.lower() not in interpretation_text.lower():
                    issues_found.append({
                        'issue': f'Key finding "{finding}" not mentioned in interpretation'
                    })
                    self.log_issue('warning',
                                 f"Key finding '{finding}' not mentioned in interpretation")
                else:
                    self.log_pass(f"Key finding '{finding}' mentioned in interpretation")
        else:
            issues_found.append({
                'issue': 'Economic interpretation file not found'
            })
            self.log_issue('warning', "Economic interpretation file not found")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Interpretation consistency: {len(issues_found)} issues found"
        }
    
    def check_table_accuracy(self) -> Dict:
        """Check that tables match source data."""
        logger.info("="*70)
        logger.info("Checking table accuracy")
        logger.info("="*70)
        
        issues_found = []
        
        # Check that table files exist
        tables_dir = os.path.join(RESULTS_REPORTS_DIR, "..", "tables")
        if not os.path.exists(tables_dir):
            tables_dir = RESULTS_REPORTS_DIR
        
        expected_tables = [
            'table1_capm_timeseries_summary.csv',
            'table2_fama_macbeth_results.csv',
            'table3_subperiod_results.csv',
            'table4_country_level_results.csv',
            'table5_beta_sorted_portfolios.csv',
            'table6_descriptive_statistics.csv'
        ]
        
        for table_file in expected_tables:
            table_path = os.path.join(tables_dir, table_file)
            if os.path.exists(table_path):
                try:
                    df = pd.read_csv(table_path)
                    if len(df) == 0:
                        issues_found.append({
                            'table': table_file,
                            'issue': 'Table is empty'
                        })
                        self.log_issue('critical', f"Table {table_file} is empty")
                    else:
                        self.log_pass(f"Table {table_file} exists and has data ({len(df)} rows)")
                except Exception as e:
                    issues_found.append({
                        'table': table_file,
                        'issue': f'Error reading table: {str(e)}'
                    })
                    self.log_issue('critical', f"Error reading table {table_file}: {e}")
            else:
                issues_found.append({
                    'table': table_file,
                    'issue': 'Table file not found'
                })
                self.log_issue('warning', f"Table {table_file} not found")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Table accuracy: {len(issues_found)} issues found"
        }
    
    def check_figure_existence(self) -> Dict:
        """Check that expected figures exist."""
        logger.info("="*70)
        logger.info("Checking figure existence")
        logger.info("="*70)
        
        issues_found = []
        
        plots_dir = RESULTS_PLOTS_DIR
        
        expected_figures = [
            'beta_distribution_by_country.png',
            'r2_distribution_by_country.png',
            'gamma1_timeseries.png',
            'beta_vs_return_scatter.png',
            'beta_sorted_returns.png',
            'fm_gamma1_by_country.png'
        ]
        
        for fig_file in expected_figures:
            fig_path = os.path.join(plots_dir, fig_file)
            if os.path.exists(fig_path):
                file_size = os.path.getsize(fig_path)
                if file_size < 1000:  # Less than 1KB is suspicious
                    issues_found.append({
                        'figure': fig_file,
                        'issue': f'Figure file is very small ({file_size} bytes)'
                    })
                    self.log_issue('warning', f"Figure {fig_file} is very small ({file_size} bytes)")
                else:
                    self.log_pass(f"Figure {fig_file} exists ({file_size/1024:.1f} KB)")
            else:
                issues_found.append({
                    'figure': fig_file,
                    'issue': 'Figure file not found'
                })
                self.log_issue('warning', f"Figure {fig_file} not found")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'summary': f"Figure existence: {len(issues_found)} issues found"
        }
    
    def run_all_checks(self) -> Dict:
        """Run all interpretation and reporting checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 6: INTERPRETATION & REPORTING AUDIT")
        logger.info("="*70 + "\n")
        
        results = {
            'interpretation_consistency': self.check_interpretation_consistency(),
            'table_accuracy': self.check_table_accuracy(),
            'figure_existence': self.check_figure_existence()
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

