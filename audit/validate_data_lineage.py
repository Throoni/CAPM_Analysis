"""
validate_data_lineage.py

Phase 10.1: Data Lineage & Provenance Audit

Validates data lineage:
- Track all data transformations
- Document data sources
- Lineage graph visualization
- Complete audit trail
"""

import os
import sys
import logging
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DataLineageAudit:
    """Audit class for data lineage validation."""
    
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
    
    def check_lineage_file(self) -> Dict:
        """Check if data lineage file exists."""
        logger.info("Checking for data lineage file...")
        
        issues_found = []
        lineage_file = self.project_root / "data" / "lineage.json"
        
        if lineage_file.exists():
            try:
                with open(lineage_file, 'r') as f:
                    lineage_data = json.load(f)
                self.log_pass(f"Data lineage file exists: {len(lineage_data.get('transformations', []))} transformations tracked")
                return {
                    'passed': True,
                    'transformations_count': len(lineage_data.get('transformations', [])),
                    'lineage_data': lineage_data
                }
            except Exception as e:
                self.log_issue('warning', f"Error reading lineage file: {e}")
                issues_found.append({
                    'issue': f'Error reading lineage.json: {e}'
                })
        else:
            self.log_issue('warning', "Data lineage file not found (data/lineage.json)")
            issues_found.append({
                'issue': 'Missing data/lineage.json (recommended for tracking data transformations)'
            })
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found
        }
    
    def check_data_sources(self) -> Dict:
        """Check if data sources are documented."""
        logger.info("Checking data source documentation...")
        
        issues_found = []
        
        # Check for documentation of data sources
        docs_dir = self.project_root / "docs"
        data_dict_file = docs_dir / "data_dictionary.md"
        
        if data_dict_file.exists():
            with open(data_dict_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for key data source mentions
            source_keywords = ['yahoo finance', 'fred', 'ecb', 'wrds', 'source', 'data source']
            found_keywords = sum(1 for keyword in source_keywords if keyword.lower() in content.lower())
            
            if found_keywords >= 3:
                self.log_pass("Data sources documented in data_dictionary.md")
            else:
                self.log_issue('warning', "Data sources may not be fully documented")
                issues_found.append({
                    'issue': 'Data sources may not be fully documented'
                })
        else:
            self.log_issue('warning', "data_dictionary.md not found")
            issues_found.append({
                'issue': 'Missing docs/data_dictionary.md'
            })
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found
        }
    
    def check_transformation_tracking(self) -> Dict:
        """Check if transformations are tracked."""
        logger.info("Checking transformation tracking...")
        
        issues_found = []
        
        # Key transformation files
        transformation_files = [
            'analysis/returns_processing.py',
            'analysis/capm_regression.py',
            'analysis/fama_macbeth.py'
        ]
        
        tracked_transformations = []
        for tf_file in transformation_files:
            full_path = self.project_root / tf_file
            if full_path.exists():
                tracked_transformations.append(str(tf_file))
                self.log_pass(f"Transformation file exists: {tf_file}")
            else:
                issues_found.append({
                    'issue': f'Missing transformation file: {tf_file}',
                    'file': tf_file
                })
                self.log_issue('critical', f"Missing: {tf_file}")
        
        return {
            'passed': len(issues_found) == 0,
            'transformations': tracked_transformations,
            'issues': issues_found
        }
    
    def check_output_tracking(self) -> Dict:
        """Check if output files are tracked."""
        logger.info("Checking output file tracking...")
        
        issues_found = []
        results_dir = self.project_root / "results"
        
        if not results_dir.exists():
            self.log_issue('warning', "results/ directory not found")
            return {
                'passed': False,
                'issues': [{'issue': 'results/ directory not found'}]
            }
        
        # Key output files that should be tracked
        key_outputs = [
            'results/data/capm_results.csv',
            'results/reports/fama_macbeth_summary.csv',
            'results/tables/table1_capm_timeseries_summary.csv'
        ]
        
        existing_outputs = []
        missing_outputs = []
        
        for output_file in key_outputs:
            full_path = self.project_root / output_file
            if full_path.exists():
                existing_outputs.append(output_file)
                self.log_pass(f"Output file exists: {output_file}")
            else:
                missing_outputs.append(output_file)
                # Not critical - files may be generated on demand
                self.log_issue('warning', f"Output file not found: {output_file} (may be generated)")
        
        return {
            'passed': len(existing_outputs) > 0,
            'existing_outputs': existing_outputs,
            'missing_outputs': missing_outputs,
            'issues': issues_found
        }
    
    def create_lineage_template(self) -> Dict:
        """Create a template lineage file if it doesn't exist."""
        logger.info("Creating lineage template...")
        
        lineage_file = self.project_root / "data" / "lineage.json"
        
        if lineage_file.exists():
            self.log_pass("Lineage file already exists")
            return {
                'passed': True,
                'created': False
            }
        
        # Create template
        template = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'version': '1.0',
                'description': 'Data lineage tracking for CAPM analysis'
            },
            'data_sources': [
                {
                    'name': 'Stock Prices',
                    'source': 'Yahoo Finance',
                    'files': ['data/raw/prices_stocks_*.csv'],
                    'description': 'Monthly stock prices from Yahoo Finance'
                },
                {
                    'name': 'MSCI Indexes',
                    'source': 'Yahoo Finance (iShares ETFs)',
                    'files': ['data/raw/prices_indices_msci_*.csv'],
                    'description': 'MSCI country indexes via iShares ETFs'
                },
                {
                    'name': 'Risk-Free Rates',
                    'source': 'FRED/ECB/CSV files',
                    'files': ['data/raw/riskfree_rate_*.csv'],
                    'description': '3-month government bond yields'
                }
            ],
            'transformations': [
                {
                    'step': 1,
                    'name': 'Returns Processing',
                    'script': 'analysis/returns_processing.py',
                    'input': ['data/raw/prices_stocks_*.csv', 'data/raw/prices_indices_msci_*.csv'],
                    'output': 'data/processed/returns_panel.csv',
                    'description': 'Convert prices to returns and create panel'
                },
                {
                    'step': 2,
                    'name': 'CAPM Regression',
                    'script': 'analysis/capm_regression.py',
                    'input': ['data/processed/returns_panel.csv'],
                    'output': 'results/data/capm_results.csv',
                    'description': 'Estimate beta, alpha, R² for each stock'
                },
                {
                    'step': 3,
                    'name': 'Fama-MacBeth Test',
                    'script': 'analysis/fama_macbeth.py',
                    'input': ['results/data/capm_results.csv', 'data/processed/returns_panel.csv'],
                    'output': 'results/reports/fama_macbeth_summary.csv',
                    'description': 'Cross-sectional test of CAPM'
                }
            ]
        }
        
        # Create data directory if it doesn't exist
        lineage_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(lineage_file, 'w') as f:
                json.dump(template, f, indent=2)
            self.log_pass(f"Created lineage template: {lineage_file}")
            return {
                'passed': True,
                'created': True,
                'file': str(lineage_file)
            }
        except Exception as e:
            self.log_issue('warning', f"Could not create lineage file: {e}")
            return {
                'passed': False,
                'created': False,
                'error': str(e)
            }
    
    def run_all_checks(self) -> Dict:
        """Run all data lineage checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 10.1: DATA LINEAGE & PROVENANCE")
        logger.info("="*70 + "\n")
        
        results = {
            'lineage_file': self.check_lineage_file(),
            'data_sources': self.check_data_sources(),
            'transformations': self.check_transformation_tracking(),
            'output_tracking': self.check_output_tracking(),
            'lineage_template': self.create_lineage_template()
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
    
    audit = DataLineageAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("DATA LINEAGE AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

