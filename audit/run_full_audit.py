"""
Comprehensive CAPM Analysis Audit Orchestrator.

This module coordinates the execution of all audit phases to ensure the
integrity and correctness of the CAPM analysis pipeline.

Audit phases:
    Phase 1: Raw Data Validation
        - Price data completeness and format
        - Date range coverage
        - Missing value patterns

    Phase 2: Risk-Free Rate Validation
        - Rate magnitude verification
        - Currency consistency
        - Temporal coverage

    Phase 3: Processed Data Validation
        - Returns calculation correctness
        - Excess returns methodology
        - Panel data structure

    Phase 4: Statistical Methodology
        - Regression assumptions (normality, homoscedasticity)
        - Serial correlation tests
        - Multicollinearity checks

    Phase 5: Financial Calculations
        - Beta interpretation (economic plausibility)
        - Alpha significance testing
        - Sharpe ratio verification

    Phase 6: Results Interpretation
        - Consistency across specifications
        - Economic significance assessment
        - Robustness of conclusions

Output:
    Comprehensive audit report (JSON and Markdown) with:
    - Pass/fail status for each check
    - Warnings and recommendations
    - Summary statistics

Usage:
    python -m audit.run_full_audit
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit.validate_raw_data import RawDataAudit
from audit.validate_riskfree_rates import RiskFreeRateAudit
from audit.validate_processed_data import ProcessedDataAudit
from audit.validate_financial_calculations import FinancialCalculationsAudit
from audit.validate_statistical_methodology import StatisticalMethodologyAudit
from audit.check_data_leakage import DataLeakageAudit
from audit.check_assumptions import AssumptionsAudit
from audit.validate_results import ResultsAudit
from audit.validate_interpretation import InterpretationAudit
from audit.validate_market_cap_analysis import MarketCapAnalysisAudit
from audit.validate_portfolio_optimization import PortfolioOptimizationAudit
from audit.validate_value_effects import ValueEffectsAudit
from audit.validate_portfolio_recommendation import PortfolioRecommendationAudit
from audit.validate_code_quality import CodeQualityAudit
from audit.validate_test_coverage import TestCoverageAudit
from audit.validate_integration import IntegrationAudit
from audit.validate_reproducibility import ReproducibilityAudit
from audit.validate_dependencies import DependencyAudit
from audit.validate_performance import PerformanceAudit
from audit.validate_data_lineage import DataLineageAudit
from audit.validate_cross_validation import CrossValidationAudit
from audit.validate_out_of_sample import OutOfSampleAudit
from audit.validate_model_stability import ModelStabilityAudit
from audit.validate_sensitivity import SensitivityAudit
from audit.validate_monte_carlo import MonteCarloAudit
from audit.validate_backtesting import BacktestingAudit
from audit.validate_documentation import DocumentationAudit
from audit.validate_error_handling import ErrorHandlingAudit
from audit.validate_edge_cases import EdgeCaseAudit
from audit.validate_stress_tests import StressTestAudit
from audit.validate_regression_tests import RegressionTestAudit
from audit.monitoring import SystemMonitor
from audit.validate_data_quality_comprehensive import ComprehensiveDataQualityAudit

from analysis.utils.config import PROJECT_ROOT, LOGS_DIR

# Setup logging
os.makedirs(LOGS_DIR, exist_ok=True)
log_file = os.path.join(LOGS_DIR, f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def generate_audit_report(all_results: dict, output_dir: str = None) -> str:
    """Generate comprehensive audit report."""
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "audit")
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, "audit_report.md")
    
    with open(report_file, 'w') as f:
        f.write("# Comprehensive CAPM Analysis Audit Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        
        # Count critical issues and warnings
        total_critical = 0
        total_warnings = 0
        
        for r in all_results.values():
            summary = r.get('summary', {})
            if isinstance(summary, dict):
                # Check if critical_issues is a count (int) or list
                crit = summary.get('critical_issues', 0)
                if isinstance(crit, int):
                    total_critical += crit
                elif isinstance(crit, list):
                    total_critical += len(crit)
                
                # Check if warnings is a count (int) or list
                warn = summary.get('warnings', 0)
                if isinstance(warn, int):
                    total_warnings += warn
                elif isinstance(warn, list):
                    total_warnings += len(warn)
            
            # Also check issues list directly
            issues = r.get('issues', [])
            if isinstance(issues, list):
                for issue in issues:
                    if isinstance(issue, dict) and issue.get('severity') == 'critical':
                        total_critical += 1
                    elif isinstance(issue, dict) and issue.get('severity') == 'warning':
                        total_warnings += 1
        
        f.write(f"- **Total Critical Issues:** {total_critical}\n")
        f.write(f"- **Total Warnings:** {total_warnings}\n")
        f.write(f"- **Total Phases:** {len(all_results)}\n\n")
        
        if total_critical == 0:
            f.write(" **No critical issues found.**\n\n")
        else:
            f.write(f" **{total_critical} critical issues require attention.**\n\n")
        
        # Phase-by-phase results
        f.write("## Phase-by-Phase Results\n\n")
        
        phase_names = {
            'phase1_raw_data': 'Phase 1.1: Raw Data Validation',
            'phase1_riskfree': 'Phase 1.2: Risk-Free Rate Validation',
            'phase1_processed': 'Phase 1.3: Processed Data Validation',
            'phase2_financial': 'Phase 2: Financial Calculations Audit',
            'phase3_statistical': 'Phase 3: Statistical Methodology Audit',
            'phase4_leakage': 'Phase 4.2: Data Leakage Checks',
            'phase4_assumptions': 'Phase 4.3: Assumption Violations',
            'phase5_results': 'Phase 5: Results Validation Audit',
            'phase6_interpretation': 'Phase 6: Interpretation & Reporting Audit',
            'phase7_market_cap': 'Phase 7: Market-Cap Weighted Betas Validation',
            'phase7_portfolio_opt': 'Phase 7: Portfolio Optimization Validation',
            'phase7_value_effects': 'Phase 7: Value Effects Validation',
            'phase7_recommendation': 'Phase 7: Portfolio Recommendation Validation',
            'phase8_code_quality': 'Phase 8.1: Code Quality & Static Analysis',
            'phase8_test_coverage': 'Phase 8.2: Test Coverage',
            'phase8_integration': 'Phase 8.3: Integration Testing',
            'phase9_performance': 'Phase 9.1: Performance Benchmarking',
            'phase9_reproducibility': 'Phase 9.2: Computational Reproducibility',
            'phase9_dependencies': 'Phase 9.4: Dependency Auditing',
            'phase10_data_lineage': 'Phase 10.1: Data Lineage & Provenance',
            'phase10_cross_validation': 'Phase 10.2: Cross-Validation Framework',
            'phase10_out_of_sample': 'Phase 10.3: Out-of-Sample Validation',
            'phase10_model_stability': 'Phase 10.4: Model Stability Analysis',
            'phase10_sensitivity': 'Phase 10.5: Sensitivity Analysis',
            'phase10_monte_carlo': 'Phase 10.6: Monte Carlo Validation',
            'phase10_backtesting': 'Phase 10.7: Backtesting Framework',
            'phase11_documentation': 'Phase 11.1: Documentation Completeness',
            'phase11_error_handling': 'Phase 11.2: Error Handling Validation',
            'phase11_edge_cases': 'Phase 11.3: Edge Case Testing',
            'phase11_stress_tests': 'Phase 11.4: Stress Testing',
            'phase12_regression_tests': 'Phase 12.1: Automated Regression Testing',
            'phase12_monitoring': 'Phase 12.3: Real-Time Monitoring'
        }
        
        for phase_key, phase_name in phase_names.items():
            if phase_key in all_results:
                f.write(f"### {phase_name}\n\n")
                phase_result = all_results[phase_key]
                
                summary = phase_result.get('summary', {})
                if isinstance(summary, dict):
                    f.write(f"- **Checks:** {summary.get('total_checks', 'N/A')}\n")
                    f.write(f"- **Issues:** {summary.get('total_issues', 'N/A')}\n")
                    f.write(f"- **Critical:** {summary.get('critical_issues', 'N/A')}\n")
                    f.write(f"- **Warnings:** {summary.get('warnings', 'N/A')}\n")
                    f.write(f"- **Passed:** {summary.get('passed_checks', 'N/A')}\n\n")
                
                # List critical issues
                issues = phase_result.get('issues', [])
                if isinstance(issues, list) and len(issues) > 0:
                    critical_issues = [i for i in issues if isinstance(i, dict) and i.get('severity') == 'critical']
                    if critical_issues:
                        f.write("**Critical Issues:**\n")
                        for issue in critical_issues[:5]:  # First 5
                            f.write(f"- {issue.get('message', str(issue))}\n")
                        f.write("\n")
        
        # Critical Issues Summary
        f.write("## Critical Issues Summary\n\n")
        
        all_critical = []
        for phase_key, phase_result in all_results.items():
            issues = phase_result.get('issues', [])
            if isinstance(issues, list):
                for issue in issues:
                    if isinstance(issue, dict) and issue.get('severity') == 'critical':
                        all_critical.append({
                            'phase': phase_names.get(phase_key, phase_key),
                            'message': issue.get('message', str(issue))
                        })
        
        if len(all_critical) == 0:
            f.write(" No critical issues found.\n\n")
        else:
            for issue in all_critical:
                f.write(f"- **{issue['phase']}:** {issue['message']}\n")
            f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        recommendations = [
            "Review all critical issues and address before finalizing results",
            "Document Fama-MacBeth assumption about full-sample beta estimation",
            "Verify risk-free rate conversion formula (compounding vs simple)",
            "Cross-validate risk-free rates with external sources",
            "Review date alignment across all data files",
            "Document any assumption violations in methodology section"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
        
        f.write("\n---\n\n")
        f.write(f"*Report generated by CAPM Analysis Audit System*\n")
    
    return report_file


def run_full_audit():
    """Run complete audit of CAPM analysis."""
    logger.info("="*70)
    logger.info("COMPREHENSIVE CAPM ANALYSIS AUDIT")
    logger.info("="*70)
    logger.info(f"Log file: {log_file}\n")
    
    all_results = {}
    
    # Phase 1: Data Quality & Integrity
    logger.info("\n" + "="*70)
    logger.info("STARTING PHASE 1: DATA QUALITY & INTEGRITY")
    logger.info("="*70)
    
    # Phase 1.1: Raw Data
    logger.info("\nRunning Phase 1.1: Raw Data Validation...")
    raw_data_audit = RawDataAudit()
    all_results['phase1_raw_data'] = raw_data_audit.run_all_checks()
    
    # Phase 1.2: Risk-Free Rates
    logger.info("\nRunning Phase 1.2: Risk-Free Rate Validation...")
    riskfree_audit = RiskFreeRateAudit()
    all_results['phase1_riskfree'] = riskfree_audit.run_all_checks()
    
    # Phase 1.3: Processed Data
    logger.info("\nRunning Phase 1.3: Processed Data Validation...")
    processed_audit = ProcessedDataAudit()
    all_results['phase1_processed'] = processed_audit.run_all_checks()
    
    # Phase 2: Financial Calculations
    logger.info("\n" + "="*70)
    logger.info("STARTING PHASE 2: FINANCIAL CALCULATIONS")
    logger.info("="*70)
    
    financial_audit = FinancialCalculationsAudit()
    all_results['phase2_financial'] = financial_audit.run_all_checks()
    
    # Phase 3: Statistical Methodology
    logger.info("\n" + "="*70)
    logger.info("STARTING PHASE 3: STATISTICAL METHODOLOGY")
    logger.info("="*70)
    
    statistical_audit = StatisticalMethodologyAudit()
    all_results['phase3_statistical'] = statistical_audit.run_all_checks()
    
    # Phase 4: Code Quality & Data Leakage
    logger.info("\n" + "="*70)
    logger.info("STARTING PHASE 4: CODE QUALITY & DATA LEAKAGE")
    logger.info("="*70)
    
    # Phase 4.2: Data Leakage
    logger.info("\nRunning Phase 4.2: Data Leakage Checks...")
    leakage_audit = DataLeakageAudit()
    all_results['phase4_leakage'] = leakage_audit.run_all_checks()
    
    # Phase 4.3: Assumptions
    logger.info("\nRunning Phase 4.3: Assumption Violations...")
    assumptions_audit = AssumptionsAudit()
    all_results['phase4_assumptions'] = assumptions_audit.run_all_checks()
    
    # Phase 5: Results Validation
    logger.info("\n" + "="*70)
    logger.info("STARTING PHASE 5: RESULTS VALIDATION")
    logger.info("="*70)
    
    results_audit = ResultsAudit()
    all_results['phase5_results'] = results_audit.run_all_checks()
    
    # Phase 6: Interpretation & Reporting
    logger.info("\n" + "="*70)
    logger.info("STARTING PHASE 6: INTERPRETATION & REPORTING")
    logger.info("="*70)
    
    interpretation_audit = InterpretationAudit()
    all_results['phase6_interpretation'] = interpretation_audit.run_all_checks()
    
    # Phase 8: Code Quality & Testing
    logger.info("\n" + "="*70)
    logger.info("STARTING PHASE 8: CODE QUALITY & TESTING")
    logger.info("="*70)
    
    # Phase 8.1: Code Quality
    logger.info("\nRunning Phase 8.1: Code Quality & Static Analysis...")
    code_quality_audit = CodeQualityAudit()
    all_results['phase8_code_quality'] = code_quality_audit.run_all_checks()
    
    # Phase 8.2: Test Coverage
    logger.info("\nRunning Phase 8.2: Test Coverage...")
    test_coverage_audit = TestCoverageAudit()
    all_results['phase8_test_coverage'] = test_coverage_audit.run_all_checks()
    
    # Phase 8.3: Integration Testing
    logger.info("\nRunning Phase 8.3: Integration Testing...")
    integration_audit = IntegrationAudit()
    all_results['phase8_integration'] = integration_audit.run_all_checks()
    
    # Phase 9: Performance & Reproducibility
    logger.info("\n" + "="*70)
    logger.info("STARTING PHASE 9: PERFORMANCE & REPRODUCIBILITY")
    logger.info("="*70)
    
    # Phase 9.2: Reproducibility
    logger.info("\nRunning Phase 9.2: Computational Reproducibility...")
    reproducibility_audit = ReproducibilityAudit()
    all_results['phase9_reproducibility'] = reproducibility_audit.run_all_checks()
    
    # Phase 9.1: Performance
    logger.info("\nRunning Phase 9.1: Performance Benchmarking...")
    performance_audit = PerformanceAudit()
    all_results['phase9_performance'] = performance_audit.run_all_checks()
    
    # Phase 9.4: Dependencies
    logger.info("\nRunning Phase 9.4: Dependency Auditing...")
    dependency_audit = DependencyAudit()
    all_results['phase9_dependencies'] = dependency_audit.run_all_checks()
    
    # Phase 10: Advanced Validation
    logger.info("\n" + "="*70)
    logger.info("STARTING PHASE 10: ADVANCED VALIDATION")
    logger.info("="*70)
    
    # Phase 10.1: Data Lineage
    logger.info("\nRunning Phase 10.1: Data Lineage & Provenance...")
    data_lineage_audit = DataLineageAudit()
    all_results['phase10_data_lineage'] = data_lineage_audit.run_all_checks()
    
    # Phase 10.2: Cross-Validation
    logger.info("\nRunning Phase 10.2: Cross-Validation Framework...")
    cv_audit = CrossValidationAudit()
    all_results['phase10_cross_validation'] = cv_audit.run_all_checks()
    
    # Phase 10.3: Out-of-Sample Validation
    logger.info("\nRunning Phase 10.3: Out-of-Sample Validation...")
    oos_audit = OutOfSampleAudit()
    all_results['phase10_out_of_sample'] = oos_audit.run_all_checks()
    
    # Phase 10.4: Model Stability
    logger.info("\nRunning Phase 10.4: Model Stability Analysis...")
    stability_audit = ModelStabilityAudit()
    all_results['phase10_model_stability'] = stability_audit.run_all_checks()
    
    # Phase 10.5: Sensitivity Analysis
    logger.info("\nRunning Phase 10.5: Sensitivity Analysis...")
    sensitivity_audit = SensitivityAudit()
    all_results['phase10_sensitivity'] = sensitivity_audit.run_all_checks()
    
    # Phase 10.6: Monte Carlo Validation
    logger.info("\nRunning Phase 10.6: Monte Carlo Validation...")
    mc_audit = MonteCarloAudit()
    all_results['phase10_monte_carlo'] = mc_audit.run_all_checks()
    
    # Phase 10.7: Backtesting Framework
    logger.info("\nRunning Phase 10.7: Backtesting Framework...")
    backtesting_audit = BacktestingAudit()
    all_results['phase10_backtesting'] = backtesting_audit.run_all_checks()
    
    # Phase 11: Documentation & Error Handling
    logger.info("\n" + "="*70)
    logger.info("STARTING PHASE 11: DOCUMENTATION & ERROR HANDLING")
    logger.info("="*70)
    
    # Phase 11.1: Documentation
    logger.info("\nRunning Phase 11.1: Documentation Completeness...")
    documentation_audit = DocumentationAudit()
    all_results['phase11_documentation'] = documentation_audit.run_all_checks()
    
    # Phase 11.2: Error Handling
    logger.info("\nRunning Phase 11.2: Error Handling Validation...")
    error_handling_audit = ErrorHandlingAudit()
    all_results['phase11_error_handling'] = error_handling_audit.run_all_checks()
    
    # Phase 11.3: Edge Cases
    logger.info("\nRunning Phase 11.3: Edge Case Testing...")
    edge_case_audit = EdgeCaseAudit()
    all_results['phase11_edge_cases'] = edge_case_audit.run_all_checks()
    
    # Phase 11.4: Stress Tests
    logger.info("\nRunning Phase 11.4: Stress Testing...")
    stress_test_audit = StressTestAudit()
    all_results['phase11_stress_tests'] = stress_test_audit.run_all_checks()
    
    # Phase 12: Continuous Monitoring
    logger.info("\n" + "="*70)
    logger.info("PHASE 12: CONTINUOUS MONITORING")
    logger.info("="*70)
    
    # Phase 12.1: Regression Testing
    logger.info("\nRunning Phase 12.1: Automated Regression Testing...")
    regression_audit = RegressionTestAudit()
    all_results['phase12_regression_tests'] = regression_audit.run_all_checks()
    
    # Phase 12.2: Comprehensive Data Quality Audit
    logger.info("\nRunning Phase 12.2: Comprehensive Data Quality Audit...")
    data_quality_audit = ComprehensiveDataQualityAudit()
    all_results['phase12_data_quality'] = data_quality_audit.run_all_checks()
    
    # Phase 12.3: Real-Time Monitoring
    logger.info("\nRunning Phase 12.3: Real-Time Monitoring...")
    monitor = SystemMonitor()
    monitoring_summary = monitor.get_monitoring_summary()
    all_results['phase12_monitoring'] = {
        'health': monitoring_summary['health'],
        'performance': monitoring_summary['performance'],
        'errors': monitoring_summary['errors'],
        'alerts': monitoring_summary['alerts'],
        'summary': {
            'overall_health': monitoring_summary['health']['overall_health'],
            'total_errors': monitoring_summary['errors']['total_errors'],
            'critical_alerts': len(monitoring_summary['alerts']['critical'])
        }
    }
    
    # Generate report
    logger.info("\n" + "="*70)
    logger.info("GENERATING AUDIT REPORT")
    logger.info("="*70)
    
    report_file = generate_audit_report(all_results)
    
    logger.info(f"\n Audit complete!")
    logger.info(f" Report saved to: {report_file}")
    logger.info(f" Log file: {log_file}")
    
    # Print summary
    total_critical = sum(
        len([i for i in r.get('issues', []) if isinstance(i, dict) and i.get('severity') == 'critical'])
        for r in all_results.values()
    )
    total_warnings = sum(
        len([i for i in r.get('issues', []) if isinstance(i, dict) and i.get('severity') == 'warning'])
        for r in all_results.values()
    )
    
    print("\n" + "="*70)
    print("AUDIT SUMMARY")
    print("="*70)
    print(f"Total Critical Issues: {total_critical}")
    print(f"Total Warnings: {total_warnings}")
    print(f"Report: {report_file}")
    print("="*70)
    
    return all_results, report_file


if __name__ == "__main__":
    results, report = run_full_audit()

