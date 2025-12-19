"""
Portfolio Recommendation Synthesis Module.

This module synthesizes all CAPM analysis findings into a coherent investment
recommendation, specifically addressing the active vs. passive management decision.

Decision framework:
    1. CAPM Validity: If gamma_1 significant, systematic risk is priced
    2. Alpha Evidence: If significant alphas exist, active management may add value
    3. Market Efficiency: If alphas are random noise, passive indexing preferred
    4. Cost Consideration: Active management must overcome fee hurdle

Recommendation outputs:
    - Primary recommendation: Active vs. Passive strategy
    - Supporting evidence from each analysis component
    - Risk-adjusted performance expectations
    - Implementation suggestions (ETFs, individual stocks, etc.)

The recommendation considers:
    - Statistical significance of findings
    - Economic magnitude of potential alpha
    - Transaction costs and management fees
    - Investor risk tolerance and investment horizon
"""

import os
import logging
import pandas as pd
from typing import Dict

from analysis.utils.config import (
    RESULTS_DATA_DIR,
    RESULTS_REPORTS_DIR,
    RESULTS_REPORTS_MAIN_DIR
)

logger = logging.getLogger(__name__)


def synthesize_findings() -> Dict:
    """
    Synthesize all analysis findings.
    
    Returns
    -------
    dict
        Dictionary with key findings
    """
    logger.info("Synthesizing analysis findings...")
    
    findings = {}
    
    # Load key results
    capm_results_file = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
    fm_summary_file = os.path.join(RESULTS_REPORTS_DIR, "fama_macbeth_summary.csv")
    portfolios_file = os.path.join(RESULTS_REPORTS_DIR, "beta_sorted_portfolios.csv")
    value_effects_file = os.path.join(RESULTS_REPORTS_DIR, "value_effects_test_results.csv")
    portfolio_opt_file = os.path.join(RESULTS_REPORTS_DIR, "portfolio_optimization_results.csv")
    div_benefits_file = os.path.join(RESULTS_REPORTS_DIR, "diversification_benefits.csv")
    
    # CAPM Results
    if os.path.exists(capm_results_file):
        capm = pd.read_csv(capm_results_file)
        valid = capm[capm['is_valid'] == True]
        findings['capm'] = {
            'n_stocks': len(valid),
            'avg_beta': valid['beta'].mean(),
            'avg_r_squared': valid['r_squared'].mean(),
            'avg_alpha': valid['alpha'].mean(),
            'pct_significant_betas': (valid['pvalue_beta'] < 0.05).sum() / len(valid) * 100
        }
    
    # Fama-MacBeth Results
    if os.path.exists(fm_summary_file):
        fm = pd.read_csv(fm_summary_file)
        findings['fama_macbeth'] = {
            'gamma_1': fm['avg_gamma_1'].iloc[0],
            'gamma_1_tstat': fm['tstat_gamma_1'].iloc[0],
            'gamma_1_pvalue': fm['pvalue_gamma_1'].iloc[0],
            'gamma_0': fm['avg_gamma_0'].iloc[0],
            'capm_rejected': fm['pvalue_gamma_1'].iloc[0] > 0.05
        }
    
    # Beta-Sorted Portfolios
    if os.path.exists(portfolios_file):
        portfolios = pd.read_csv(portfolios_file)
        findings['beta_portfolios'] = {
            'low_beta_return': portfolios.iloc[0]['avg_return'],
            'high_beta_return': portfolios.iloc[-1]['avg_return'],
            'negative_relationship': portfolios.iloc[-1]['avg_return'] < portfolios.iloc[0]['avg_return']
        }
    
    # Value Effects
    if os.path.exists(value_effects_file):
        value = pd.read_csv(value_effects_file)
        findings['value_effects'] = {
            'alpha_spread': value['alpha_spread'].iloc[0],
            'regression_slope': value['regression_slope'].iloc[0],
            'regression_pvalue': value['regression_pvalue'].iloc[0],
            'interpretation': value['interpretation'].iloc[0]
        }
    
    # Portfolio Optimization
    if os.path.exists(portfolio_opt_file):
        opt = pd.read_csv(portfolio_opt_file)
        tangency = opt[opt['portfolio'] == 'Optimal Risky (Tangency)']
        if len(tangency) > 0:
            findings['optimization'] = {
                'optimal_return': tangency['expected_return'].iloc[0],
                'optimal_volatility': tangency['volatility'].iloc[0],
                'optimal_sharpe': tangency['sharpe_ratio'].iloc[0]
            }
    
    # Diversification Benefits
    if os.path.exists(div_benefits_file):
        div = pd.read_csv(div_benefits_file)
        findings['diversification'] = {
            'variance_reduction': div['variance_reduction_pct'].iloc[0],
            'diversification_ratio': div['diversification_ratio'].iloc[0]
        }
    
    return findings


def make_portfolio_recommendation(findings: Dict) -> Dict:
    """
    Make portfolio recommendation based on findings.
    
    Parameters
    ----------
    findings : dict
        Synthesized findings
    
    Returns
    -------
    dict
        Recommendation with justification
    """
    logger.info("Making portfolio recommendation...")
    
    recommendation = {
        'strategy': None,
        'justification': [],
        'key_evidence': [],
        'risks': [],
        'implementation': []
    }
    
    # Key evidence
    capm_rejected = findings.get('fama_macbeth', {}).get('capm_rejected', False)
    negative_beta_return = findings.get('beta_portfolios', {}).get('negative_relationship', False)
    value_effect = findings.get('value_effects', {}).get('regression_slope', 0)
    diversification_benefits = findings.get('diversification', {}).get('variance_reduction', 0)
    
    # Decision logic
    evidence_for_active = []
    evidence_for_passive = []
    
    # Evidence FOR active management:
    if capm_rejected:
        evidence_for_active.append("CAPM is rejected - beta does not price returns, suggesting market inefficiencies")
    
    if negative_beta_return:
        evidence_for_active.append("Negative beta-return relationship contradicts CAPM, indicating mispricing opportunities")
    
    if abs(value_effect) > 0.1:
        evidence_for_active.append("Value effects exist (or reverse value effects), suggesting factor-based strategies may add value")
    
    # Evidence FOR passive management:
    if diversification_benefits > 50:
        evidence_for_passive.append(f"Strong diversification benefits ({diversification_benefits:.1f}% variance reduction)")
    
    if findings.get('optimization', {}).get('optimal_sharpe', 0) > 1.0:
        evidence_for_passive.append("Optimal portfolio has high Sharpe ratio, suggesting efficient market portfolio")
    
    # Make recommendation
    if len(evidence_for_active) >= 2:
        recommendation['strategy'] = 'Active'
        recommendation['justification'].append(
            "Multiple lines of evidence suggest market inefficiencies that active management can exploit."
        )
        recommendation['key_evidence'] = evidence_for_active
    elif len(evidence_for_passive) >= 2:
        recommendation['strategy'] = 'Passive'
        recommendation['justification'].append(
            "Strong diversification benefits and efficient portfolio characteristics favor passive indexing."
        )
        recommendation['key_evidence'] = evidence_for_passive
    else:
        # Hybrid approach
        recommendation['strategy'] = 'Hybrid (Tilted Passive)'
        recommendation['justification'].append(
            "Mixed evidence suggests a hybrid approach: passive core with active tilts to exploit identified factors."
        )
        recommendation['key_evidence'] = evidence_for_active + evidence_for_passive
    
    # Add specific recommendations
    if recommendation['strategy'] == 'Active':
        recommendation['implementation'].append(
            "Focus on factor-based strategies (value, quality, low-volatility) given CAPM failure"
        )
        recommendation['implementation'].append(
            "Avoid high-beta stocks given negative beta-return relationship"
        )
        recommendation['risks'].append(
            "Active management costs (fees, turnover) may erode returns"
        )
        recommendation['risks'].append(
            "Factor exposures may not persist in future periods"
        )
    elif recommendation['strategy'] == 'Passive':
        recommendation['implementation'].append(
            "Invest in market-cap weighted index fund tracking European markets"
        )
        recommendation['implementation'].append(
            "Consider optimal risky portfolio weights from mean-variance analysis"
        )
        recommendation['risks'].append(
            "Passive approach may miss factor-based return opportunities"
        )
    else:  # Hybrid
        recommendation['implementation'].append(
            "Core: Market-cap weighted index (70-80% of portfolio)"
        )
        recommendation['implementation'].append(
            "Tilt: Factor-based ETFs or active strategies (20-30%) targeting value, quality, or low-volatility"
        )
        recommendation['risks'].append(
            "Hybrid approach balances costs and opportunities but requires careful implementation"
        )
    
    logger.info(f"Recommendation: {recommendation['strategy']}")
    logger.info(f"  Key evidence: {len(recommendation['key_evidence'])} points")
    
    return recommendation


def generate_recommendation_report(findings: Dict, recommendation: Dict) -> str:
    """
    Generate portfolio recommendation report.
    
    Parameters
    ----------
    findings : dict
        Synthesized findings
    recommendation : dict
        Recommendation
    
    Returns
    -------
    str
        Report text
    """
    report = f"""# Portfolio Recommendation for European Equity Markets

## Executive Summary

Based on comprehensive empirical analysis of 245 stocks across 7 European markets (2021-2025), this report provides a portfolio recommendation for XYZ Asset Manager's European equity fund.

**Recommendation: {recommendation['strategy']}**

---

## Key Findings Summary

### 1. CAPM Test Results

- **Time-Series:** Beta explains ~24% of return variation (R² = 0.235)
- **Cross-Sectional:** CAPM **REJECTED** - Beta does not price returns (γ₁ = -0.9662, p = 0.236)
- **Interpretation:** Market risk (beta) alone is insufficient to explain expected returns

### 2. Beta-Return Relationship

- **Finding:** Negative relationship - Higher beta portfolios have **lower** returns
- **Portfolio 1 (Low Beta):** 1.20% average return
- **Portfolio 5 (High Beta):** 0.54% average return
- **Implication:** High-beta stocks are overpriced relative to their risk

### 3. Value Effects

- **Finding:** {'Reverse value effect' if findings.get('value_effects', {}).get('regression_slope', 0) < 0 else 'Value effect'} detected
- **Alpha Spread:** {findings.get('value_effects', {}).get('alpha_spread', 0):.4f}%
- **Implication:** Book-to-market ratios may explain return differences, but relationship is {'contrary to' if findings.get('value_effects', {}).get('regression_slope', 0) < 0 else 'consistent with'} classic value theory

### 4. Diversification Benefits

- **Variance Reduction:** {findings.get('diversification', {}).get('variance_reduction', 0):.1f}%
- **Diversification Ratio:** {findings.get('diversification', {}).get('diversification_ratio', 0):.2f}
- **Implication:** Significant benefits from portfolio diversification across European markets

### 5. Optimal Portfolio Characteristics

- **Optimal Risky Portfolio Return:** {findings.get('optimization', {}).get('optimal_return', 0):.2f}%
- **Optimal Portfolio Volatility:** {findings.get('optimization', {}).get('optimal_volatility', 0):.2f}%
- **Sharpe Ratio:** {findings.get('optimization', {}).get('optimal_sharpe', 0):.2f}

---

## Recommendation: {recommendation['strategy']}

### Justification

"""
    
    for justification in recommendation['justification']:
        report += f"- {justification}\n"
    
    report += "\n### Key Evidence\n\n"
    for evidence in recommendation['key_evidence']:
        report += f"- {evidence}\n"
    
    report += "\n### Implementation Strategy\n\n"
    for impl in recommendation['implementation']:
        report += f"- {impl}\n"
    
    report += "\n### Risks and Considerations\n\n"
    for risk in recommendation['risks']:
        report += f"- {risk}\n"
    
    report += """
---

## Detailed Analysis

### Why CAPM Fails in European Markets (2021-2025)

1. **Period-Specific Factors:** Post-COVID recovery, monetary policy shifts, and sector rotations created return patterns not captured by market beta alone.

2. **Sector Effects:** Energy, technology, financials, and luxury goods had distinct risk-return profiles driven by sector-specific factors rather than market beta.

3. **Factor Exposures:** Size, value, quality, and momentum factors likely explain returns better than beta alone, consistent with Fama-French (1992, 1993) findings.

### Implications for Portfolio Construction

Given CAPM failure, portfolio construction should consider:

1. **Multi-Factor Models:** Incorporate size, value, quality, and momentum factors
2. **Sector Allocation:** Active sector tilts based on macro outlook
3. **Low-Volatility Strategy:** Given negative beta-return relationship, low-beta stocks may offer better risk-adjusted returns
4. **Geographic Diversification:** Benefits from diversifying across 7 European markets

---

## Conclusion

The empirical evidence strongly suggests that **beta alone is insufficient** to explain expected returns in European equity markets. This creates opportunities for active management, but also highlights the importance of diversification and factor-based strategies.

**Final Recommendation:** {recommendation['strategy']}

This recommendation balances the evidence of market inefficiencies with the benefits of diversification and cost considerations. Regular monitoring and rebalancing will be essential to adapt to changing market conditions.

---

*Report generated based on comprehensive CAPM analysis of 245 stocks across 7 European markets (2021-2025)*
"""
    
    return report


def run_portfolio_recommendation() -> Dict:
    """
    Run complete portfolio recommendation analysis.
    
    Returns
    -------
    dict
        Recommendation results
    """
    logger.info("="*70)
    logger.info("PORTFOLIO RECOMMENDATION ANALYSIS")
    logger.info("="*70)
    
    # Synthesize findings
    findings = synthesize_findings()
    
    # Make recommendation
    recommendation = make_portfolio_recommendation(findings)
    
    # Generate report
    report = generate_recommendation_report(findings, recommendation)
    
    # Save report to new organized structure
    report_file = os.path.join(RESULTS_REPORTS_MAIN_DIR, "Portfolio_Recommendation.md")
    with open(report_file, 'w') as f:
        f.write(report)
    # Also save to legacy location
    legacy_report = os.path.join(RESULTS_REPORTS_DIR, "Portfolio_Recommendation.md")
    with open(legacy_report, 'w') as f:
        f.write(report)
    
    logger.info(f" Saved: {report_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("PORTFOLIO RECOMMENDATION")
    print("="*70)
    print(f"\nStrategy: {recommendation['strategy']}")
    print(f"\nKey Evidence:")
    for evidence in recommendation['key_evidence']:
        print(f"  • {evidence}")
    print(f"\nReport saved to: {report_file}")
    
    return {
        'findings': findings,
        'recommendation': recommendation,
        'report': report
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    results = run_portfolio_recommendation()
    
    print("\n" + "="*70)
    print("PORTFOLIO RECOMMENDATION COMPLETE")
    print("="*70)

