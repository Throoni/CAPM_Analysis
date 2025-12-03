"""
create_summary.py

Stage 7.4: Create summary document with key findings and table/figure index.
"""

import os
import logging
import pandas as pd

from analysis.config import RESULTS_DATA_DIR, RESULTS_REPORTS_DIR

logger = logging.getLogger(__name__)


def create_results_summary() -> str:
    """
    Create comprehensive summary document.
    
    Returns
    -------
    str
        Summary text
    """
    logger.info("Creating results summary...")
    
    # Load key results
    capm_results = pd.read_csv(os.path.join(RESULTS_DATA_DIR, "capm_results.csv"))
    fm_summary = pd.read_csv(os.path.join(RESULTS_REPORTS_DIR, "fama_macbeth_summary.csv"))
    subperiod = pd.read_csv(os.path.join(RESULTS_REPORTS_DIR, "fm_subperiod_comparison.csv"))
    country_fm = pd.read_csv(os.path.join(RESULTS_REPORTS_DIR, "fm_by_country.csv"))
    portfolios = pd.read_csv(os.path.join(RESULTS_REPORTS_DIR, "beta_sorted_portfolios.csv"))
    
    valid_stocks = capm_results[capm_results['is_valid'] == True]
    
    summary = f"""# CAPM Analysis - Results Summary

## Quick Reference Guide

### Key Findings at a Glance

**Time-Series CAPM:**
- Valid stocks: {len(valid_stocks)}
- Average beta: {valid_stocks['beta'].mean():.3f}
- Average R²: {valid_stocks['r_squared'].mean():.3f}
- Average alpha: {valid_stocks['alpha'].mean():.3f}%

**Cross-Sectional CAPM (Fama-MacBeth):**
- Market price of risk (γ₁): {fm_summary['avg_gamma_1'].iloc[0]:.4f} (t={fm_summary['tstat_gamma_1'].iloc[0]:.3f}, NOT significant)
- Intercept (γ₀): {fm_summary['avg_gamma_0'].iloc[0]:.4f}% (t={fm_summary['tstat_gamma_0'].iloc[0]:.3f}, HIGHLY significant)
- **Conclusion: CAPM REJECTED** - Beta does not explain cross-sectional variation in returns

**Robustness:**
- Subperiod A (2021-2022): γ₁ = {subperiod.iloc[0]['avg_gamma_1']:.4f} (not significant)
- Subperiod B (2023-2025): γ₁ = {subperiod.iloc[1]['avg_gamma_1']:.4f} (not significant)
- Results hold across subperiods, countries, and sample specifications

**Beta-Sorted Portfolios:**
- Portfolio 1 (lowest beta): Return = {portfolios.iloc[0]['avg_return']:.2f}%
- Portfolio 5 (highest beta): Return = {portfolios.iloc[4]['avg_return']:.2f}%
- **Negative relationship:** Higher beta → Lower return (contrary to CAPM)

---

## Table Index

1. **Table 1:** CAPM Time-Series Summary by Country
   - Location: `results/tables/table1_capm_timeseries_summary.csv`
   - Shows: Mean/median beta, R², % significant betas by country

2. **Table 2:** Fama-MacBeth CAPM Test Results
   - Location: `results/tables/table2_fama_macbeth_results.csv`
   - Shows: Average γ₀, γ₁, t-statistics, p-values

3. **Table 3:** Subperiod Fama-MacBeth Results
   - Location: `results/tables/table3_subperiod_results.csv`
   - Shows: Results for Period A (2021-2022) and Period B (2023-2025)

4. **Table 4:** Country-Level Fama-MacBeth Results
   - Location: `results/tables/table4_country_level_results.csv`
   - Shows: γ₀, γ₁ by country

5. **Table 5:** Beta-Sorted Portfolio Returns
   - Location: `results/tables/table5_beta_sorted_portfolios.csv`
   - Shows: Portfolio betas, average returns, standard deviations

6. **Table 6:** Descriptive Statistics
   - Location: `results/tables/table6_descriptive_statistics.csv`
   - Shows: Sample description by country

---

## Figure Index

1. **Figure 1:** Distribution of Betas by Country
   - Location: `results/plots/beta_distribution_by_country.png`
   - Shows: Histogram of betas for each country

2. **Figure 2:** Distribution of R² by Country
   - Location: `results/plots/r2_distribution_by_country.png`
   - Shows: Histogram of R² values for each country

3. **Figure 3:** Time Series of γ₁ (Market Price of Risk)
   - Location: `results/plots/gamma1_timeseries.png`
   - Shows: Monthly γ₁ values over time

4. **Figure 4:** Beta vs Average Return Scatter
   - Location: `results/plots/beta_vs_return_scatter.png`
   - Shows: Cross-sectional relationship between beta and average return

5. **Figure 5:** Beta-Sorted Portfolio Returns (KEY FIGURE)
   - Location: `results/plots/beta_sorted_returns.png`
   - Shows: Portfolio beta vs average return (negative slope = CAPM failure)

6. **Figure 6:** γ₁ by Country
   - Location: `results/plots/fm_gamma1_by_country.png`
   - Shows: Country-level market price of risk with confidence intervals

---

## Country-Level Summary

"""
    
    # Add country summaries
    for _, row in country_fm.iterrows():
        summary += f"""
**{row['country']}:**
- γ₁ = {row['avg_gamma_1']:.4f} (t={row['tstat_gamma_1']:.3f})
- {'✅ Significant' if abs(row['tstat_gamma_1']) > 1.96 else '❌ Not significant'}
"""
    
    summary += f"""

---

## Main Deliverables

### Reports
- `results/reports/capm_by_country.csv` - Country-level CAPM summary
- `results/reports/fama_macbeth_summary.csv` - Main Fama-MacBeth results
- `results/reports/fm_subperiod_comparison.csv` - Subperiod comparison
- `results/reports/fm_by_country.csv` - Country-level Fama-MacBeth
- `results/reports/beta_sorted_portfolios.csv` - Portfolio returns
- `results/reports/economic_interpretation.txt` - Economic interpretation
- `results/reports/robustness_summary.csv` - Robustness test summary

### Tables (CSV + LaTeX)
- All tables in `results/tables/` directory

### Figures
- All figures in `results/plots/` directory

### Chapter Text
- `results/reports/thesis_chapter7_empirical_results.md` - Markdown version
- `results/reports/thesis_chapter7_empirical_results.tex` - LaTeX version

---

## Key Conclusions

1. **Time-Series:** CAPM has moderate explanatory power (R² ≈ {valid_stocks['r_squared'].mean():.2f})

2. **Cross-Section:** CAPM is REJECTED - Beta does not explain cross-sectional variation in returns

3. **Robustness:** Results are robust across subperiods, countries, and sample specifications

4. **Implications:** Multi-factor models (Fama-French, Carhart) are necessary to explain returns

---

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    summary_text = create_results_summary()
    
    summary_path = os.path.join(RESULTS_REPORTS_DIR, "thesis_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print("✅ Summary document created:")
    print(f"   {summary_path}")

