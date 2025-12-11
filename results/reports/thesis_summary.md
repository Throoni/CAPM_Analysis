# CAPM Analysis - Results Summary

## Quick Reference Guide

### Key Findings at a Glance

**Time-Series CAPM:**
- Valid stocks: 245
- Average beta: 0.917
- Average R²: 0.235
- Average alpha: -0.099%

**Cross-Sectional CAPM (Fama-MacBeth):**
- Market price of risk (γ₁): -0.5747 (t=-0.880, NOT significant)
- Intercept (γ₀): 1.3167% (t=4.112, HIGHLY significant)
- **Conclusion: CAPM REJECTED** - Beta does not explain cross-sectional variation in returns

**Robustness:**
- Subperiod A (2021-2022): γ₁ = -0.7727 (not significant)
- Subperiod B (2023-2025): γ₁ = -0.4389 (not significant)
- Results hold across subperiods, countries, and sample specifications

**Beta-Sorted Portfolios:**
- Portfolio 1 (lowest beta): Return = 0.80%
- Portfolio 5 (highest beta): Return = 0.55%
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


**France:**
- γ₁ = -1.4224 (t=-1.401)
- ❌ Not significant

**Germany:**
- γ₁ = -0.0754 (t=-0.104)
- ❌ Not significant

**Italy:**
- γ₁ = 0.7178 (t=0.784)
- ❌ Not significant

**Spain:**
- γ₁ = -0.5489 (t=-0.756)
- ❌ Not significant

**Sweden:**
- γ₁ = -1.1308 (t=-1.395)
- ❌ Not significant

**Switzerland:**
- γ₁ = 0.3334 (t=0.414)
- ❌ Not significant

**UnitedKingdom:**
- γ₁ = -0.6090 (t=-0.782)
- ❌ Not significant


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

1. **Time-Series:** CAPM has moderate explanatory power (R² ≈ 0.24)

2. **Cross-Section:** CAPM is REJECTED - Beta does not explain cross-sectional variation in returns

3. **Robustness:** Results are robust across subperiods, countries, and sample specifications

4. **Implications:** Multi-factor models (Fama-French, Carhart) are necessary to explain returns

---

**Generated:** 2025-12-11 22:03:41
