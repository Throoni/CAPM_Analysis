# Final Results Data Index

## Complete List of All Files in Submission Package

---

##  Data Files (70 files)

### Core Results

| File | Description | Rows/Content |
|------|-------------|--------------|
| `data/capm_results.csv` | Full CAPM regression results | 245 stocks |
| `data/fama_macbeth_summary.csv` | Cross-sectional test statistics | Key FM stats |
| `data/fama_macbeth_monthly_coefficients.csv` | Monthly γ₀, γ₁ estimates | 59 months |
| `data/beta_quintile_portfolios.csv` | Beta-sorted portfolio analysis | 5 quintiles |

### Portfolio Optimization Results

| File | Description |
|------|-------------|
| `data/portfolio_results_long_only.csv` | Long-only portfolio metrics |
| `data/portfolio_results_short_selling_full_sample.csv` | Full sample short-selling (N=245) |
| `data/portfolio_results_short_selling_n15.csv` | Constrained sample short-selling (N=15) |
| `data/efficient_frontier_long_only.csv` | Long-only frontier points (41 points) |
| `data/efficient_frontier_short_selling.csv` | Unconstrained frontier points (15 points) |
| `data/efficient_frontier_short_selling_n15.csv` | N=15 constrained frontier |
| `data/portfolio_optimization_results.csv` | Combined portfolio results |
| `data/portfolio_optimization_realistic_short.csv` | Realistic short-selling with costs |
| `data/portfolio_optimization_top15_short_selling.csv` | Top 15 by market cap |

### Country-Level Analysis

| File | Description |
|------|-------------|
| `data/country_level/country_beta_statistics.csv` | Beta mean, median, std by country |
| `data/country_level/cross_sectional_country_summary.csv` | FM statistics by country |
| `data/country_level/capm_by_country.csv` | CAPM results by country |
| `data/country_level/capm_clean_sample.csv` | Clean sample CAPM results |
| `data/country_level/fm_by_country.csv` | Fama-MacBeth by country |

### Robustness Analysis

| File | Description |
|------|-------------|
| `data/robustness/fm_subperiod_A.csv` | Period A (2021-2022) FM results |
| `data/robustness/fm_subperiod_B.csv` | Period B (2023-2025) FM results |
| `data/robustness/fm_subperiod_comparison.csv` | Subperiod comparison |
| `data/robustness/beta_sorted_portfolios.csv` | Beta quintile analysis |
| `data/robustness/diversification_benefits.csv` | Diversification metrics |
| `data/robustness/robustness_summary.csv` | Overall robustness summary |

### Value Effects

| File | Description |
|------|-------------|
| `data/value_effects_portfolios.csv` | Value/Growth portfolio returns |
| `data/value_effects_test_results.csv` | Value premium test statistics |

### Raw Data

| Folder | Contents |
|--------|----------|
| `data/raw/stock_prices/` | Stock prices for 7 countries (CSV) |
| `data/raw/riskfree_rates/` | Risk-free rate data (German Bund) |
| `data/raw/index_prices/` | MSCI index prices |
| `data/raw/exchange_rates/` | Currency exchange rates |

### Processed Data

| File | Description |
|------|-------------|
| `data/processed/returns_panel.csv` | Full returns panel (14,692 obs) |

---

##  Figures (29 files)

### Main Figures

| Figure | Description |
|--------|-------------|
| `figures/efficient_frontier_no_shortselling.png` | Long-only efficient frontier |
| `figures/efficient_frontier_with_shortselling.png` | Unconstrained efficient frontier |
| `figures/efficient_frontier_short_selling_n15.png` | N=15 constrained frontier |
| `figures/beta_distribution_by_country.png` | Beta histograms by country |
| `figures/alpha_distribution_by_country.png` | Alpha histograms by country |
| `figures/r2_distribution_by_country.png` | R² histograms by country |
| `figures/gamma1_timeseries.png` | Market risk premium over time |
| `figures/fama_macbeth_by_country.png` | Country-level FM results |
| `figures/value_effect_analysis.png` | Value portfolio analysis |
| `figures/beta_sorted_returns.png` | Returns by beta quintile |

### Country-Level Figures

| Figure | Description |
|--------|-------------|
| `figures/country_level/average_beta_by_country.png` | Mean beta by country |
| `figures/country_level/beta_boxplot_by_country.png` | Beta box plots |
| `figures/country_level/beta_vs_return_scatter.png` | SML visualization |
| `figures/country_level/beta_vs_r2_scatter.png` | Beta vs R² relationship |
| `figures/country_level/fm_gamma1_by_country.png` | γ₁ by country |
| `figures/country_level/fm_subperiod_comparison.png` | Subperiod comparison |
| `figures/country_level/gamma1_histogram.png` | γ₁ distribution |
| `figures/country_level/top_bottom_beta_stocks.png` | Extreme beta stocks |

### Portfolio Figures

| Figure | Description |
|--------|-------------|
| `figures/efficient_frontier.png` | Combined efficient frontier |
| `figures/efficient_frontier_constrained.png` | Constrained only |
| `figures/efficient_frontier_unconstrained.png` | Unconstrained only |
| `figures/efficient_frontier_realistic_short.png` | With transaction costs |
| `figures/efficient_frontier_top15_short_selling.png` | Top 15 stocks |

---

##  Documentation (5 files)

| Document | Description | Lines |
|----------|-------------|-------|
| `EXECUTIVE_SUMMARY.md` | 1-page key findings | 95 |
| `README.md` | Navigation guide | ~80 |
| `DATA_INDEX.md` | This file | ~200 |
| `methodology/METHODOLOGY.md` | Complete methodology | 212 |
| `reports/EMPIRICAL_RESULTS.md` | Detailed results | 309 |
| `reports/INVESTMENT_RECOMMENDATIONS.md` | US investor guide | 267 |

---

## Key Statistics Summary

| Metric | Value |
|--------|-------|
| **Total Stocks** | 245 |
| **Countries** | 7 |
| **Sample Period** | Jan 2021 – Nov 2025 |
| **Months** | 58-59 |
| **Total Observations** | 14,692 |
| **Risk-Free Rate** | German 3-month Bund |
| **Market Proxy** | MSCI Europe |

---

## Quick Reference: Key Tables

### Table 1: CAPM Time-Series Results
→ `data/capm_results.csv`

### Table 2: Fama-MacBeth Cross-Sectional Results
→ `data/fama_macbeth_summary.csv`
→ `data/fama_macbeth_monthly_coefficients.csv`

### Table 3: Country-Level Summary
→ `data/country_level/cross_sectional_country_summary.csv`
→ `data/country_level/country_beta_statistics.csv`

### Table 4: Portfolio Optimization
→ `data/portfolio_results_long_only.csv`
→ `data/portfolio_results_short_selling_full_sample.csv`
→ `data/portfolio_results_short_selling_n15.csv`

### Table 5: Beta-Sorted Portfolios
→ `data/beta_quintile_portfolios.csv`
→ `data/robustness/beta_sorted_portfolios.csv`

### Table 6: Robustness - Subperiod Analysis
→ `data/robustness/fm_subperiod_comparison.csv`

### Table 7: Value Effects
→ `data/value_effects_portfolios.csv`
→ `data/value_effects_test_results.csv`

---

*Index generated: December 2025*

