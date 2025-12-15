# Final Results Package

## CAPM Analysis of European Equity Markets
**Submission Package | December 2025**

---

## Quick Start

1. **Start Here**: Read `EXECUTIVE_SUMMARY.md` for key findings
2. **Methodology**: See `methodology/METHODOLOGY.md` for theoretical background
3. **Full Results**: Review `reports/EMPIRICAL_RESULTS.md` for detailed analysis

---

## Contents

```
final_results/
├── EXECUTIVE_SUMMARY.md              # 1-page summary of key findings
├── README.md                         # This file
│
├── methodology/
│   └── METHODOLOGY.md                # Complete methodology paper
│
├── reports/
│   ├── EMPIRICAL_RESULTS.md          # Detailed empirical results
│   └── INVESTMENT_RECOMMENDATIONS.md # Investment guide for US investors
│
├── figures/                          # Main figures
│   ├── efficient_frontier_no_shortselling.png
│   ├── efficient_frontier_with_shortselling.png
│   ├── efficient_frontier_short_selling_n15.png   # n=15 constrained sample
│   ├── beta_distribution_by_country.png
│   ├── alpha_distribution_by_country.png
│   ├── r2_distribution_by_country.png
│   ├── gamma1_timeseries.png
│   ├── fama_macbeth_by_country.png
│   ├── value_effect_analysis.png
│   ├── beta_sorted_returns.png
│   │
│   └── country_level/                # Country-level comparison figures
│       ├── average_beta_by_country.png
│       ├── beta_boxplot_by_country.png
│       ├── beta_vs_r2_scatter.png
│       ├── beta_vs_return_scatter.png
│       ├── fm_gamma1_by_country.png
│       └── fm_subperiod_comparison.png
│
├── tables/                           # Publication-ready tables
│   ├── table1_capm_timeseries_summary.csv/.tex
│   ├── table2_fama_macbeth_results.csv/.tex
│   ├── table3_subperiod_results.csv/.tex
│   ├── table4_country_level_results.csv/.tex
│   ├── table5_beta_sorted_portfolios.csv/.tex
│   ├── table6_descriptive_statistics.csv/.tex
│   └── table7_market_cap_weighted_betas.csv
│
├── data/
│   ├── capm_results.csv                              # Full CAPM regression (245 stocks)
│   ├── fama_macbeth_summary.csv                      # FM test statistics
│   ├── fama_macbeth_monthly_coefficients.csv         # Monthly γ estimates
│   ├── portfolio_results_long_only.csv               # Long-only portfolios
│   ├── portfolio_results_short_selling_full_sample.csv  # Short-selling (full sample)
│   ├── portfolio_results_short_selling_n15.csv       # Short-selling (n=15 constrained)
│   ├── efficient_frontier_long_only.csv
│   ├── efficient_frontier_short_selling.csv
│   ├── efficient_frontier_short_selling_n15.csv
│   ├── value_effects_portfolios.csv
│   ├── value_effects_test_results.csv
│   │
│   ├── country_level/                # Country-level analysis
│   │   ├── country_beta_summary.csv          # Mean/median betas by country
│   │   ├── fama_macbeth_country_summary.csv  # FM results by country
│   │   ├── capm_by_country.csv
│   │   ├── capm_clean_sample.csv
│   │   ├── capm_extremes.csv
│   │   ├── fm_by_country.csv
│   │   ├── fm_subperiod_comparison.csv
│   │   ├── fm_subperiod_A.csv
│   │   ├── fm_subperiod_B.csv
│   │   ├── fama_macbeth_beta_returns.csv
│   │   └── fama_macbeth_clean_sample.csv
│   │
│   ├── robustness/                   # Robustness checks
│   │   ├── robustness_summary.csv
│   │   ├── diversification_benefits.csv
│   │   ├── beta_sorted_portfolios.csv
│   │   ├── table5_beta_sorted_portfolios.csv
│   │   ├── portfolio_optimization_realistic_short.csv
│   │   └── efficient_frontier_unconstrained.csv
│   │
│   ├── processed/                    # Processed analysis data
│   │   ├── returns_panel.csv                 # Main returns panel
│   │   └── returns_panel_with_real_rates.csv
│   │
│   └── raw/                          # Raw source data
│       ├── stock_prices/             # Stock price data (7 countries)
│       │   ├── prices_stocks_Germany.csv
│       │   ├── prices_stocks_France.csv
│       │   ├── prices_stocks_Italy.csv
│       │   ├── prices_stocks_Spain.csv
│       │   ├── prices_stocks_Sweden.csv
│       │   ├── prices_stocks_Switzerland.csv
│       │   └── prices_stocks_UnitedKingdom.csv
│       │
│       ├── index_prices/             # MSCI index prices
│       │   ├── prices_indices_msci_Europe.csv
│       │   ├── prices_indices_msci_Germany.csv
│       │   ├── prices_indices_msci_France.csv
│       │   └── ... (8 files)
│       │
│       ├── riskfree_rates/           # Risk-free rate data
│       │   ├── Germany 3 Month Bond Yield Historical Data.csv
│       │   ├── riskfree_rate_Germany.csv
│       │   └── ... (5 files)
│       │
│       └── exchange_rates/           # Currency exchange rates
│           ├── GBP_EUR.csv
│           ├── CHF_EUR.csv
│           ├── SEK_EUR.csv
│           └── ECB Data Portal_20251203101849.csv
```

---

## Key Findings Summary

| Finding | Details |
|---------|---------|
| **CAPM Status** | Partially rejected |
| **Market Premium (γ₁)** | -0.575%/month (not significant) |
| **Low-Beta Anomaly** | Low-beta stocks earn +1.12%/month more than high-beta |
| **Best Sharpe Ratio** | 0.88 (long-only tangency portfolio) |
| **Sample** | 245 stocks, 7 countries, 58 months (2021-2025) |

---

## Country-Level Summary

| Country | N | Beta Mean | Beta Median | R² Mean | Sig Alpha % |
|---------|---|-----------|-------------|---------|-------------|
| Germany | 39 | 0.94 | 0.91 | 22.2% | 10.3% |
| France | 37 | 0.88 | 0.82 | 21.4% | 18.9% |
| Italy | 27 | 0.83 | 0.77 | 20.9% | 22.2% |
| Spain | 35 | 0.77 | 0.74 | 18.3% | 28.6% |
| Sweden | 27 | 1.14 | 1.08 | 32.2% | 7.4% |
| Switzerland | 16 | 0.83 | 0.78 | 24.6% | 6.3% |
| United Kingdom | 64 | 0.97 | 0.91 | 25.7% | 10.9% |

---

## Data Files Description

### Main Results
| File | Description |
|------|-------------|
| `capm_results.csv` | CAPM regression for all 245 stocks |
| `fama_macbeth_summary.csv` | Cross-sectional test statistics |
| `portfolio_results_long_only.csv` | Long-only portfolio metrics |
| `portfolio_results_short_selling_n15.csv` | **Short-selling with n=15 constrained sample** |

### Country-Level Analysis
| File | Description |
|------|-------------|
| `country_beta_summary.csv` | Mean/median beta by country |
| `fama_macbeth_country_summary.csv` | FM results by country |
| `fm_subperiod_comparison.csv` | Subperiod robustness |

### Robustness Checks
| File | Description |
|------|-------------|
| `beta_sorted_portfolios.csv` | Beta quintile analysis |
| `diversification_benefits.csv` | Diversification metrics |
| `robustness_summary.csv` | Summary of robustness tests |

### Raw Data
- **Stock Prices**: 7 CSV files (one per country)
- **Index Prices**: 8 MSCI index files
- **Risk-Free Rates**: German Bund yield data
- **Exchange Rates**: EUR conversion rates

---

## Technical Specifications

| Specification | Value |
|--------------|-------|
| Programming Language | Python 3.13 |
| Key Libraries | pandas, numpy, statsmodels, scipy |
| Statistical Methods | OLS with HC0 robust SE, Fama-MacBeth |
| Optimization | scipy.optimize (SLSQP) |
| Data Sources | Yahoo Finance, Investing.com, ECB |

---

## How to Use This Package

### For Academic Submission
1. `EXECUTIVE_SUMMARY.md` - Quick overview
2. `methodology/METHODOLOGY.md` - Methods section content
3. `reports/EMPIRICAL_RESULTS.md` - Results section content
4. `tables/` - Publication-ready tables (CSV and LaTeX)
5. `figures/` - Publication-ready figures
6. `data/` - Supporting data tables

### For Replication
1. Raw data in `data/raw/` contains original source files
2. Processed data in `data/processed/` contains analysis-ready data
3. All results can be reproduced using the main project code

---

*Package generated: December 2025*
