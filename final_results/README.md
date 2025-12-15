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
├── EXECUTIVE_SUMMARY.md      # 1-page summary of key findings
├── README.md                 # This file
│
├── methodology/
│   └── METHODOLOGY.md        # Complete methodology paper
│
├── reports/
│   └── EMPIRICAL_RESULTS.md  # Detailed empirical results
│
├── figures/
│   ├── efficient_frontier_no_shortselling.png    # Long-only frontier
│   ├── efficient_frontier_with_shortselling.png  # Unconstrained frontier
│   ├── beta_distribution_by_country.png          # Beta analysis
│   ├── alpha_distribution_by_country.png         # Alpha analysis
│   ├── r2_distribution_by_country.png            # R² analysis
│   ├── gamma1_timeseries.png                     # Risk premium over time
│   ├── fama_macbeth_by_country.png              # Country-level FM results
│   └── value_effect_analysis.png                 # Value effect analysis
│
└── data/
    ├── capm_results.csv                          # Full CAPM regression results
    ├── fama_macbeth_summary.csv                  # FM test statistics
    ├── fama_macbeth_monthly_coefficients.csv     # Monthly γ estimates
    ├── portfolio_results_long_only.csv           # Long-only portfolios
    ├── portfolio_results_short_selling.csv       # Short-selling portfolios
    ├── efficient_frontier_long_only.csv          # Long-only frontier points
    ├── efficient_frontier_short_selling.csv      # Short-selling frontier points
    └── robustness_summary.csv                    # Robustness check results
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

## How to Use This Package

### For Academic Submission
1. `EXECUTIVE_SUMMARY.md` - Quick overview
2. `methodology/METHODOLOGY.md` - Methods section content
3. `reports/EMPIRICAL_RESULTS.md` - Results section content
4. `figures/` - Publication-ready figures
5. `data/` - Supporting data tables

### For Presentations
1. Use figures from `figures/` folder
2. Reference key statistics from `EXECUTIVE_SUMMARY.md`
3. Cite methodology from `methodology/METHODOLOGY.md`

---

## Technical Specifications

| Specification | Value |
|--------------|-------|
| Programming Language | Python 3.13 |
| Key Libraries | pandas, numpy, statsmodels, scipy |
| Statistical Methods | OLS with HC0 robust SE, Fama-MacBeth |
| Optimization | scipy.optimize (SLSQP) |
| Data Sources | Yahoo Finance, Investing.com |

---

## Contact

For questions about this analysis, refer to the main project repository or contact the project maintainer.

---

*Package generated: December 2025*

