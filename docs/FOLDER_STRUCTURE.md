# CAPM Analysis Project - Folder Structure

This document describes the organization of the CAPM Analysis project folder structure.

## Overview

The project follows a clear separation of concerns:
- **`analysis/`** - All analysis code (core, extensions, utilities)
- **`data/`** - All data files (raw, processed, metadata)
- **`results/`** - All analysis outputs (organized by analysis type)
- **`audit/`** - Audit and validation modules
- **`tests/`** - Test files
- **`docs/`** - Documentation
- **`scripts/`** - Utility scripts

## Detailed Structure

```
CAPM_Analysis/
├── analysis/                    # Analysis code
│   ├── core/                   # Core CAPM analysis modules
│   │   ├── capm_regression.py
│   │   ├── fama_macbeth.py
│   │   ├── returns_processing.py
│   │   └── robustness_checks.py
│   ├── data/                   # Data collection and processing
│   │   ├── data_collection.py
│   │   ├── riskfree_helper.py
│   │   └── yf_helper.py
│   ├── extensions/             # Extended analyses
│   │   ├── portfolio_optimization.py
│   │   ├── value_effects.py
│   │   ├── market_cap_analysis.py
│   │   └── investment_recommendations.py
│   ├── frameworks/             # Framework modules (some not yet implemented)
│   ├── reporting/              # Report generation
│   └── utils/                  # Utility functions and configuration
│       └── config.py           # Central configuration
│
├── data/                       # Data files
│   ├── raw/                   # Raw data files
│   │   ├── prices/            # Stock price data
│   │   ├── riskfree_rates/   # Risk-free rate data
│   │   └── exchange_rates/   # Exchange rate data
│   ├── processed/            # Processed data (panels, returns)
│   ├── metadata/             # Data metadata
│   └── baselines/            # Baseline data for regression tests
│
├── results/                   # Analysis outputs (ORGANIZED STRUCTURE)
│   ├── portfolio_optimization/
│   │   ├── long_only/        # Long-only (constrained) portfolios
│   │   │   ├── efficient_frontier.csv
│   │   │   ├── efficient_frontier.png
│   │   │   └── portfolios.csv
│   │   ├── short_selling/     # Unconstrained (short-selling) portfolios
│   │   │   ├── efficient_frontier.csv
│   │   │   ├── efficient_frontier.png
│   │   │   └── portfolios.csv
│   │   └── realistic_short/   # Realistic short-selling (with constraints)
│   │       ├── efficient_frontier.csv
│   │       ├── efficient_frontier.png
│   │       └── portfolios.csv
│   ├── capm_analysis/
│   │   ├── time_series/       # Time-series CAPM results
│   │   │   ├── by_country.csv
│   │   │   ├── clean_sample.csv
│   │   │   └── figures/       # CAPM time-series figures
│   │   └── cross_sectional/   # Fama-MacBeth results
│   │       ├── summary.csv
│   │       ├── monthly_coefficients.csv
│   │       └── figures/       # Fama-MacBeth figures
│   ├── figures/               # Consolidated figures (ORGANIZED)
│   │   ├── capm/              # CAPM-related figures
│   │   │   ├── alpha_distribution_by_country.png
│   │   │   ├── average_beta_by_country.png
│   │   │   ├── beta_boxplot_by_country.png
│   │   │   ├── beta_distribution_by_country.png
│   │   │   ├── beta_vs_r2_scatter.png
│   │   │   ├── top_bottom_beta_stocks.png
│   │   │   └── r2_distribution_by_country.png
│   │   ├── fama_macbeth/      # Fama-MacBeth figures
│   │   │   ├── gamma1_timeseries.png
│   │   │   ├── gamma1_histogram.png
│   │   │   ├── fama_macbeth_by_country.png
│   │   │   ├── fm_subperiod_comparison.png
│   │   │   ├── fm_gamma1_by_country.png
│   │   │   └── beta_vs_return_scatter.png
│   │   ├── portfolio/         # Portfolio optimization figures
│   │   │   ├── efficient_frontier.png
│   │   │   ├── efficient_frontier_constrained.png
│   │   │   ├── efficient_frontier_unconstrained.png
│   │   │   ├── efficient_frontier_realistic_short.png
│   │   │   ├── efficient_frontier_top15_short_selling.png
│   │   │   └── beta_sorted_returns.png
│   │   └── value_effects/     # Value effects figures
│   │       └── value_effect_analysis.png
│   ├── tables/                # LaTeX tables for papers
│   └── reports/               # Markdown and CSV reports (ORGANIZED)
│       ├── main/              # Main analysis reports
│       │   ├── CAPM_Analysis_Report.md
│       │   ├── Executive_Summary.md
│       │   ├── Implementation_Summary.md
│       │   └── R2_improvements_analysis.md
│       ├── investment/        # Investment recommendations
│       │   └── Investment_Recommendations_US_Investor.md
│       ├── thesis/           # Thesis chapter outputs
│       │   └── thesis_chapter7_empirical_results.md
│       ├── capm/             # CAPM analysis CSVs
│       │   ├── capm_by_country.csv
│       │   ├── capm_extremes.csv
│       │   └── capm_clean_sample.csv
│       ├── fama_macbeth/     # Fama-MacBeth CSVs
│       │   ├── fama_macbeth_summary.csv
│       │   ├── fama_macbeth_monthly_coefficients.csv
│       │   ├── fama_macbeth_beta_returns.csv
│       │   ├── fama_macbeth_clean_sample.csv
│       │   ├── fm_by_country.csv
│       │   ├── fm_subperiod_A.csv
│       │   ├── fm_subperiod_B.csv
│       │   └── fm_subperiod_comparison.csv
│       ├── portfolio/        # Portfolio optimization CSVs
│       │   ├── portfolio_optimization_results.csv
│       │   ├── portfolio_optimization_realistic_short.csv
│       │   ├── portfolio_optimization_top15_short_selling.csv
│       │   ├── efficient_frontier_unconstrained.csv
│       │   └── efficient_frontier_top15_short_selling.csv
│       ├── value_effects/     # Value effects CSVs
│       │   └── value_effects_test_results.csv
│       ├── robustness/        # Robustness checks CSVs
│       │   ├── robustness_summary.csv
│       │   └── diversification_benefits.csv
│       └── constrained_only/ # Constrained-only results
│           ├── efficient_frontier.csv
│           ├── portfolio_optimization_results.csv
│           └── README.md
│
├── audit/                     # Audit and validation modules
│   ├── run_full_audit.py      # Main audit orchestrator
│   ├── validate_*.py          # Various validation modules
│   └── reports/               # Audit reports (ORGANIZED)
│       ├── audit_report.md
│       ├── verification_report.md
│       ├── sharpe_ratio_audit_report.md
│       └── warnings_summary.md
│
├── tests/                     # Test files
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
│
├── docs/                      # Documentation (ORGANIZED)
│   ├── methodology/           # Methodology documentation
│   │   ├── METHODOLOGY.md
│   │   ├── RESULTS_INTERPRETATION.md
│   │   └── RESULTS_PORTRAYAL.md
│   ├── implementation/        # Implementation notes
│   │   └── [implementation guides]
│   ├── summaries/            # Summary documents (ORGANIZED)
│   │   ├── progress/         # Progress summaries
│   │   │   ├── PROGRESS_SUMMARY.md
│   │   │   ├── FINAL_SUMMARY.md
│   │   │   ├── IMPLEMENTATION_COMPLETE_SUMMARY.md
│   │   │   └── REORGANIZATION_SUMMARY.md
│   │   ├── optimization/     # Optimization summaries
│   │   │   ├── COMPLETE_OPTIMIZATION_SUMMARY.md
│   │   │   ├── COMPREHENSIVE_OPTIMIZATION_SUMMARY.md
│   │   │   ├── FINAL_OPTIMIZATION_REPORT.md
│   │   │   ├── PORTFOLIO_OPTIMIZATION_IMPROVEMENTS.md
│   │   │   └── PORTFOLIO_OPTIMIZATION_FIXES.md
│   │   ├── fixes/            # Fix summaries
│   │   │   ├── ALL_WARNINGS_FIXED.md
│   │   │   ├── FINAL_WARNINGS_FIXED.md
│   │   │   ├── WARNINGS_AND_OPTIMIZATIONS.md
│   │   │   ├── AUDIT_FIXES_SUMMARY.md
│   │   │   └── FINANCIAL_CORRECTNESS_SUMMARY.md
│   │   └── audit/            # Audit summaries
│   ├── reports/              # Documentation reports
│   │   └── data_quality_report.md
│   ├── EXECUTION_GUIDE.md    # How to run analyses
│   ├── INVESTMENT_SUMMARY.md  # Investment summary
│   ├── FOLDER_STRUCTURE.md    # This file
│   └── NEXT_STEPS.md         # Future improvements
│
├── scripts/                   # Utility scripts
│   ├── monitor_progress.sh
│   └── show_progress.sh
│
└── logs/                      # Log files (last 7 days kept)
```

## Key Principles

1. **Separation by Analysis Type**: Results are organized by the type of analysis (portfolio optimization, CAPM time-series, CAPM cross-sectional, etc.)

2. **Clear Hierarchy**: Each analysis type has its own subdirectory with data files and figures together

3. **No Duplication**: Each file exists in one location only

4. **Logical Grouping**: Related files are grouped together (e.g., all portfolio optimization results in one place)

5. **Organized Documentation**: Documentation is organized by purpose (methodology, implementation, summaries, reports)

6. **Clean Structure**: All files are in appropriate subdirectories, no clutter at root levels

## Recent Improvements (2025-12-XX)

### Documentation Organization
- **Created `docs/summaries/`** with subdirectories:
  - `progress/` - Progress and status summaries
  - `optimization/` - Optimization-related summaries
  - `fixes/` - Bug fixes and warning resolution summaries
  - `audit/` - Audit-related summaries
- **Created `docs/reports/`** for documentation reports
- Moved all summary files from root `docs/` to appropriate subdirectories

### Results Organization
- **Organized `results/figures/`** into subdirectories:
  - `capm/` - CAPM time-series figures
  - `fama_macbeth/` - Fama-MacBeth figures
  - `portfolio/` - Portfolio optimization figures
  - `value_effects/` - Value effects figures
- **Organized `results/reports/`** into subdirectories:
  - `capm/` - CAPM analysis CSVs
  - `fama_macbeth/` - Fama-MacBeth CSVs
  - `portfolio/` - Portfolio optimization CSVs
  - `value_effects/` - Value effects CSVs
  - `robustness/` - Robustness check CSVs

### Audit Organization
- **Created `audit/reports/`** for all audit markdown reports
- Moved audit reports from root `audit/` to `audit/reports/`

### Cleanup
- Cleaned old log files (kept last 7 days)
- Removed Python cache files (`__pycache__/`, `*.pyc`)

## Adding New Files

When adding new analysis outputs:

1. **Portfolio Optimization**: Add to `results/portfolio_optimization/[type]/`
2. **CAPM Analysis**: Add to `results/capm_analysis/[time_series|cross_sectional]/`
3. **Figures**: Add to `results/figures/[category]/`
4. **Reports**: Add to `results/reports/[category]/`
5. **Documentation**: Add to `docs/[category]/` (methodology, implementation, summaries, reports)

Always update `analysis/utils/config.py` with new path constants if needed.

## Configuration

All paths are defined in `analysis/utils/config.py`. Import from there:

```python
from analysis.utils.config import (
    RESULTS_PORTFOLIO_LONG_ONLY_DIR,
    RESULTS_CAPM_TIMESERIES_DIR,
    RESULTS_FIGURES_CAPM_DIR,
    # etc.
)
```
