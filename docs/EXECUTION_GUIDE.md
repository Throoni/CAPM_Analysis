# CAPM Analysis Execution Guide

## Overview

This guide provides step-by-step instructions to run all analyses in the correct order. **All analyses use real data only** - no placeholder values are used.

## Prerequisites

1. **Data Files**: Ensure all required data files are in `data/raw/`:
   - Stock prices: `data/raw/prices/prices_stocks_*.csv` (7 countries)
   - MSCI index prices: `data/raw/prices/prices_indices_msci_*.csv` (7 countries)
   - Risk-free rates: `data/raw/riskfree_rates/riskfree_rate_*.csv` (7 countries)

2. **Python Environment**: Activate virtual environment and install dependencies:
   ```bash
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

3. **Environment Variables** (Optional but recommended):
   - Set `FRED_API_KEY` in `.env` file for automatic risk-free rate updates
   - See `README.md` for setup instructions

## Execution Order

### Step 1: Data Collection & Validation (If Needed)

**Purpose**: Download and validate raw data files.

**Command**:
```bash
python3 -m analysis.data.data_collection
```

**Output**: Raw data files in `data/raw/prices/` and `data/raw/riskfree_rates/`

**Note**: Skip if data files already exist.

---

### Step 2: Returns Processing

**Purpose**: Convert prices to returns and create the returns panel.

**Command**:
```bash
python3 -m analysis.core.returns_processing
```

**Output**: 
- `data/processed/returns_panel.csv` - Main returns panel with excess returns

**Dependencies**: Requires raw price data and risk-free rate files.

**Expected Result**: Panel with columns: `date`, `country`, `ticker`, `stock_return`, `msci_index_return`, `riskfree_rate`, `stock_excess_return`, `market_excess_return`

---

### Step 3: CAPM Regression Analysis

**Purpose**: Estimate beta, alpha, and R² for each stock using time-series regression.

**Command**:
```bash
python3 -m analysis.core.capm_regression
```

**Output**:
- `results/data/capm_results.csv` - Full regression results for all stocks
- `results/reports/data/capm_by_country.csv` - Country-level summaries
- `results/reports/data/capm_extremes.csv` - Top/bottom stocks by beta
- `results/reports/main/CAPM_Analysis_Report.md` - Comprehensive report
- Visualizations in `results/figures/`

**Dependencies**: Requires `data/processed/returns_panel.csv`

**Expected Result**: ~245 stocks with beta, alpha, R², p-values, and validation flags

---

### Step 4: Fama-MacBeth Cross-Sectional Test

**Purpose**: Test whether beta explains cross-sectional variation in expected returns.

**Command**:
```bash
python3 -m analysis.core.fama_macbeth
```

**Output**:
- `results/reports/data/fama_macbeth_summary.csv` - Main test results
- `results/reports/data/fama_macbeth_monthly_coefficients.csv` - Monthly γ₁ estimates
- `results/reports/data/fama_macbeth_beta_returns.csv` - Beta vs. average returns
- `results/reports/data/fm_by_country.csv` - Country-level results
- `results/reports/data/fm_subperiod_*.csv` - Subperiod analysis
- Visualizations in `results/figures/`

**Dependencies**: Requires `results/data/capm_results.csv` and `data/processed/returns_panel.csv`

**Expected Result**: Test of whether γ₁ (market price of risk) is significantly different from zero

---

### Step 5: Robustness Checks

**Purpose**: Validate results across subperiods, countries, and specifications.

**Command**:
```bash
python3 -m analysis.core.robustness_checks
```

**Output**:
- `results/reports/data/robustness_summary.csv` - Summary of all robustness tests
- `results/reports/data/capm_clean_sample.csv` - Results excluding extreme betas
- `results/reports/data/fm_subperiod_comparison.csv` - Subperiod comparison
- Visualizations in `results/figures/`

**Dependencies**: Requires `results/data/capm_results.csv` and `data/processed/returns_panel.csv`

**Expected Result**: Validation that results hold across different specifications

---

### Step 6: Portfolio Optimization

**Purpose**: Construct optimal portfolios and analyze diversification benefits.

**Command**:
```bash
python3 -m analysis.extensions.portfolio_optimization
```

**Output**:
- `results/reports/data/portfolio_optimization_results.csv` - Optimal portfolio characteristics
- `results/reports/data/diversification_benefits.csv` - Diversification metrics
- `results/reports/data/beta_sorted_portfolios.csv` - Beta-sorted portfolio returns
- Visualizations in `results/figures/`

**Dependencies**: Requires `results/data/capm_results.csv` and `data/processed/returns_panel.csv`

**Expected Result**: Efficient frontier, optimal portfolio weights, Sharpe ratios, diversification benefits

---

### Step 7: Value Effects Analysis

**Purpose**: Test whether book-to-market ratios explain returns beyond beta.

**Command**:
```bash
python3 -m analysis.extensions.value_effects
```

**Output**:
- `results/reports/data/value_effects_test_results.csv` - Value effect test results
- `results/reports/data/value_effects_portfolios.csv` - Value-sorted portfolio returns
- Visualizations in `results/figures/`

**Dependencies**: Requires `results/data/capm_results.csv` and book-to-market data

**Expected Result**: Test of whether value stocks outperform growth stocks

---

### Step 8: Market Cap Analysis

**Purpose**: Analyze whether market capitalization affects beta estimates.

**Command**:
```bash
python3 -m analysis.extensions.market_cap_analysis
```

**Output**:
- Analysis results and visualizations

**Dependencies**: Requires `results/data/capm_results.csv` and market cap data

**Expected Result**: Comparison of beta estimates across market cap quintiles

---

### Step 9: Portfolio Recommendation

**Purpose**: Synthesize all findings and make active vs. passive recommendation.

**Command**:
```bash
python3 -m analysis.extensions.portfolio_recommendation
```

**Output**:
- `results/reports/Portfolio_Recommendation.md` - Comprehensive recommendation report

**Dependencies**: Requires outputs from Steps 3-8

**Expected Result**: Active/Passive/Hybrid recommendation with justification

---

### Step 10: Investment Recommendations (US Investor Focus)

**Purpose**: Generate investment recommendations specifically for American equity investors.

**Command**:
```bash
python3 -m analysis.extensions.investment_recommendations
```

**Output**:
- `results/reports/Investment_Recommendations_US_Investor.md` - Investment guide for US investors

**Dependencies**: Requires outputs from Steps 3-9

**Expected Result**: Currency hedging recommendations, country/sector allocation, factor-based strategies, ETF recommendations

---

## Quick Run (All Steps)

To run all analyses in sequence:

```bash
# Step 1: Returns processing
python3 -m analysis.core.returns_processing

# Step 2: CAPM regression
python3 -m analysis.core.capm_regression

# Step 3: Fama-MacBeth test
python3 -m analysis.core.fama_macbeth

# Step 4: Robustness checks
python3 -m analysis.core.robustness_checks

# Step 5: Portfolio optimization
python3 -m analysis.extensions.portfolio_optimization

# Step 6: Value effects
python3 -m analysis.extensions.value_effects

# Step 7: Market cap analysis
python3 -m analysis.extensions.market_cap_analysis

# Step 8: Portfolio recommendation
python3 -m analysis.extensions.portfolio_recommendation

# Step 9: Investment recommendations
python3 -m analysis.extensions.investment_recommendations
```

## Verification

After running all steps, verify outputs:

```bash
# Check key output files exist
ls -la results/data/capm_results.csv
ls -la results/reports/data/fama_macbeth_summary.csv
ls -la results/reports/Portfolio_Recommendation.md
ls -la results/reports/Investment_Recommendations_US_Investor.md

# Check data quality
python3 -m analysis.utils.data_quality_report

# Run full audit
python3 -m audit.run_full_audit
```

## Troubleshooting

### Error: "All data sources failed for [country]"

**Cause**: Risk-free rate data not available for a country.

**Solution**: 
1. Check that CSV file exists: `data/raw/riskfree_rates/riskfree_rate_[Country].csv`
2. Verify CSV file has data for the analysis period (2021-2025)
3. System requires real data - no placeholder fallback

### Error: "No panels created - check data availability"

**Cause**: Returns processing failed for all countries.

**Solution**:
1. Verify stock price files exist: `data/raw/prices/prices_stocks_*.csv`
2. Verify MSCI index files exist: `data/raw/prices/prices_indices_msci_*.csv`
3. Check that files have sufficient data (minimum 60 months)

### Missing Output Files

**Cause**: A dependency step was skipped.

**Solution**: Run steps in order - each step depends on previous outputs.

## Expected Runtime

- **Returns Processing**: ~2-5 minutes
- **CAPM Regression**: ~5-10 minutes
- **Fama-MacBeth**: ~2-5 minutes
- **Robustness Checks**: ~3-5 minutes
- **Portfolio Optimization**: ~2-3 minutes
- **Value Effects**: ~1-2 minutes
- **Market Cap Analysis**: ~1-2 minutes
- **Portfolio Recommendation**: ~30 seconds
- **Investment Recommendations**: ~1-2 minutes

**Total**: ~20-35 minutes for full pipeline

## Notes

- **Real Data Only**: System requires real data - no placeholder values are used
- **Fail Hard**: If data is missing, system will fail with clear error messages
- **Reproducibility**: All analyses use fixed random seeds where applicable
- **Documentation**: See `docs/methodology/METHODOLOGY.md` for detailed methodology

