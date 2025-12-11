# Data Dictionary

Comprehensive documentation of all data files used in the CAPM analysis.

## Raw Data Files

### Stock Prices
**Location:** `data/raw/prices_stocks_*.csv` (one file per country)

**Format:** CSV with dates as index, stock tickers as columns

**Columns:**
- Index: Month-end dates (YYYY-MM-DD format)
- Columns: Stock tickers (e.g., 'ATO.PA', 'SAP.DE')

**Data Source:** Yahoo Finance (yfinance library)

**Update Frequency:** Monthly (end of month prices)

**Countries:**
- France: `prices_stocks_France.csv` (39 stocks)
- Germany: `prices_stocks_Germany.csv` (40 stocks)
- Italy: `prices_stocks_Italy.csv` (29 stocks)
- Spain: `prices_stocks_Spain.csv` (35 stocks)
- Sweden: `prices_stocks_Sweden.csv` (27 stocks)
- Switzerland: `prices_stocks_Switzerland.csv` (16 stocks)
- UnitedKingdom: `prices_stocks_UnitedKingdom.csv` (66 stocks)

**Data Quality Notes:**
- Prices are adjusted for splits and dividends
- Extreme returns (>10,000%) have been filtered for ATO.PA and III.L
- Missing values indicate delistings or data unavailability

### MSCI Index Prices
**Location:** `data/raw/prices_indices_msci_*.csv` (one file per country)

**Format:** CSV with dates as index, single column for index price

**Columns:**
- Index: Month-end dates
- Column: MSCI index price (via iShares ETF)

**Data Source:** Yahoo Finance (iShares ETFs: EWG, EWQ, EWI, EWP, EWD, EWL, EWU)

**Update Frequency:** Monthly

**Note:** These are proxy indices using US-listed ETFs that track MSCI country indices.

### Risk-Free Rates
**Location:** `data/raw/riskfree_rate_*.csv` (one file per country)

**Format:** CSV with dates as index, single column for risk-free rate

**Columns:**
- Index: Month-end dates
- Column: 3-month government bond yield (annual percentage)

**Data Source:** 
- EUR countries (France, Italy, Spain): German 3-month Bund (ECB or FRED)
- Germany: German 3-month Bund (ECB or FRED)
- Sweden: Swedish 3-month government bond (FRED)
- UnitedKingdom: UK 3-month government bond (FRED)
- Switzerland: Swiss 3-month government bond (FRED)

**Update Frequency:** Monthly

**Units:** Annual percentage (e.g., 0.5% = 0.5, not 0.005)

**Conversion:** Converted to monthly rates in processing: `monthly = (1 + annual/100)^(1/12) - 1` (compounding)

## Processed Data Files

### Returns Panel
**Location:** `data/processed/returns_panel.csv`

**Format:** Long-format CSV

**Columns:**
- `date`: Month-end date (YYYY-MM-DD)
- `country`: Country name
- `ticker`: Stock ticker
- `stock_return`: Monthly stock return (%)
- `msci_index_return`: Monthly MSCI index return (%)
- `riskfree_rate`: Monthly risk-free rate (%)
- `stock_excess_return`: Stock excess return = stock_return - riskfree_rate (%)
- `market_excess_return`: Market excess return = msci_index_return - riskfree_rate (%)

**Statistics:**
- Total rows: 14,691
- Unique stocks: 249
- Date range: 2021-01-31 to 2025-11-30 (59 months)
- Countries: 7

**Processing Steps:**
1. Convert prices to returns: `return_t = (price_t - price_{t-1}) / price_{t-1} * 100`
2. Filter stocks with <59 months of data
3. Handle missing values (forward fill, limit=2)
4. Align with MSCI index dates
5. Calculate excess returns

**See:** `data/processed/returns_panel_metadata.json` for detailed metadata

## Results Data Files

### CAPM Regression Results
**Location:** `results/data/capm_results.csv`

**Format:** CSV with one row per stock

**Columns:**
- `country`: Country name
- `ticker`: Stock ticker
- `beta`: Estimated beta coefficient
- `alpha`: Estimated alpha (intercept) in percentage
- `beta_tstat`: T-statistic for beta
- `alpha_tstat`: T-statistic for alpha
- `r_squared`: R-squared of regression
- `beta_se`: Standard error of beta
- `alpha_se`: Standard error of alpha
- `pvalue_beta`: P-value for beta (two-tailed)
- `pvalue_alpha`: P-value for alpha (two-tailed)
- `n_observations`: Number of observations used (should be 59)
- `is_valid`: Boolean flag for data quality (True if passes validation)

**Statistics:**
- Total stocks: 249
- Valid stocks: 245
- Average beta: 0.688
- Average R²: 0.235

### Fama-MacBeth Summary
**Location:** `results/reports/fama_macbeth_summary.csv`

**Format:** CSV with single row of summary statistics

**Columns:**
- `avg_gamma_1`: Average market price of risk (γ₁)
- `avg_gamma_0`: Average intercept (γ₀)
- `std_gamma_1`: Standard deviation of monthly γ₁
- `std_gamma_0`: Standard deviation of monthly γ₀
- `se_gamma_1`: Fama-MacBeth standard error of γ₁
- `se_gamma_0`: Fama-MacBeth standard error of γ₀
- `tstat_gamma_1`: T-statistic for γ₁
- `tstat_gamma_0`: T-statistic for γ₀
- `pvalue_gamma_1`: P-value for γ₁
- `pvalue_gamma_0`: P-value for γ₀
- `n_months`: Number of monthly regressions (should be 59)

**Key Result:** γ₁ = -0.9662 (t=-1.199, p=0.236) - NOT significant, CAPM rejected

### Market-Cap Weighted Betas
**Location:** `results/tables/table7_market_cap_weighted_betas.csv`

**Format:** CSV with one row per country plus overall

**Columns:**
- `country`: Country name or 'Overall'
- `n_stocks`: Number of stocks in sample
- `equal_weighted_beta`: Equal-weighted average beta
- `market_cap_weighted_beta`: Market-capitalization weighted average beta
- `market_cap_total_millions`: Total market cap in millions
- `n_stocks_with_mcap`: Number of stocks with market cap data
- `mcap_data_source`: Data source ('yfinance' or 'estimated')

**Key Finding:** Overall MW beta (0.636) vs EW beta (0.688) - 7.6% difference

### Portfolio Optimization Results
**Location:** `results/reports/portfolio_optimization_results.csv`

**Format:** CSV with one row per portfolio

**Columns:**
- `portfolio`: Portfolio name ('Minimum-Variance', 'Optimal Risky (Tangency)', 'Equal-Weighted')
- `expected_return`: Expected return in percentage (monthly)
- `volatility`: Portfolio volatility (standard deviation) in percentage (monthly)
- `sharpe_ratio`: Sharpe ratio (return / volatility)

**Key Results:**
- Minimum-Variance: Return 0.79%, Vol 1.31%, Sharpe 0.60
- Optimal Risky: Return 2.00%, Vol 1.93%, Sharpe 1.03

### Diversification Benefits
**Location:** `results/reports/diversification_benefits.csv`

**Format:** CSV with single row

**Columns:**
- `avg_stock_volatility`: Average individual stock volatility (%)
- `portfolio_volatility`: Portfolio volatility (%)
- `diversification_ratio`: Ratio of avg stock vol / portfolio vol
- `variance_reduction_pct`: Percentage reduction in variance
- `portfolio_return`: Portfolio expected return (%)

**Key Finding:** 82% variance reduction from diversification

### Value Effects Portfolios
**Location:** `results/reports/value_effects_portfolios.csv`

**Format:** CSV with one row per portfolio (quintile)

**Columns:**
- `portfolio`: Portfolio identifier (P1=Growth, P5=Value)
- `n_stocks`: Number of stocks in portfolio
- `avg_book_to_market`: Average book-to-market ratio
- `median_book_to_market`: Median book-to-market ratio
- `avg_alpha`: Average alpha from CAPM regression (%)
- `median_alpha`: Median alpha (%)
- `avg_beta`: Average beta
- `avg_r_squared`: Average R²
- `std_alpha`: Standard deviation of alphas

**Key Finding:** Reverse value effect (not significant): Value portfolio alpha = -0.42%, Growth = -0.10%

### Value Effects Test Results
**Location:** `results/reports/value_effects_test_results.csv`

**Format:** CSV with single row

**Columns:**
- `correlation`: Correlation between B/M and alpha
- `pvalue_correlation`: P-value for correlation
- `value_alpha`: Alpha of value portfolio (P5)
- `growth_alpha`: Alpha of growth portfolio (P1)
- `alpha_spread`: Alpha spread (value - growth)
- `regression_slope`: Regression slope (alpha on B/M)
- `regression_intercept`: Regression intercept
- `regression_r_squared`: R-squared of regression
- `regression_pvalue`: P-value for regression
- `interpretation`: Text interpretation of results

## Tables

**Location:** `results/tables/`

All tables are available in both CSV and LaTeX formats:
- `table1_capm_timeseries_summary.csv/.tex` - CAPM time-series summary by country
- `table2_fama_macbeth_results.csv/.tex` - Fama-MacBeth test results
- `table3_subperiod_results.csv/.tex` - Subperiod Fama-MacBeth results
- `table4_country_level_results.csv/.tex` - Country-level Fama-MacBeth results
- `table5_beta_sorted_portfolios.csv/.tex` - Beta-sorted portfolio returns
- `table6_descriptive_statistics.csv/.tex` - Descriptive statistics by country
- `table7_market_cap_weighted_betas.csv` - Market-cap weighted betas

## Plots

**Location:** `results/plots/`

**Key Visualizations:**
- `beta_distribution_by_country.png` - Beta distributions by country
- `r2_distribution_by_country.png` - R² distributions by country
- `beta_vs_return_scatter.png` - Beta vs average return scatter plot
- `beta_sorted_returns.png` - Beta-sorted portfolio returns (key figure)
- `efficient_frontier.png` - Mean-variance efficient frontier
- `value_effect_analysis.png` - Value effect analysis visualization
- `fm_gamma1_by_country.png` - Fama-MacBeth γ₁ by country

## Reports

**Location:** `results/reports/`

**Main Reports:**
- `CAPM_Analysis_Report.md` - Comprehensive analysis report
- `Executive_Summary.md` - Executive summary
- `Portfolio_Recommendation.md` - Portfolio recommendation for XYZ Asset Manager
- `Implementation_Summary.md` - Summary of implementation

## Data Quality Notes

1. **Extreme Returns:** ATO.PA and III.L had extreme returns (>10,000%) that were filtered
2. **Missing Data:** Stocks with >10% missing data were excluded
3. **Market Cap Data:** Some market cap values are estimated from price data when yfinance data unavailable
4. **Book-to-Market:** All B/M ratios obtained from Yahoo Finance; timing may not perfectly match analysis period
5. **Risk-Free Rates:** EUR countries use German Bund rates (common practice for Eurozone)

## Version History

- **v2.0 (2025-12-03):** Added market-cap weighted betas, portfolio optimization, value effects, portfolio recommendations
- **v1.0 (2025-12-01):** Initial analysis with CAPM regressions and Fama-MacBeth tests

