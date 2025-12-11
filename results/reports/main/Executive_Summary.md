# Executive Summary
## Testing the Capital Asset Pricing Model in European Equity Markets
### 2021-2025

---

## Overview

This study tests the Capital Asset Pricing Model (CAPM) across seven European equity markets using a sample of 245 valid stocks (out of 249 total) from Germany, France, Italy, Spain, Sweden, United Kingdom, and Switzerland. We employ both time-series and cross-sectional methodologies to evaluate whether beta explains stock returns.

**Period:** January 2021 - November 2025 (59 months)  
**Markets:** 7 European countries  
**Stocks:** 245 valid stocks (out of 249 total) with complete data

---

## Key Findings

### 1. Time-Series Results: Moderate Explanatory Power

- **Average R²:** 0.235 (market beta explains ~24% of return variation)
- **Average Beta:** 0.688 (median: 0.646)
- **Significant Betas:** 95.9% of stocks have statistically significant betas

**Interpretation:** CAPM has moderate time-series explanatory power, but approximately 76% of return variation remains unexplained by market risk alone.

### 2. Cross-Sectional Results: CAPM REJECTED

**Fama-MacBeth Test Results:**
- **Market Price of Risk (γ₁):** -0.9662
- **t-statistic:** -1.199
- **p-value:** 0.236
- **Conclusion:** **NOT statistically significant**

**Intercept (γ₀):** 1.5385% (t=4.444, highly significant)

**Interpretation:** Beta does **NOT** explain cross-sectional variation in expected returns. Higher beta stocks do not earn higher average returns than lower beta stocks.

### 3. Robustness: Results Hold Across Specifications

**Subperiod Analysis:**
- Period A (2021-2022): γ₁ = -1.7935 (not significant)
- Period B (2023-2025): γ₁ = -0.3989 (not significant)

**Country-Level Analysis:**
- France: γ₁ = -2.4455 (borderline significant negative)
- Sweden: γ₁ = -2.4246 (negative)
- Other countries: Weak or insignificant relationships

**Beta-Sorted Portfolios:**
- Portfolio 1 (lowest beta): Return = 1.20%
- Portfolio 5 (highest beta): Return = 0.54%
- **Negative relationship:** Higher beta → Lower return (contrary to CAPM)

**Clean Sample:** Results confirmed after removing outliers

---

## Main Conclusion

**The Capital Asset Pricing Model fails to explain cross-sectional variation in European stock returns during 2021-2025.**

While beta has moderate time-series explanatory power (R² ≈ 0.24), it does not price returns in the cross-section. This finding is:
- Consistent across subperiods
- Robust across countries
- Confirmed by portfolio analysis
- Aligned with decades of empirical finance research

---

## Implications

### For Researchers
- CAPM remains a useful theoretical benchmark but fails empirically
- Multi-factor models (Fama-French, Carhart) are necessary
- Country-specific factors matter in international settings

### For Practitioners
- Beta alone is insufficient for risk assessment
- Multi-factor models should be used for portfolio construction
- Sector-specific and company-specific factors must be considered

### For Investors
- Higher beta does not guarantee higher returns
- Diversification across factors (size, value, quality) may be more important
- Understanding sector and company-specific risks is crucial

---

## Why CAPM Might Fail (2021-2025)

1. **Macro Regime Shifts:**
   - COVID recovery and stimulus effects (2021)
   - Interest rate hikes and inflation (2022)
   - Policy normalization (2023-2025)

2. **Sector Effects:**
   - European markets have heavy concentrations in banks, cyclicals, and defensive sectors
   - Sector-specific factors create return patterns orthogonal to market beta

3. **Alternative Factors:**
   - Size, value, quality, momentum factors may dominate beta
   - ESG considerations and ETF flows may create return patterns unrelated to beta

---

## Supporting Evidence

### Visual Evidence
- **Beta-Sorted Portfolios:** Clear negative slope (higher beta → lower return)
- **Time Series of γ₁:** Values bounce around zero with no positive trend
- **Country-Level Results:** Negative or insignificant relationships across countries

### Statistical Evidence
- Fama-MacBeth test: γ₁ insignificant (p = 0.236)
- Subperiod tests: Consistent rejection in both periods
- Portfolio analysis: Negative beta-return relationship

---

## Recommendations

1. **Use Multi-Factor Models:** Fama-French three-factor or Carhart four-factor models
2. **Consider Sector Factors:** Beyond market beta, sector-specific risks are crucial
3. **Account for Size/Value:** Small and value stocks may have different risk-return profiles
4. **Monitor Quality Factors:** Profitability and investment factors may explain returns
5. **CAPM as Starting Point:** Use CAPM as a benchmark but adjust for additional factors

---

## Data & Methodology

**Data Sources:**
- Stock prices: Yahoo Finance (monthly, 2020-12 to 2025-11)
- Market proxies: MSCI country indices via iShares ETFs
- Risk-free rates: 3-month government bond yields (country-specific)

**Methodology:**
- Time-series CAPM regressions for each stock
- Fama-MacBeth two-pass cross-sectional test
- Robustness checks: subperiods, countries, portfolios, clean sample

**Sample:**
- 219 stocks with complete data
- 7 European markets
- 59 months of returns

---

## Full Report

For complete analysis, tables, figures, and detailed interpretation, see:
- **Full Report:** `results/reports/CAPM_Analysis_Report.md`
- **Thesis Chapter:** `results/reports/thesis_chapter7_empirical_results.md`
- **Tables:** `results/tables/`
- **Figures:** `results/plots/`

---

*Generated: December 3, 2025*

