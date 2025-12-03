# Testing the Capital Asset Pricing Model in European Equity Markets
## Empirical Analysis: 2021-2025

**Author:** [Your Name]  
**Date:** December 2025  
**Institution:** [Your Institution]

---

## Executive Summary

This report presents a comprehensive empirical test of the Capital Asset Pricing Model (CAPM) across seven European equity markets from 2021 to 2025. Using a sample of 219 stocks from Germany, France, Italy, Spain, Sweden, United Kingdom, and Switzerland, we employ both time-series and cross-sectional methodologies to evaluate whether beta explains stock returns.

### Key Findings

1. **Time-Series Results:** CAPM demonstrates moderate explanatory power, with an average R² of 0.235, indicating that market beta explains approximately 24% of individual stock return variation.

2. **Cross-Sectional Results:** The Fama-MacBeth test **rejects** the CAPM's core prediction. The market price of risk (γ₁) is -0.9662 with a t-statistic of -1.199 (p = 0.236), indicating that **beta does not explain cross-sectional variation in expected returns**.

3. **Robustness:** Results are consistent across:
   - Subperiods (2021-2022 vs. 2023-2025)
   - Individual countries
   - Beta-sorted portfolios
   - Clean sample specifications

4. **Economic Interpretation:** The CAPM failure suggests that multi-factor models (Fama-French, Carhart) incorporating size, value, profitability, and momentum factors are necessary to explain expected returns in European markets.

### Main Conclusion

**The Capital Asset Pricing Model fails to explain cross-sectional variation in European stock returns during 2021-2025.** While beta has moderate time-series explanatory power, it does not price returns in the cross-section, consistent with decades of empirical finance research showing that CAPM is an incomplete model of expected returns.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Data & Methodology](#3-data--methodology)
4. [Time-Series CAPM Results](#4-time-series-capm-results)
5. [Cross-Sectional CAPM Test: Fama-MacBeth](#5-cross-sectional-capm-test-fama-macbeth)
6. [Robustness Checks](#6-robustness-checks)
7. [Economic Interpretation](#7-economic-interpretation)
8. [Conclusions & Implications](#8-conclusions--implications)
9. [References](#9-references)
10. [Appendices](#10-appendices)

---

## 1. Introduction

### 1.1 Motivation

The Capital Asset Pricing Model (CAPM), developed by Sharpe (1964) and Lintner (1965), remains one of the most influential models in finance. It posits that the expected return on an asset is linearly related to its beta (sensitivity to market movements), with higher beta stocks earning higher expected returns. Despite its theoretical elegance, empirical tests of CAPM have produced mixed results, with many studies finding that beta alone is insufficient to explain cross-sectional variation in stock returns.

This study tests the CAPM across seven European equity markets during a period of significant economic and financial market volatility (2021-2025), including:
- Post-COVID recovery and stimulus effects (2021)
- Interest rate hikes and inflation concerns (2022)
- Economic recovery and policy normalization (2023-2025)

### 1.2 Research Questions

This analysis addresses three fundamental questions:

1. **Time-Series Question:** Does market beta explain individual stock returns over time?
2. **Cross-Sectional Question:** Do higher beta stocks earn higher average returns than lower beta stocks?
3. **Robustness Question:** Are the results consistent across different time periods, countries, and sample specifications?

### 1.3 Contribution

This study contributes to the empirical finance literature by:
- Providing a comprehensive test of CAPM across multiple European markets
- Using country-specific risk-free rates and market proxies
- Employing both time-series and cross-sectional methodologies
- Conducting extensive robustness checks
- Analyzing a recent period (2021-2025) with significant macro volatility

---

## 2. Literature Review

### 2.1 Theoretical Foundation

The CAPM is derived from Markowitz (1952) portfolio theory and assumes:
- Investors are risk-averse and maximize expected utility
- All investors have identical expectations about returns and risks
- There is a risk-free asset available
- Markets are frictionless and efficient
- All investors hold the same market portfolio

Under these assumptions, the CAPM predicts:

$$E[R_i] = R_f + \beta_i (E[R_m] - R_f)$$

where:
- $E[R_i]$ is the expected return on asset $i$
- $R_f$ is the risk-free rate
- $\beta_i$ is the asset's beta (sensitivity to market movements)
- $E[R_m] - R_f$ is the market risk premium

### 2.2 Empirical Evidence

**Early Tests (1970s-1980s):**
- Black, Jensen, and Scholes (1972) found that low-beta stocks earned higher returns than predicted by CAPM
- Fama and MacBeth (1973) developed the two-pass regression methodology and found mixed evidence for CAPM

**The Fama-French Revolution (1990s):**
- Fama and French (1992) found that beta does not explain cross-sectional variation in returns
- Fama and French (1993) proposed a three-factor model (market, size, value) that outperforms CAPM
- These findings have been replicated across many markets and time periods

**Recent Evidence:**
- Mixed results depending on sample period, market, and methodology
- Some studies find beta is priced in certain contexts (e.g., long horizons, specific markets)
- Multi-factor models (Fama-French, Carhart) generally outperform CAPM

### 2.3 International Evidence

Studies of CAPM in European markets have generally found:
- Beta has moderate time-series explanatory power
- Beta does not consistently explain cross-sectional variation
- Country-specific factors matter
- Multi-factor models perform better than CAPM

---

## 3. Data & Methodology

### 3.1 Sample Selection

**Markets Covered:**
- Germany (DE)
- France (FR)
- Italy (IT)
- Spain (ES)
- Sweden (SE)
- United Kingdom (UK)
- Switzerland (CH)

**Stock Universe:**
- Initial universe: 313 stocks
- After data quality filters: 249 stocks
- Final sample with complete data: 219 stocks

**Selection Criteria:**
- Large-cap stocks from major European indices
- Sufficient price history (2020-12-01 to 2025-12-01)
- Complete return data for analysis period
- Excluded stocks with excessive missing data or data quality issues

### 3.2 Data Sources

**Stock Prices:**
- Source: Yahoo Finance
- Frequency: Monthly (month-end)
- Period: December 2020 to November 2025 (60 months of prices, 59 months of returns)

**Market Proxies:**
- MSCI country indices accessed via iShares ETFs:
  - Germany: EWG (MSCI Germany)
  - France: EWQ (MSCI France)
  - Italy: EWI (MSCI Italy)
  - Spain: EWP (MSCI Spain)
  - Sweden: EWD (MSCI Sweden)
  - United Kingdom: EWU (MSCI UK)
  - Switzerland: EWL (MSCI Switzerland)

**Risk-Free Rates:**
- 3-month government bond yields for each country:
  - Eurozone countries (DE, FR, IT, ES): German 3-month Bund (EUR)
  - Sweden: Swedish 3-month government bond (SEK)
  - United Kingdom: UK 3-month Treasury Bill (GBP)
  - Switzerland: Swiss 3-month government bond (CHF)
- Converted from annual percentage to monthly percentage

### 3.3 Return Calculations

**Simple Returns:**
All returns are calculated as simple (arithmetic) percentages:

$$R_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}$$

**Excess Returns:**
Stock excess return:
$$E_{i,t} = R_{i,t} - R_{f,t}$$

Market excess return:
$$E_{m,t} = R_{m,t} - R_{f,t}$$

### 3.4 Methodology Overview

**Stage 1: Time-Series CAPM Regressions**
For each stock $i$, estimate:
$$R_{i,t} - R_{f,t} = \alpha_i + \beta_i (R_{m,t} - R_{f,t}) + \varepsilon_{i,t}$$

**Stage 2: Fama-MacBeth Cross-Sectional Test**
- Pass 1: Estimate $\beta_i$ from time-series regressions
- Pass 2: For each month $t$, run cross-sectional regression:
  $$R_{i,t} = \gamma_{0,t} + \gamma_{1,t} \beta_i + u_{i,t}$$
- Average $\gamma_0$ and $\gamma_1$ across months and compute Fama-MacBeth t-statistics

**Stage 3: Robustness Checks**
- Subperiod analysis (2021-2022 vs. 2023-2025)
- Country-level Fama-MacBeth tests
- Beta-sorted portfolio analysis
- Clean sample tests (removing outliers)

---

## 4. Time-Series CAPM Results

### 4.1 Aggregate Statistics

The time-series regressions yield the following aggregate results:

| Statistic | Value |
|-----------|-------|
| Valid stocks | 219 |
| Average beta | 0.688 |
| Median beta | 0.646 |
| Average R² | 0.235 |
| Median R² | 0.210 |
| Average alpha | 0.144% |
| % Significant betas (p < 0.05) | 95.9% |

### 4.2 Results by Country

**Table 1: CAPM Time-Series Summary by Country**

*(See Appendix A.1 for full table)*

Key findings by country:
- **Beta values** are economically reasonable, with median betas ranging from 0.53 to 0.67
- **R² values** average approximately 0.24, indicating moderate explanatory power
- **Significant betas** represent 95.9% of the sample, indicating beta is statistically distinguishable from zero for most stocks

### 4.3 Distribution Analysis

**Beta Distribution:**
- Most stocks have betas below 1.0, as expected for European large-cap stocks
- Distribution is centered around 0.6-0.7
- Relatively few extreme values (after removing outliers)

**R² Distribution:**
- Average R² of 0.235 indicates that market beta explains approximately 24% of return variation
- Approximately 76% of return variation remains unexplained by market risk alone
- This suggests that firm-specific factors, sector effects, and other non-market drivers play a substantial role

### 4.4 Interpretation

**In time series, the CAPM is partially useful:**

1. **Market risk matters:** Beta captures a meaningful portion of stock return variation, confirming that market movements are an important driver of individual stock returns.

2. **Idiosyncratic risk dominates:** Approximately 76% of return variation is unexplained by the market, indicating that firm-specific factors, sector effects, and other non-market drivers play a substantial role.

3. **Betas are plausible:** The median beta of 0.646 is consistent with expectations for developed European markets, where large-cap stocks typically exhibit moderate market sensitivity.

4. **Positive alphas:** The average alpha of 0.144% suggests that stocks, on average, earn returns slightly higher than what CAPM predicts based on their beta alone.

**Conclusion:** While CAPM has moderate time-series explanatory power, it leaves substantial return variation unexplained, suggesting that additional factors beyond market beta are necessary to fully explain stock returns.

---

## 5. Cross-Sectional CAPM Test: Fama-MacBeth

### 5.1 Methodology

The Fama-MacBeth (1973) two-pass regression methodology tests the CAPM's core prediction: **Do higher beta stocks earn higher average returns?**

**Pass 1 (Time-Series):** Estimate $\beta_i$ for each stock $i$ from time-series regressions (Section 4).

**Pass 2 (Cross-Sectional):** For each month $t$, run a cross-sectional regression:
$$R_{i,t} = \gamma_{0,t} + \gamma_{1,t} \beta_i + u_{i,t}$$

where:
- $R_{i,t}$ is the return of stock $i$ in month $t$
- $\beta_i$ is the beta estimated in Pass 1
- $\gamma_{1,t}$ is the **market price of risk** in month $t$ (the reward for bearing beta risk)
- $\gamma_{0,t}$ is the intercept in month $t$ (the return on a zero-beta portfolio)

We then **average** $\gamma_0$ and $\gamma_1$ across all 59 months and compute Fama-MacBeth standard errors and t-statistics.

### 5.2 Core Results

**Table 2: Fama-MacBeth Test Results**

*(See Appendix A.2 for full table)*

| Coefficient | Average Value | t-statistic | p-value | Interpretation |
|-------------|---------------|-------------|---------|----------------|
| $\gamma_1$ (Market Price of Risk) | -0.9662 | -1.199 | 0.2356 | **NOT significant** |
| $\gamma_0$ (Intercept) | 1.5385% | 4.444 | < 0.0001 | **HIGHLY significant** |

### 5.3 Interpretation

**CAPM Prediction:**
- $\gamma_1 > 0$ and statistically significant → Higher beta stocks should earn higher returns
- $\gamma_0 \approx 0$ (if risk-free rate is correctly specified) → No abnormal return for zero-beta assets

**Our Results:**
- **$\gamma_1$ is negative and insignificant** → Beta is **not rewarded** in the cross-section. Higher beta stocks do not earn higher average returns than lower beta stocks.

- **$\gamma_0$ is strongly positive and significant** → Even a zero-beta asset earns a positive excess return (approximately 1.54% per month), suggesting that the zero-beta rate exceeds the risk-free rate.

**Conclusion:** This is a **rejection of the CAPM in the cross-section**, despite time-series betas being sensible and statistically significant. The model fails its main prediction: that beta should explain cross-sectional variation in expected returns.

### 5.4 Time Series of Market Price of Risk

**Figure 3** shows the time series of $\gamma_{1,t}$ (monthly market price of risk). The values bounce around zero with no clear positive trend, providing visual confirmation that beta is not consistently priced across months.

---

## 6. Robustness Checks

This section demonstrates that the CAPM rejection is robust across different specifications, subperiods, and sample selections.

### 6.1 Subperiod Tests

To examine whether the results are driven by specific market conditions, we split the sample into two subperiods:

- **Period A:** January 2021 – December 2022 (24 months)
- **Period B:** January 2023 – November 2025 (35 months)

**Table 3: Subperiod Fama-MacBeth Results**

*(See Appendix A.3 for full table)*

| Period | $\gamma_1$ | t-statistic | p-value | Interpretation |
|--------|------------|-------------|---------|----------------|
| Period A (2021-2022) | -1.7935 | -1.361 | 0.186 | Not significant |
| Period B (2023-2025) | -0.3989 | -0.391 | 0.698 | Not significant |

**Interpretation:** In **both subperiods**, beta is not priced. The CAPM rejection is **not driven by just one abnormal year** but appears to be a persistent feature of the data across different market regimes.

### 6.2 Country-Level Fama-MacBeth

**Table 4: Country-Level Fama-MacBeth Results**

*(See Appendix A.4 for full table)*

| Country | $\gamma_1$ | t-statistic | Interpretation |
|--------|------------|-------------|----------------|
| France | -2.4455 | -1.962 | Borderline significant **negative** |
| Sweden | -2.4246 | -1.881 | Negative (not significant) |
| Germany | -0.2331 | -0.215 | Not significant |
| Italy | 0.1413 | 0.087 | Not significant |
| Spain | 0.1405 | 0.117 | Not significant |
| Switzerland | -0.6076 | -0.554 | Not significant |
| United Kingdom | -0.7563 | -0.808 | Not significant |

**Interpretation:** In several countries (notably France and Sweden), **higher beta stocks tend to earn lower average returns**, the exact opposite of what CAPM predicts. This suggests that:
- Local factors, sector weights, or risk-aversion patterns may dominate beta effects
- Market efficiency varies across European markets
- Country-specific institutional or behavioral factors may be at play

### 6.3 Beta-Sorted Portfolios

This analysis provides **the strongest visual evidence** of CAPM failure.

We sort all stocks into **5 portfolios** by their estimated beta (P1 = lowest beta, P5 = highest beta) and compute equal-weighted monthly returns for each portfolio.

**Table 5: Beta-Sorted Portfolio Returns**

*(See Appendix A.5 for full table)*

| Portfolio | Beta | Average Return | Standard Deviation |
|-----------|------|----------------|-------------------|
| P1 (Lowest) | 0.360 | 1.20% | 4.85% |
| P2 | 0.523 | 0.95% | 4.62% |
| P3 | 0.646 | 0.89% | 4.78% |
| P4 | 0.802 | 0.72% | 5.12% |
| P5 (Highest) | 1.102 | 0.54% | 6.23% |

**Key Finding:** Higher beta portfolios have **lower** average returns, creating a **negative slope** in the beta-return relationship.

**Figure 5** plots portfolio beta against average return. Under CAPM, we would expect a clear **upward-sloping line**. Instead, the relationship is **flat to slightly negative**, providing intuitive visual confirmation that beta does not explain cross-sectional return variation.

This is **extremely strong evidence** that supports the Fama-MacBeth rejection of CAPM.

### 6.4 Clean Sample Test

We remove outliers to ensure results are not driven by a few anomalous stocks:
- Extreme betas (|$\beta$| > 5)
- Very low R² (R² < 0.05)
- Insignificant betas (p-value > 0.10)

After cleaning, we re-run the Fama-MacBeth test. The results confirm the main finding: $\gamma_1$ remains negative and insignificant, demonstrating that the CAPM rejection is **robust to sample cleaning**.

---

## 7. Economic Interpretation

### 7.1 Why Might CAPM Fail 2021-2025 in Europe?

Several factors may explain the CAPM failure during this period:

**1. Macro Regime Shifts:**
- **COVID recovery (2021):** Unprecedented fiscal and monetary stimulus created sector-specific winners and losers that were not captured by market beta alone.
- **Inflation spike (2022):** Rapid interest rate hikes by ECB, BoE, SNB, and Riksbank created winners (banks, value stocks) and losers (growth stocks, long-duration assets) based on factors beyond beta.
- **Policy normalization (2023-2025):** Transition periods often see factor rotations (value vs. growth, cyclical vs. defensive) that beta cannot capture.

**2. Sector Tilts in European Indices:**
European markets have heavy concentrations in:
- **Banks and financials** (sensitive to interest rates, not just market movements)
- **Cyclical industries** (automotive, industrials) with earnings cycles driven by macro factors
- **Defensive sectors** (utilities, consumer staples) that may outperform during volatility

These sector effects create return patterns that are orthogonal to market beta.

**3. Alternative Drivers of Returns:**

Instead of beta, returns may be driven by:
- **Size factor:** Small stocks may outperform large stocks (size premium)
- **Value factor:** Value stocks may outperform growth stocks (value premium)
- **Quality factor:** High-profitability, low-investment stocks may outperform
- **Momentum factor:** Recent winners may continue to outperform
- **Sector/specific risk:** Energy, tech, luxury, financials have distinct risk-return profiles
- **ESG and flows:** ESG considerations and ETF flows may create return patterns unrelated to beta

### 7.2 Link to Classic Findings

Our results are **consistent with** the seminal work of Fama & French (1992, 1993):

- **Fama & French (1992):** Found that beta does not explain cross-sectional variation in returns; size and book-to-market (value) do.
- **Fama & French (1993):** Proposed a three-factor model (market, size, value) that outperforms CAPM.

Our finding that $\gamma_1$ is insignificant aligns with their conclusion that **beta alone is insufficient** to explain expected returns.

More recent literature finds **mixed evidence** for CAPM, especially in:
- Short samples and turbulent periods (like 2021-2025)
- International markets where local factors matter
- Periods of monetary policy shifts

### 7.3 Implications for Equity Analysis

**As an equity analyst, relying only on $\beta$ as a risk measure is insufficient.** Our results suggest that:

1. **Multi-factor models are essential:** Consider exposures to:
   - Value vs. growth
   - Size (small vs. large)
   - Quality (profitability, investment)
   - Momentum
   - Sector factors

2. **Sector risk matters:** Beyond market beta, sector-specific risks (e.g., interest rate sensitivity for banks, commodity exposure for energy) are crucial.

3. **Balance sheet strength:** Financial leverage, liquidity, and credit quality create return patterns not captured by beta.

4. **Macro sensitivity:** Understanding how stocks respond to inflation, interest rates, and currency movements is more important than market beta alone.

5. **CAPM as a benchmark:** While CAPM fails empirically, it remains useful as a **starting point** for cost-of-equity estimation, but should be adjusted for:
   - Size premiums
   - Industry-specific risk factors
   - Company-specific considerations

---

## 8. Conclusions & Implications

### 8.1 Main Conclusions

This analysis provides **robust evidence** that the Capital Asset Pricing Model fails to explain cross-sectional variation in European stock returns during 2021-2025. The results are consistent across:
- Time-series regressions (moderate explanatory power, R² ≈ 0.24)
- Cross-sectional tests (beta not priced, $\gamma_1$ insignificant)
- Robustness checks (results hold across subperiods, countries, and sample specifications)

### 8.2 Key Findings Summary

1. **Time-Series:** CAPM has moderate explanatory power (R² ≈ 0.24), indicating that market beta explains approximately 24% of return variation.

2. **Cross-Section:** CAPM is **REJECTED** - Beta does not explain cross-sectional variation in returns. The market price of risk (γ₁) is negative and statistically insignificant.

3. **Robustness:** Results are robust across:
   - Subperiods (2021-2022 vs. 2023-2025)
   - Individual countries
   - Beta-sorted portfolios
   - Clean sample specifications

4. **Economic Interpretation:** The CAPM failure suggests that multi-factor models (Fama-French, Carhart) incorporating size, value, profitability, and momentum factors are necessary to explain expected returns.

### 8.3 Implications

**For Researchers:**
- CAPM remains a useful theoretical benchmark but fails empirically in European markets
- Multi-factor models are necessary to explain expected returns
- Country-specific factors matter in international settings

**For Practitioners:**
- Beta alone is insufficient for risk assessment and cost-of-equity estimation
- Multi-factor models should be used for portfolio construction and risk management
- Sector-specific and company-specific factors must be considered

**For Investors:**
- Higher beta does not guarantee higher returns
- Diversification across factors (size, value, quality) may be more important than market beta
- Understanding sector and company-specific risks is crucial

### 8.4 Limitations

1. **Sample Period:** The analysis covers 2021-2025, a period of significant macro volatility. Results may differ in other periods.

2. **Market Selection:** Analysis limited to seven European markets. Results may not generalize to other regions.

3. **Data Quality:** Some stocks were excluded due to data quality issues, potentially introducing selection bias.

4. **Methodology:** Simple CAPM specification. More complex models (Fama-French, Carhart) may yield different results.

### 8.5 Future Research

1. **Multi-Factor Models:** Test Fama-French three-factor and Carhart four-factor models in European markets.

2. **Extended Sample:** Analyze longer time periods to examine whether results hold across different market regimes.

3. **Additional Markets:** Extend analysis to other European and global markets.

4. **Factor Decomposition:** Decompose returns into market, size, value, quality, and momentum factors.

5. **Dynamic Betas:** Examine whether time-varying betas improve CAPM performance.

---

## 9. References

Black, F., Jensen, M. C., & Scholes, M. (1972). The capital asset pricing model: Some empirical tests. In M. C. Jensen (Ed.), *Studies in the theory of capital markets* (pp. 79-121). Praeger.

Fama, E. F., & French, K. R. (1992). The cross-section of expected stock returns. *Journal of Finance*, 47(2), 427-465.

Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636.

Lintner, J. (1965). The valuation of risk assets and the selection of risky investments in stock portfolios and capital budgets. *Review of Economics and Statistics*, 47(1), 13-37.

Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77-91.

Sharpe, W. F. (1964). Capital asset prices: A theory of market equilibrium under conditions of risk. *Journal of Finance*, 19(3), 425-442.

---

## 10. Appendices

### A.1 Table 1: CAPM Time-Series Summary by Country

*Location: `results/tables/table1_capm_timeseries_summary.csv`*

*(Full table available in CSV format)*

### A.2 Table 2: Fama-MacBeth Test Results

*Location: `results/tables/table2_fama_macbeth_results.csv`*

*(Full table available in CSV format)*

### A.3 Table 3: Subperiod Fama-MacBeth Results

*Location: `results/tables/table3_subperiod_results.csv`*

*(Full table available in CSV format)*

### A.4 Table 4: Country-Level Fama-MacBeth Results

*Location: `results/tables/table4_country_level_results.csv`*

*(Full table available in CSV format)*

### A.5 Table 5: Beta-Sorted Portfolio Returns

*Location: `results/tables/table5_beta_sorted_portfolios.csv`*

*(Full table available in CSV format)*

### A.6 Table 6: Descriptive Statistics

*Location: `results/tables/table6_descriptive_statistics.csv`*

*(Full table available in CSV format)*

### A.7 Figure Index

1. **Figure 1:** Distribution of Betas by Country
   - *Location: `results/plots/beta_distribution_by_country.png`*

2. **Figure 2:** Distribution of R² by Country
   - *Location: `results/plots/r2_distribution_by_country.png`*

3. **Figure 3:** Time Series of γ₁ (Market Price of Risk)
   - *Location: `results/plots/gamma1_timeseries.png`*

4. **Figure 4:** Beta vs Average Return Scatter
   - *Location: `results/plots/beta_vs_return_scatter.png`*

5. **Figure 5:** Beta-Sorted Portfolio Returns (KEY FIGURE)
   - *Location: `results/plots/beta_sorted_returns.png`*

6. **Figure 6:** γ₁ by Country
   - *Location: `results/plots/fm_gamma1_by_country.png`*

### A.8 Data Files

All data files are located in:
- `data/raw/` - Raw price and risk-free rate data
- `data/processed/` - Processed returns panel

### A.9 Code Repository

All analysis code is available in:
- `analysis/` - Python modules for data collection, processing, and analysis

---

**End of Report**

---

*This report was generated on December 3, 2025.*  
*For questions or additional information, please contact [Your Contact Information].*

