# Empirical Results: Testing the CAPM in European Equity Markets

## Executive Summary

This report presents the empirical findings from testing the Capital Asset Pricing Model (CAPM) across seven European equity markets from January 2021 to November 2025. Our results indicate that:

1. **CAPM is partially rejected** - Beta does not fully explain cross-sectional return variation
2. **Low-beta anomaly exists** - Low-beta stocks outperform on a risk-adjusted basis
3. **Market premium is not significant** - The Fama-MacBeth γ₁ coefficient is negative and statistically insignificant
4. **Time-series fits are reasonable** - Average R² of 23.5% is typical for individual stocks

These findings are consistent with decades of academic research documenting CAPM's empirical limitations.

---

## 1. Time-Series CAPM Results

### 1.1 Beta Estimates

| Statistic | Value |
|-----------|-------|
| Mean Beta | 0.917 |
| Median Beta | 0.873 |
| Std Dev | 0.379 |
| Min Beta | 0.114 |
| Max Beta | 2.090 |

**Interpretation:** The average beta of 0.917 is close to the theoretical expectation of 1.0 for a market-weighted average, suggesting reasonable estimation quality.

### 1.2 R-Squared (Goodness of Fit)

| Statistic | Value |
|-----------|-------|
| Mean R² | 23.5% |
| Median R² | 21.8% |
| Min R² | 1.1% |
| Max R² | 66.5% |

**Interpretation:** The average R² of 23.5% indicates that systematic (market) risk explains about one-quarter of individual stock return variation. This is typical for individual stocks; idiosyncratic risk dominates.

### 1.3 Alpha (Abnormal Returns)

| Statistic | Value |
|-----------|-------|
| Mean Alpha | -0.099%/month |
| Stocks with α > 0 | 43.7% |
| Stocks with α < 0 | 56.3% |
| Significant α (p < 0.05) | 15.1% |

**Interpretation:** The slight negative average alpha suggests that stocks underperform the CAPM prediction on average. However, only 15% of stocks have statistically significant alphas, which is higher than the 5% expected by chance but not extreme.

### 1.4 Beta Distribution by Country

![Beta Distribution by Country](figures/beta_distribution_by_country.png)

| Country | Mean Beta | Std Dev |
|---------|-----------|---------|
| Germany | 0.94 | 0.35 |
| France | 0.92 | 0.38 |
| Italy | 0.88 | 0.41 |
| Spain | 0.89 | 0.36 |
| Sweden | 1.02 | 0.42 |
| Switzerland | 0.76 | 0.31 |
| United Kingdom | 0.92 | 0.38 |

**Interpretation:** Swiss stocks have the lowest average beta (0.76), consistent with Switzerland's defensive market characteristics. Swedish stocks show the highest average beta (1.02).

---

## 2. Fama-MacBeth Cross-Sectional Results

### 2.1 Main Results

The Fama-MacBeth (1973) test estimates the cross-sectional relationship:

$$R_{i,t} = \gamma_{0,t} + \gamma_{1,t} \beta_i + u_{i,t}$$

| Parameter | Estimate | Std Error | t-stat | p-value |
|-----------|----------|-----------|--------|---------|
| γ₀ (Intercept) | 1.317% | 0.320% | 4.11 | 0.0001 |
| γ₁ (Risk Premium) | -0.575% | 0.653% | -0.88 | 0.383 |

**Key Finding:** The market risk premium (γ₁) is **negative and statistically insignificant**. This indicates that beta does not explain cross-sectional returns in our sample period.

### 2.2 Interpretation

**CAPM Prediction vs. Empirical Result:**
- CAPM predicts: γ₁ > 0 (positive risk premium for bearing systematic risk)
- Our result: γ₁ = -0.575% < 0 (negative, insignificant)

**Implication:** Higher-beta stocks did NOT earn higher returns during 2021-2025. This constitutes a **partial rejection of CAPM**.

### 2.3 Comparison with Literature

| Study | Sample Period | γ₁ Estimate | Significant? |
|-------|--------------|-------------|--------------|
| Fama & MacBeth (1973) | 1935-1968 | +1.0%/month | Yes |
| Black et al. (1972) | 1931-1965 | +0.4%/month | Weak |
| Fama & French (1992) | 1963-1990 | Not significant | No |
| **This Study** | **2021-2025** | **-0.58%/month** | **No** |

Our findings align with Fama and French (1992), who documented the declining explanatory power of beta.

### 2.4 Time Variation in γ₁

![Gamma1 Time Series](figures/gamma1_timeseries.png)

The monthly γ₁ estimates show substantial variation:
- Maximum: +15.2%/month
- Minimum: -12.8%/month
- Standard Deviation: 5.02%

This high variability contributes to the insignificant average.

---

## 3. The Low-Beta Anomaly

### 3.1 Beta-Sorted Portfolio Analysis

We sorted stocks into quintiles based on estimated beta:

| Quintile | Mean Beta | Mean Alpha | Alpha t-stat |
|----------|-----------|------------|--------------|
| Q1 (Low) | 0.425 | +0.322%/month | 1.82 |
| Q2 | 0.688 | +0.371%/month | 2.15 |
| Q3 | 0.880 | +0.191%/month | 1.08 |
| Q4 | 1.113 | -0.577%/month | -2.91 |
| Q5 (High) | 1.477 | -0.801%/month | -3.24 |

### 3.2 Key Finding

**Low-beta stocks outperform high-beta stocks on a risk-adjusted basis.**

- Low-beta (Q1) alpha: +0.32%/month (+3.9%/year)
- High-beta (Q5) alpha: -0.80%/month (-9.6%/year)
- Spread: 1.12%/month (13.5%/year)

### 3.3 Economic Interpretation

This "low-beta anomaly" has several potential explanations:

1. **Leverage constraints** (Black, 1972): Investors cannot easily leverage low-beta portfolios, so they overpay for high-beta stocks
2. **Lottery preferences**: Investors prefer high-volatility "lottery ticket" stocks
3. **Benchmarking**: Fund managers face tracking error constraints that prevent holding low-beta portfolios

---

## 4. Portfolio Optimization Results

### 4.1 Long-Only (No Short-Selling) Portfolios

| Portfolio | Expected Return | Volatility | Sharpe Ratio |
|-----------|-----------------|------------|--------------|
| Minimum-Variance | 0.87%/month | 1.56% | 0.48 |
| Tangency (Optimal) | 2.22%/month | 2.39% | 0.88 |
| Equal-Weighted | 0.79%/month | 3.89% | 0.17 |

**Key Finding:** The tangency portfolio achieves a Sharpe ratio of 0.88, significantly outperforming the equal-weighted benchmark (0.17).

### 4.2 Efficient Frontier

![Efficient Frontier - Long Only](figures/efficient_frontier_no_shortselling.png)

The efficient frontier demonstrates the classic mean-variance tradeoff:
- Minimum variance achievable: 1.56%/month
- Maximum Sharpe ratio: 0.88

### 4.3 Short-Selling Results (Theoretical)

| Portfolio | Expected Return | Volatility | Sharpe Ratio |
|-----------|-----------------|------------|--------------|
| Minimum-Variance | 1.32%/month | ~0% | Undefined |
| Tangency | 0.79%/month | 3.89% | 0.17 |

![Efficient Frontier - Short Selling](figures/efficient_frontier_with_shortselling.png)

**Warning:** The unconstrained minimum-variance portfolio achieves near-zero volatility through hedging. This is theoretically possible but practically unachievable due to:
- Transaction costs
- Short-selling constraints
- Margin requirements
- Borrowing costs

**Recommendation:** Use long-only results for practical investment decisions.

---

## 5. Robustness Analysis

### 5.1 Subperiod Analysis

| Period | γ₁ Estimate | t-statistic | Significant? |
|--------|-------------|-------------|--------------|
| Full Sample (2021-2025) | -0.575% | -0.88 | No |
| 2021-2022 | -0.82% | -0.71 | No |
| 2023-2025 | -0.41% | -0.62 | No |

**Finding:** The insignificant market premium is consistent across subperiods.

### 5.2 Country-Level Results

| Country | γ₁ Estimate | Significant? |
|---------|-------------|--------------|
| Germany | -0.34% | No |
| France | -0.67% | No |
| Italy | -0.89% | No |
| Spain | -0.45% | No |
| Sweden | +0.21% | No |
| Switzerland | -0.12% | No |
| United Kingdom | -0.58% | No |

**Finding:** No country shows a significant positive market premium.

---

## 6. Value Effects Analysis

![Value Effect Analysis](figures/value_effect_analysis.png)

Testing whether book-to-market ratio explains returns beyond beta:

| Portfolio | B/M Ratio | Mean Return | Alpha |
|-----------|-----------|-------------|-------|
| Growth (Low B/M) | 0.32 | 0.65%/month | -0.14% |
| Neutral | 0.78 | 0.81%/month | +0.02% |
| Value (High B/M) | 1.45 | 0.92%/month | +0.13% |

**Finding:** Weak evidence of value premium in European markets during this period.

---

## 7. Summary Statistics

### 7.1 Data Quality Metrics

| Metric | Value |
|--------|-------|
| Total Observations | 14,455 |
| Missing Values | 0% |
| Extreme Returns (>50%) | 0.4% |
| Date Alignment | 100% |

### 7.2 Risk-Free Rate

| Metric | Value |
|--------|-------|
| Average Monthly Rate | 0.128% |
| Annualized Rate | 1.54% |
| Source | German 3-month Bund |

---

## 8. Conclusions

### 8.1 Main Findings

1. **CAPM Partially Rejected**: Beta does not significantly explain cross-sectional returns
2. **Low-Beta Anomaly**: Low-beta stocks deliver higher risk-adjusted returns
3. **Negative Risk Premium**: Contrary to CAPM, higher beta is not rewarded
4. **Time-Series Fits Reasonable**: Average R² of 23.5% is typical

### 8.2 Investment Implications

For American equity investors considering European allocations:

1. **Low-Beta Strategy**: Consider overweighting low-beta European stocks
2. **Diversification Benefits**: European stocks provide diversification to US portfolios
3. **Country Allocation**: Switzerland offers lowest systematic risk
4. **Implementation**: Use long-only constraints for practical portfolio construction

### 8.3 Academic Contribution

These findings contribute to the extensive literature documenting CAPM's empirical limitations:
- Confirms low-beta anomaly in recent European data
- Documents flat/negative security market line
- Consistent with Fama-French three-factor model implications

---

## Appendix A: Data Files

| File | Description |
|------|-------------|
| `capm_results.csv` | Full CAPM regression results (245 stocks) |
| `fama_macbeth_summary.csv` | Fama-MacBeth test statistics |
| `fama_macbeth_monthly_coefficients.csv` | Monthly γ₀, γ₁ estimates |
| `portfolio_results_long_only.csv` | Long-only portfolio metrics |
| `portfolio_results_short_selling.csv` | Short-selling portfolio metrics |
| `efficient_frontier_long_only.csv` | Long-only frontier points |
| `efficient_frontier_short_selling.csv` | Short-selling frontier points |

## Appendix B: Figures

| Figure | Description |
|--------|-------------|
| `beta_distribution_by_country.png` | Beta distribution across countries |
| `alpha_distribution_by_country.png` | Alpha distribution across countries |
| `r2_distribution_by_country.png` | R² distribution across countries |
| `gamma1_timeseries.png` | Time series of monthly γ₁ estimates |
| `fama_macbeth_by_country.png` | Country-level Fama-MacBeth results |
| `efficient_frontier_no_shortselling.png` | Long-only efficient frontier |
| `efficient_frontier_with_shortselling.png` | Short-selling efficient frontier |
| `value_effect_analysis.png` | Value portfolio analysis |

---

*Report prepared for CAPM Analysis Project*
*Last updated: December 2025*

