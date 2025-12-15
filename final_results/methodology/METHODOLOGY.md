# Testing the Capital Asset Pricing Model in European Equity Markets: Methodology

## 1. Introduction

This paper presents a comprehensive empirical test of the Capital Asset Pricing Model (CAPM) using data from seven European equity markets: Germany, France, Italy, Spain, Sweden, Switzerland, and the United Kingdom. We employ both time-series and cross-sectional testing methodologies, following the seminal work of Fama and MacBeth (1973).

## 2. Theoretical Framework

### 2.1 The Capital Asset Pricing Model

The CAPM, developed by Sharpe (1964), Lintner (1965), and Mossin (1966), establishes a linear relationship between expected return and systematic risk:

$$E(R_i) = R_f + \beta_i [E(R_m) - R_f]$$

Where:
- $E(R_i)$ = Expected return on asset $i$
- $R_f$ = Risk-free rate
- $\beta_i$ = Systematic risk of asset $i$
- $E(R_m)$ = Expected return on the market portfolio

### 2.2 Testable Implications

The CAPM generates several testable predictions:

1. **Linearity**: The relationship between expected return and beta is linear
2. **Positive Risk Premium**: The market risk premium $E(R_m) - R_f > 0$
3. **Beta Sufficiency**: Beta fully explains cross-sectional return variation
4. **Zero Intercept**: Alpha ($\alpha$) should equal zero for all assets

## 3. Data and Sample

### 3.1 Sample Construction

| Parameter | Value |
|-----------|-------|
| Sample Period | January 2021 – November 2025 |
| Frequency | Monthly |
| Observations | 58-59 months per stock |
| Countries | 7 European markets |
| Total Stocks | 245 |

### 3.2 Country Distribution

| Country | Number of Stocks |
|---------|-----------------|
| Germany | 40 |
| France | 38 |
| Italy | 27 |
| Spain | 35 |
| Sweden | 27 |
| Switzerland | 16 |
| United Kingdom | 66 |

### 3.3 Data Sources

- **Stock Prices**: Yahoo Finance (adjusted close prices, local currency)
- **Market Index**: MSCI Europe Index (USD-denominated)
- **Risk-Free Rate**: German 3-month Bund yield

### 3.4 Risk-Free Rate Treatment

We use the German 3-month Bund yield as the risk-free rate for all countries to ensure consistency across the sample. This approach:

1. Provides a common benchmark across eurozone and non-eurozone countries
2. Reflects the lowest-risk sovereign rate in Europe
3. Avoids complications from country-specific credit spreads

The annual yield is converted to monthly using the compounding formula:

$$R_{f,monthly} = (1 + R_{f,annual})^{1/12} - 1$$

## 4. Methodology

### 4.1 Return Calculations

Monthly simple returns are calculated as:

$$R_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}} \times 100\%$$

Excess returns are computed as:

$$E_{i,t} = R_{i,t} - R_{f,t}$$

### 4.2 Time-Series CAPM Test (First Pass)

For each stock $i$, we estimate the market model:

$$E_{i,t} = \alpha_i + \beta_i E_{m,t} + \varepsilon_{i,t}$$

Where:
- $E_{i,t}$ = Excess return on stock $i$ in month $t$
- $E_{m,t}$ = Excess return on the market index in month $t$
- $\alpha_i$ = Jensen's alpha (abnormal return)
- $\beta_i$ = Systematic risk measure

**Estimation Details:**
- Method: Ordinary Least Squares (OLS)
- Standard Errors: White (1980) heteroscedasticity-consistent (HC0)
- Minimum observations required: 10 months

### 4.3 Fama-MacBeth Cross-Sectional Test (Second Pass)

Following Fama and MacBeth (1973), we run monthly cross-sectional regressions:

$$R_{i,t} = \gamma_{0,t} + \gamma_{1,t} \beta_i + u_{i,t}$$

For each month $t = 1, ..., T$, where $\beta_i$ is estimated from the first-pass regression.

**Fama-MacBeth Statistics:**

The time-series average of the monthly coefficients:

$$\bar{\gamma}_1 = \frac{1}{T} \sum_{t=1}^{T} \gamma_{1,t}$$

Fama-MacBeth standard error:

$$SE(\bar{\gamma}_1) = \frac{\sigma(\gamma_1)}{\sqrt{T}}$$

where $\sigma(\gamma_1)$ is the standard deviation of the monthly $\gamma_{1,t}$ estimates.

t-statistic:

$$t(\bar{\gamma}_1) = \frac{\bar{\gamma}_1}{SE(\bar{\gamma}_1)}$$

### 4.4 Hypothesis Tests

**Time-Series Test (Individual Stocks):**
- $H_0$: $\alpha_i = 0$ (CAPM holds for stock $i$)
- $H_1$: $\alpha_i \neq 0$ (CAPM rejected for stock $i$)

**Cross-Sectional Test:**
- $H_0$: $\gamma_1 = E(R_m) - R_f > 0$ (Positive risk premium)
- $H_1$: $\gamma_1 \leq 0$ (No positive risk premium)

### 4.5 Robustness Checks

1. **Subperiod Analysis**: Split sample into 2021-2022 and 2023-2025
2. **Country-Level Tests**: Separate Fama-MacBeth tests by country
3. **Beta-Sorted Portfolios**: Quintile analysis based on estimated betas
4. **Clean Sample Analysis**: Exclude stocks with extreme characteristics

## 5. Portfolio Optimization

### 5.1 Mean-Variance Framework

Following Markowitz (1952), we construct efficient portfolios by:

**Minimum-Variance Portfolio:**
$$\min_w \quad w'\Sigma w$$
$$\text{s.t.} \quad \sum_i w_i = 1$$

**Tangency Portfolio (Maximum Sharpe Ratio):**
$$\max_w \quad \frac{w'\mu - R_f}{\sqrt{w'\Sigma w}}$$
$$\text{s.t.} \quad \sum_i w_i = 1$$

### 5.2 Constraint Specifications

**Long-Only (No Short-Selling):**
- $w_i \geq 0$ for all $i$
- $\sum_i w_i = 1$

**Unconstrained (Short-Selling Allowed):**
- $-1 \leq w_i \leq 1$ for all $i$
- $\sum_i w_i = 1$
- $\sum_i |w_i| \leq 2$ (Gross exposure constraint)

### 5.3 Performance Metrics

**Sharpe Ratio:**
$$SR = \frac{E(R_p) - R_f}{\sigma_p}$$

**Diversification Ratio:**
$$DR = \frac{\sum_i w_i \sigma_i}{\sigma_p}$$

## 6. Statistical Considerations

### 6.1 Heteroscedasticity

Financial return data typically exhibits heteroscedasticity (time-varying volatility). We address this using:
- White (1980) HC0 robust standard errors in all regressions
- This ensures valid inference even when error variance is not constant

### 6.2 Errors-in-Variables Problem

The Fama-MacBeth procedure uses estimated betas as regressors, introducing measurement error. This can attenuate the estimated market premium toward zero. We acknowledge this limitation.

### 6.3 Multiple Testing

With 245 stocks tested at 5% significance, we expect ~12 false rejections by chance. We interpret alpha significance rates in this context.

## 7. Limitations

1. **Survivorship Bias**: Sample includes only stocks with complete data history
2. **Look-Ahead Bias**: Full-sample beta estimation may introduce look-ahead bias
3. **Sample Period**: 2021-2025 includes COVID recovery and high-inflation periods
4. **Market Proxy**: MSCI Europe may not perfectly represent the theoretical market portfolio

## 8. References

- Black, F., Jensen, M. C., & Scholes, M. (1972). The Capital Asset Pricing Model: Some Empirical Tests. *Studies in the Theory of Capital Markets*, 79-121.
- Fama, E. F., & French, K. R. (1992). The Cross-Section of Expected Stock Returns. *Journal of Finance*, 47(2), 427-465.
- Fama, E. F., & MacBeth, J. D. (1973). Risk, Return, and Equilibrium: Empirical Tests. *Journal of Political Economy*, 81(3), 607-636.
- Lintner, J. (1965). The Valuation of Risk Assets and the Selection of Risky Investments in Stock Portfolios and Capital Budgets. *Review of Economics and Statistics*, 47(1), 13-37.
- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77-91.
- Sharpe, W. F. (1964). Capital Asset Prices: A Theory of Market Equilibrium Under Conditions of Risk. *Journal of Finance*, 19(3), 425-442.
- White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity. *Econometrica*, 48(4), 817-838.

---

*Document prepared for CAPM Analysis Project*
*Last updated: December 2025*

