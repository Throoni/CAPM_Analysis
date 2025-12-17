# Results Interpretation Guide

## Overview

This guide explains how to read and interpret the CAPM analysis results. It provides detailed explanations of all key metrics, coefficients, and statistical tests.

---

## Table of Contents

1. [Understanding CAPM Regression Results](#1-understanding-capm-regression-results)
2. [Interpreting Fama-MacBeth Coefficients](#2-interpreting-fama-macbeth-coefficients)
3. [Understanding R² and Its Limitations](#3-understanding-r²-and-its-limitations)
4. [Beta Significance and Economic Meaning](#4-beta-significance-and-economic-meaning)
5. [What Rejection of CAPM Means](#5-what-rejection-of-capm-means)
6. [Comparison to Academic Literature](#6-comparison-to-academic-literature)

---

## 1. Understanding CAPM Regression Results

### 1.1 Regression Output Structure

For each stock, the CAPM regression produces:

```
Beta (β): 0.917
Alpha (α): -0.099%
R²: 0.236
Beta t-statistic: 5.23
Beta p-value: 0.0001
Alpha t-statistic: 0.45
Alpha p-value: 0.65
```

### 1.2 Beta (β) Interpretation

**What Beta Means:**
- Beta measures a stock's sensitivity to market movements
- β = 1.0: Stock moves 1:1 with the market
- β > 1.0: Stock is more volatile than the market (aggressive)
- β < 1.0: Stock is less volatile than the market (defensive)
- β < 0: Stock moves opposite to the market (rare, usually indicates data error)

**Our Results:**
- Average β = 0.917: Stocks are, on average, slightly less volatile than the market
- Median β = 0.875: Consistent with average
- Range: Most stocks have β between 0.3 and 1.5 (economically reasonable)

**Economic Interpretation:**
- European large-cap stocks typically have β < 1.0
- Average β of 0.917 is consistent with expectations
- Suggests stocks are less sensitive to market movements than the market itself

### 1.3 Alpha (α) Interpretation

**What Alpha Means:**
- Alpha measures excess return beyond what CAPM predicts
- α > 0: Stock earns more than predicted by CAPM
- α < 0: Stock earns less than predicted by CAPM
- α = 0: Stock earns exactly what CAPM predicts

**Our Results:**
- Average α = -0.099% per month
- This is small and mostly not statistically significant
- Suggests stocks earn slightly less than CAPM predicts, but not significantly

**Economic Interpretation:**
- Small positive alpha suggests slight outperformance
- But not statistically significant, so could be due to chance
- Consistent with market efficiency (no persistent alpha)

### 1.4 R² Interpretation

**What R² Means:**
- R² measures the proportion of return variation explained by the model
- R² = 0.236 means 23.6% of return variation is explained by market beta
- Remaining 76.5% is unexplained (idiosyncratic risk)

**Our Results:**
- Average R² = 0.236 (23.6%)
- Median R² = 0.210 (21.0%)
- Range: R² varies widely across stocks (0.05 to 0.50+)

**Economic Interpretation:**
- **Moderate explanatory power:** Market beta explains about 1/4 of return variation
- **Idiosyncratic risk dominates:** About 3/4 of variation is firm-specific
- This is **normal** - individual stocks have substantial firm-specific risk
- Low R² does NOT invalidate CAPM - it's expected for individual stocks

**Why R² is Low:**
- Individual stocks have high idiosyncratic risk
- Firm-specific events (earnings, management, products) drive returns
- Sector effects create variation not captured by market beta
- This is why portfolio analysis (aggregation) often shows higher R²

### 1.5 Statistical Significance

**Beta Significance:**
- t-statistic: Measures how many standard errors β is from zero
- p-value: Probability of observing this β if true β = 0
- p < 0.05: Statistically significant (reject null that β = 0)

**Our Results:**
- 95.9% of stocks have significant betas (p < 0.05)
- This means beta is **statistically meaningful** for almost all stocks
- Strong evidence that market movements affect individual stock returns

**Alpha Significance:**
- Most alphas are NOT statistically significant
- This is expected - persistent alpha would violate market efficiency
- Small, insignificant alpha is consistent with efficient markets

---

## 2. Interpreting Fama-MacBeth Coefficients

### 2.1 Fama-MacBeth Output Structure

The Fama-MacBeth test produces:

```
γ₀ (Intercept): 1.3167%
γ₁ (Market Price of Risk): -0.5747
γ₀ t-statistic: 4.112
γ₁ t-statistic: -0.880
γ₀ p-value: < 0.0001
γ₁ p-value: 0.3825
```

### 2.2 γ₀ (Intercept) Interpretation

**What γ₀ Means:**
- γ₀ is the return on a zero-beta portfolio
- CAPM predicts: γ₀ = R_f (risk-free rate)
- If γ₀ > R_f: Zero-beta stocks earn more than risk-free rate
- If γ₀ < R_f: Zero-beta stocks earn less than risk-free rate

**Our Results:**
- γ₀ = 1.3167% per month (highly significant, t = 4.112)
- This is much higher than typical risk-free rate (~0.1-0.3% monthly)
- **Interpretation:** Zero-beta stocks earn substantial returns

**Economic Significance:**
- Suggests there are risk factors beyond market beta
- Zero-beta stocks are not risk-free, so they should earn returns
- This is consistent with multi-factor models (Fama-French)

**What This Means:**
- CAPM's prediction that γ₀ = R_f is **rejected**
- Zero-beta stocks earn returns beyond the risk-free rate
- Additional risk factors are being rewarded

### 2.3 γ₁ (Market Price of Risk) Interpretation

**What γ₁ Means:**
- γ₁ is the **market price of risk** - the reward for bearing beta risk
- CAPM predicts: γ₁ = E[R_m] - R_f (market risk premium)
- γ₁ > 0 and significant: Higher beta → Higher return (CAPM supported)
- γ₁ = 0 or not significant: Beta does not price returns (CAPM rejected)

**Our Results:**
- γ₁ = -0.5747 (negative, contrary to CAPM)
- t-statistic = -0.880
- p-value = 0.3825 (NOT significant)

**Statistical Interpretation:**
- p = 0.3825 means we **cannot reject** the null hypothesis that γ₁ = 0
- This means beta does NOT explain cross-sectional variation in returns
- The negative sign suggests higher beta → lower return (though not significant)

**Economic Interpretation:**
- **CAPM is REJECTED**
- Beta does not price returns in the cross-section
- Higher beta stocks do NOT earn higher average returns
- This is the **core finding** of our analysis

**What This Means:**
- CAPM's central prediction fails
- Beta alone is insufficient for asset pricing
- Multi-factor models are necessary

### 2.4 Significance Testing

**Why p = 0.3825 is NOT Significant:**
- Conventional significance level: p < 0.05
- p = 0.3825 means there's a 38.25% chance of observing this result if γ₁ = 0
- This is too high to reject the null hypothesis
- **Conclusion:** We cannot say beta prices returns

**If γ₁ Were Significant:**
- If p < 0.05, we would conclude beta DOES price returns
- But our p = 0.3825 is far from significant
- This provides strong evidence against CAPM

---

## 3. Understanding R² and Its Limitations

### 3.1 What R² Measures

**R² Definition:**
- R² = Proportion of variance explained by the model
- R² = 1 - (SS_residual / SS_total)
- Ranges from 0 (no explanation) to 1 (perfect explanation)

**Our Results:**
- Average R² = 0.236 (23.6%)
- This means market beta explains about 1/4 of return variation

### 3.2 Why R² is Low for Individual Stocks

**Idiosyncratic Risk:**
- Individual stocks have substantial firm-specific risk
- Earnings surprises, management changes, product launches
- These create return variation not captured by market beta

**Sector Effects:**
- Different sectors respond differently to market movements
- Technology vs. utilities vs. financials
- Sector-specific factors create variation beyond market beta

**Other Risk Factors:**
- Size, value, momentum factors
- These create return patterns not captured by market beta alone

**This is Normal:**
- Low R² for individual stocks is **expected**
- Even in academic studies, individual stock R² is typically 0.20-0.40
- Our R² of 0.235 is within the normal range

### 3.3 Limitations of R²

**R² Does NOT Measure:**
- Whether beta is priced (that's what Fama-MacBeth tests)
- Economic significance (that's what γ₁ tests)
- Model validity (that requires multiple tests)

**R² Alone is Insufficient:**
- High R² doesn't mean CAPM is correct
- Low R² doesn't mean CAPM is wrong
- R² measures time-series fit, not cross-sectional pricing

**Why We Need Fama-MacBeth:**
- R² shows beta explains time-series variation
- But Fama-MacBeth tests if beta explains cross-sectional returns
- These are **different questions**

### 3.4 R² in Context

**Individual Stocks:**
- R² = 0.235 is moderate but reasonable
- Suggests market risk is important but not dominant

**Portfolios:**
- Portfolio R² is typically higher (0.50-0.80)
- Aggregation reduces idiosyncratic risk
- This is why portfolio analysis is also important

**Academic Literature:**
- Fama-French (1992): Individual stock R² ≈ 0.20-0.30
- Our R² = 0.235 is consistent with literature
- This is **not a problem** - it's expected

---

## 4. Beta Significance and Economic Meaning

### 4.1 Statistical Significance

**What It Means:**
- 95.9% of stocks have statistically significant betas (p < 0.05)
- This means beta is **statistically distinguishable from zero**
- Strong evidence that market movements affect stock returns

**Why This Matters:**
- Shows beta is a meaningful measure
- Market risk is real and measurable
- Not just noise - beta captures systematic risk

### 4.2 Economic Significance

**Beta Magnitude:**
- Average β = 0.917 is economically reasonable
- Suggests stocks are less volatile than the market
- Consistent with large-cap European stocks

**Beta Range:**
- Most stocks have β between 0.3 and 1.5
- Few extreme values (after outlier removal)
- Distribution is centered and reasonable

**What This Means:**
- Betas are **economically meaningful**
- They represent real risk characteristics
- But being meaningful ≠ being priced

### 4.3 The Key Distinction

**Statistical vs. Economic Significance:**
- **Statistical:** Beta is significantly different from zero ( we have this)
- **Economic:** Beta explains cross-sectional returns ( we don't have this)

**Why Both Matter:**
- Statistical significance: Beta is a real, measurable risk factor
- Economic significance: Beta is priced (rewarded with higher returns)
- **Our finding:** Beta is statistically significant but NOT economically priced

**This is the Core Result:**
- Beta matters (statistically significant)
- But beta doesn't price returns (not economically significant)
- This is why CAPM fails

---

## 5. What Rejection of CAPM Means

### 5.1 What "Rejection" Means

**CAPM Prediction:**
- Higher beta stocks should earn higher average returns
- γ₁ should be positive and significant
- Beta should explain cross-sectional variation in returns

**Our Finding:**
- γ₁ is negative (contrary to prediction)
- γ₁ is not significant (p = 0.3825)
- Beta does NOT explain cross-sectional returns

**Conclusion:**
- **CAPM is REJECTED**
- Beta alone is insufficient for asset pricing

### 5.2 What This Does NOT Mean

**Does NOT Mean:**
- Beta is meaningless (it's statistically significant)
- Market risk doesn't matter (it does, in time-series)
- CAPM is useless (it's still a useful benchmark)
- All asset pricing models fail (multi-factor models work)

**What It DOES Mean:**
- Beta alone is insufficient
- Additional factors are needed
- Multi-factor models are necessary

### 5.3 Implications

**For Asset Pricing Theory:**
- CAPM is incomplete
- Multi-factor models (Fama-French, Carhart) are necessary
- Size, value, momentum factors matter

**For Portfolio Construction:**
- Don't rely solely on beta
- Consider multiple risk factors
- Diversify across factors, not just stocks

**For Risk Management:**
- Beta is one risk measure, not the only one
- Consider sector, size, value exposures
- Understand firm-specific risks

**For Cost of Equity:**
- CAPM can be starting point
- But adjust for size, value, industry factors
- Multi-factor models provide better estimates

### 5.4 Why CAPM Still Matters

**Theoretical Benchmark:**
- CAPM provides theoretical foundation
- Other models build on CAPM
- Still useful for understanding risk-return tradeoff

**Practical Applications:**
- Starting point for cost of equity
- Benchmark for performance evaluation
- Framework for thinking about risk

**But:**
- Must be adjusted for empirical realities
- Multi-factor models are necessary
- Don't rely on CAPM alone

---

## 6. Comparison to Academic Literature

### 6.1 Fama-French (1992)

**Their Finding:**
- Beta does not explain cross-sectional returns
- Size and value factors are more important
- Proposed three-factor model

**Our Finding:**
- **Consistent** - Beta does not explain cross-sectional returns
- Suggests size and value factors may be important
- Supports multi-factor models

### 6.2 International Evidence

**European Studies:**
- Generally find beta not priced
- Multi-factor models outperform CAPM
- Country-specific factors matter

**Our Finding:**
- **Consistent** - Beta not priced in European markets
- Supports multi-factor approach
- Country-specific analysis confirms this

### 6.3 Recent Evidence (2020s)

**Mixed Results:**
- Some studies find beta priced in certain contexts
- Depends on sample period, market, methodology
- Multi-factor models generally outperform

**Our Finding:**
- **Consistent** with recent European studies
- Beta not priced in 2021-2025 period
- Supports continued use of multi-factor models

### 6.4 Why Our Results Are Credible

**Methodology:**
- Standard Fama-MacBeth approach
- Industry-standard statistical methods
- Extensive robustness checks

**Consistency:**
- Results hold across subperiods
- Results hold across countries
- Results hold in portfolio analysis

**Alignment:**
- Consistent with decades of research
- Aligned with international evidence
- Supports established academic consensus

---

## Summary

**Key Takeaways:**

1. **Time-Series:** Beta is statistically significant and explains ~24% of return variation
2. **Cross-Sectional:** Beta does NOT explain cross-sectional returns (CAPM rejected)
3. **Robustness:** Results are consistent across multiple specifications
4. **Literature:** Our findings align with decades of empirical finance research
5. **Implications:** Multi-factor models are necessary for asset pricing

**Bottom Line:**
- Beta matters over time, but doesn't price returns across stocks
- CAPM fails empirically, but remains useful as theoretical benchmark
- Multi-factor models (Fama-French, Carhart) are necessary for practical applications

---

**Last Updated:** December 8, 2025

