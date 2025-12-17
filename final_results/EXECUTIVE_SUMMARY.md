# Executive Summary: CAPM Analysis of European Equity Markets

## Project Overview

This project tests the Capital Asset Pricing Model (CAPM) across seven European equity markets using 245 stocks over a 58-month period (January 2021 – November 2025).

---

## Key Results at a Glance

| Test | Result | Interpretation |
|------|--------|----------------|
| **Time-Series CAPM** | Average R² = 23.5% | Typical fit for individual stocks |
| **Beta Significance** | 15% with significant α | Moderate CAPM departures |
| **Fama-MacBeth γ₁** | -0.575% (t = -0.88) | **CAPM rejected** |
| **Low-Beta Anomaly** | +1.12%/month spread | Low beta outperforms |
| **Tangency Sharpe** | 0.88 (long-only) | Strong risk-adjusted returns |

---

## Main Findings

### 1. CAPM is Partially Rejected

The Fama-MacBeth cross-sectional test shows that beta does **not** explain returns:
- Market risk premium (γ₁) = -0.575%/month
- t-statistic = -0.88 (not significant)
- **Higher beta did NOT earn higher returns**

### 2. Low-Beta Anomaly Confirmed

| Beta Quintile | Alpha |
|---------------|-------|
| Q1 (Low Beta) | +0.32%/month |
| Q5 (High Beta) | -0.80%/month |
| **Spread** | **+1.12%/month** |

Low-beta stocks outperformed by 13.5% annually on a risk-adjusted basis.

### 3. Portfolio Optimization

**Long-Only Tangency Portfolio:**
- Expected Return: 2.22%/month (26.6% annualized)
- Volatility: 2.39%/month (8.3% annualized)
- Sharpe Ratio: 0.88

---

## Investment Implications for US Investors

1. **Consider Low-Beta Strategy** in European equities
2. **Use Long-Only Constraints** for practical implementation
3. **Switzerland** offers lowest systematic risk (β = 0.76)
4. **Diversification Benefits** from European allocation

---

## Technical Highlights

-  **245 stocks** across 7 European countries
-  **Robust standard errors** (White HC0) for all regressions
-  **German 3-month Bund** as consistent risk-free rate
-  **Fama-MacBeth (1973)** two-pass methodology
-  **Mean-variance optimization** with proper constraints

---

## Deliverables

### Data Files (`/data/`)
- `capm_results.csv` - Full regression results
- `fama_macbeth_summary.csv` - Cross-sectional test statistics
- `portfolio_results_long_only.csv` - Optimized portfolio metrics
- `efficient_frontier_*.csv` - Frontier data points

### Figures (`/figures/`)
- `efficient_frontier_no_shortselling.png` - Long-only frontier
- `efficient_frontier_with_shortselling.png` - Unconstrained frontier
- `beta_distribution_by_country.png` - Beta analysis
- `gamma1_timeseries.png` - Risk premium over time

### Documentation (`/methodology/` and `/reports/`)
- `METHODOLOGY.md` - Complete methodology paper
- `EMPIRICAL_RESULTS.md` - Detailed results report

---

## Conclusion

The CAPM fails to explain cross-sectional stock returns in European markets during 2021-2025. The low-beta anomaly provides an opportunity for enhanced risk-adjusted returns. These findings are consistent with decades of academic research questioning the empirical validity of CAPM.

---

*Analysis completed December 2025*

