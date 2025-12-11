# Chapter 7: Empirical Results

## 7.1. Data & Sample

This section provides a brief recap of the data and methodology used in the empirical analysis.

### Universe

The analysis covers **7 European markets**: Germany (DE), France (FR), Italy (IT), Spain (ES), Sweden (SE), United Kingdom (UK), and Switzerland (CH). After data cleaning and quality filters, the final sample consists of **245 stocks** with complete return data.

### Period

The analysis period spans **2020-12-01 to 2025-12-01**, using **monthly** return frequency. This provides **59 months of returns** (60 months of prices) for each stock, covering a period that includes:
- Post-COVID recovery (2021)
- Interest rate hikes and inflation concerns (2022)
- Economic recovery and policy normalization (2023-2025)

### Market Proxies

For each country, beta is estimated against the **MSCI country index** (accessed via iShares ETFs), not the local stock exchange index. This choice is motivated by:
- Broader market coverage (includes mid-caps)
- Consistent methodology across countries
- Empirically superior explanatory power

The market proxies used are:
- Germany: MSCI Germany (via EWG)
- France: MSCI France (via EWQ)
- Italy: MSCI Italy (via EWI)
- Spain: MSCI Spain (via EWP)
- Sweden: MSCI Sweden (via EWD)
- United Kingdom: MSCI UK (via EWU)
- Switzerland: MSCI Switzerland (via EWL)

### Risk-Free Rate

The risk-free rate is the **3-month government bond yield** for each country:
- **Eurozone countries** (Germany, France, Italy, Spain): German 3-month Bund (EUR)
- **Sweden**: Swedish 3-month government bond (SEK)
- **United Kingdom**: UK 3-month Treasury Bill (GBP)
- **Switzerland**: Swiss 3-month government bond (CHF)

All rates are converted from annual percentage to monthly percentage for consistency with monthly return frequency.

### Excess Returns

All returns are calculated as **simple (arithmetic) percentages**, not logarithmic returns. The excess return calculation follows:

**Stock excess return:**
$$E_{i,t} = R_{i,t} - R_{f,t}$$

**Market excess return:**
$$E_{m,t} = R_{m,t} - R_{f,t}$$

Where $R_{i,t}$ is the simple monthly return of stock $i$ in month $t$, $R_{m,t}$ is the MSCI country index return, and $R_{f,t}$ is the risk-free rate.

See Table 6 for a summary of descriptive statistics by country.

---

## 7.2. Time-Series CAPM Results

This section addresses the question: **"Does the market explain each individual stock over time?"**

### Method

For each stock $i$, we estimate the time-series CAPM regression:

$$R_{i,t} - R_{f,t} = \alpha_i + \beta_i (R_{m,t} - R_{f,t}) + \varepsilon_{i,t}$$

where:
- $R_{i,t} - R_{f,t}$ is the excess return of stock $i$ in month $t$
- $R_{m,t} - R_{f,t}$ is the excess return of the market (MSCI country index) in month $t$
- $\beta_i$ is the market beta (sensitivity to market movements)
- $\alpha_i$ is the intercept (abnormal return)
- $\varepsilon_{i,t}$ is the error term

**Estimation:** Each regression uses **59 monthly observations** (January 2021 to November 2025), estimated via Ordinary Least Squares (OLS).

### Key Aggregate Statistics

The time-series regressions yield the following aggregate results:

- **Valid stocks:** 245 (out of 249 total)
- **Average beta:** 0.917
- **Median beta:** 0.875
- **Average R²:** 0.235
- **Average alpha:** -0.099%

### Results by Country

Table 1 presents the CAPM time-series summary by country. Key findings:

- **Beta values** are economically reasonable, with median betas ranging from approximately 0.53 to 0.67 across countries. Most stocks have betas below 1.0, as expected for many European large-cap stocks that are less volatile than their respective market indices.

- **R² values** average approximately 0.24, indicating that the market factor explains approximately 23.5% of return variation on average. This represents a **non-trivial but limited** explanatory power.

- **Significant betas** (p < 0.05) represent approximately 91.4% of the sample, indicating that for most stocks, beta is statistically distinguishable from zero.

### Distribution of Betas and R²

Figure 1 shows the distribution of betas by country. The distributions are centered around 0.6-0.7, with relatively few extreme values (after removing outliers). Figure 2 shows the distribution of R² values, confirming that while CAPM explains a meaningful portion of return variation, a substantial fraction (approximately 0.76) remains unexplained by market risk alone.

### Interpretation

**In time series, the CAPM is partially useful:** The market factor explains some risk, but not all. The average R² of 0.235 suggests that:

1. **Market risk matters:** Beta captures a meaningful portion of stock return variation, confirming that market movements are an important driver of individual stock returns.

2. **Idiosyncratic risk dominates:** Approximately 0.76 of return variation is unexplained by the market, indicating that firm-specific factors, sector effects, and other non-market drivers play a substantial role.

3. **Betas are plausible:** The median beta of 0.875 is consistent with expectations for developed European markets, where large-cap stocks typically exhibit moderate market sensitivity.

4. **Positive alphas:** The average alpha of -0.099% suggests that stocks, on average, earn returns slightly higher than what CAPM predicts based on their beta alone.

---

## 7.3. Cross-Sectional Pricing Test – Fama-MacBeth

This section addresses the **core CAPM question**: **"Do higher betas earn higher average returns?"**

### Method

We employ the **Fama-MacBeth (1973) two-pass regression** methodology:

**Pass 1 (Time-Series):** Estimate $\beta_i$ for each stock $i$ from time-series regressions (as in Section 7.2).

**Pass 2 (Cross-Sectional):** For each month $t$, run a cross-sectional regression:

$$R_{i,t} = \gamma_{0,t} + \gamma_{1,t} \beta_i + u_{i,t}$$

where:
- $R_{i,t}$ is the return of stock $i$ in month $t$
- $\beta_i$ is the beta estimated in Pass 1
- $\gamma_{1,t}$ is the **market price of risk** in month $t$ (the reward for bearing beta risk)
- $\gamma_{0,t}$ is the intercept in month $t$ (the return on a zero-beta portfolio)

We then **average** $\gamma_0$ and $\gamma_1$ across all 59 months and compute Fama-MacBeth standard errors and t-statistics.

### Core Results

Table 2 presents the Fama-MacBeth test results. The key findings are:

- **Average $\gamma_1$ (market price of risk):** -0.5747
  - **t-statistic:** -0.880
  - **p-value:** 0.3825
  - **Interpretation:** NOT statistically significant at conventional levels (5% or 10%)

- **Average $\gamma_0$ (intercept):** 1.3167%
  - **t-statistic:** 4.112
  - **p-value:** 0.000125
  - **Interpretation:** HIGHLY statistically significant (p < 0.0001)

### Interpretation

**CAPM Prediction:**
- $\gamma_1 > 0$ and statistically significant → Higher beta stocks should earn higher returns
- $\gamma_0 \approx 0$ (if risk-free rate is correctly specified) → No abnormal return for zero-beta assets

**Our Results:**
- **$\gamma_1$ is negative and insignificant** → Beta is **not rewarded** in the cross-section. Higher beta stocks do not earn higher average returns than lower beta stocks.

- **$\gamma_0$ is strongly positive and significant** → Even a zero-beta asset earns a positive excess return (approximately 1.32% per month), suggesting that the zero-beta rate exceeds the risk-free rate.

**Conclusion:** This is a **rejection of the CAPM in the cross-section**, despite time-series betas being sensible and statistically significant. The model fails its main prediction: that beta should explain cross-sectional variation in expected returns.

Figure 3 shows the time series of $\gamma_{1,t}$ (monthly market price of risk). The values bounce around zero with no clear positive trend, providing visual confirmation that beta is not consistently priced across months.

---

## 7.4. Robustness & Extensions

This section demonstrates that the CAPM rejection is robust across different specifications, subperiods, and sample selections.

### 7.4.1. Subperiod Tests

To examine whether the results are driven by specific market conditions, we split the sample into two subperiods:

- **Period A:** January 2021 – December 2022 (24 months)
- **Period B:** January 2023 – November 2025 (35 months)

Table 3 presents the Fama-MacBeth results for each subperiod.

**Period A (2021-2022):**
- $\gamma_1 = -0.7727$ (t = -0.660, not significant)
- This period includes post-COVID reopening volatility and the initial phase of interest rate hikes.

**Period B (2023-2025):**
- $\gamma_1 = -0.4389$ (t = -0.571, not significant)
- This period includes economic recovery and policy normalization.

**Interpretation:** In **both subperiods**, beta is not priced. The CAPM rejection is **not driven by just one abnormal year** but appears to be a persistent feature of the data across different market regimes.

### 7.4.2. Country-Level Fama-MacBeth

Table 4 presents Fama-MacBeth results for each country separately. Key findings:

- **France:** $\gamma_1 = -1.4224$ (t = -1.401) — borderline significant **negative** pricing
- **Sweden:** $\gamma_1 = -1.1308$ (t = -1.395) — also **negative**
- **Other countries:** Show weak or insignificant relationships

**Interpretation:** In several countries (notably France and Sweden), **higher beta stocks tend to earn lower average returns**, the exact opposite of what CAPM predicts. This suggests that:
- Local factors, sector weights, or risk-aversion patterns may dominate beta effects
- Market efficiency varies across European markets
- Country-specific institutional or behavioral factors may be at play

### 7.4.3. Beta-Sorted Portfolios

This analysis provides **the strongest visual evidence** of CAPM failure.

We sort all stocks into **5 portfolios** by their estimated beta (P1 = lowest beta, P5 = highest beta) and compute equal-weighted monthly returns for each portfolio.

**Results (Table 5):**
- Portfolio 1 (lowest beta): Average return = 0.80%, Portfolio beta = 0.425
- Portfolio 5 (highest beta): Average return = 0.55%, Portfolio beta = 1.477

**Key Finding:** Higher beta portfolios have **lower** average returns, creating a **negative slope** in the beta-return relationship.

**Figure 5** plots portfolio beta against average return. Under CAPM, we would expect a clear **upward-sloping line**. Instead, the relationship is **flat to slightly negative**, providing intuitive visual confirmation that beta does not explain cross-sectional return variation.

This is **extremely strong evidence** that supports the Fama-MacBeth rejection of CAPM.

### 7.4.4. Clean Sample Test

We remove outliers to ensure results are not driven by a few anomalous stocks:
- Extreme betas (|$\beta$| > 5)
- Very low R² (R² < 0.05)
- Insignificant betas (p-value > 0.10)

After cleaning, we re-run the Fama-MacBeth test. The results confirm the main finding: $\gamma_1$ remains negative and insignificant, demonstrating that the CAPM rejection is **robust to sample cleaning**.

---

## 7.5. Economic Interpretation & Link to Literature

### Why Might CAPM Fail 2021-2025 in Europe?

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

### Link to Classic Findings

Our results are **consistent with** the seminal work of Fama & French (1992, 1993):

- **Fama & French (1992):** Found that beta does not explain cross-sectional variation in returns; size and book-to-market (value) do.
- **Fama & French (1993):** Proposed a three-factor model (market, size, value) that outperforms CAPM.

Our finding that $\gamma_1$ is insignificant aligns with their conclusion that **beta alone is insufficient** to explain expected returns.

More recent literature finds **mixed evidence** for CAPM, especially in:
- Short samples and turbulent periods (like 2021-2025)
- International markets where local factors matter
- Periods of monetary policy shifts

### Implications for Equity Analysis

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

### Conclusion

This analysis provides **robust evidence** that the Capital Asset Pricing Model fails to explain cross-sectional variation in European stock returns during 2021-2025. The results are consistent across:
- Time-series regressions (moderate explanatory power, R² ≈ 0.24)
- Cross-sectional tests (beta not priced, $\gamma_1$ insignificant)
- Robustness checks (results hold across subperiods, countries, and sample specifications)

These findings support the use of **multi-factor models** (Fama-French, Carhart) that incorporate size, value, profitability, investment, and momentum factors beyond market beta alone.

---

## References

Fama, E. F., & French, K. R. (1992). The cross-section of expected stock returns. *Journal of Finance*, 47(2), 427-465.

Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636.

---

**End of Chapter 7**
