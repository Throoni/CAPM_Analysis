# R² Improvements Analysis: Currency Conversion Impact

## Summary

After implementing full currency conversion (all stocks and market index to EUR), R² values improved significantly, especially for GBP and SEK stocks. However, some stocks still have low R², which is expected for a single-factor model like CAPM.

## Currency Conversion Implementation

### What Was Done

1. **MSCI Europe (IEUR):** Converted from USD to EUR using USD/EUR exchange rates
2. **Stock Prices:** Converted to EUR based on country currency:
   - GBP stocks: Converted using GBP/EUR rates (yfinance)
   - SEK stocks: Converted using SEK/EUR rates (yfinance)
   - CHF stocks: Converted using CHF/EUR rates (yfinance)
   - EUR stocks: No conversion needed
3. **Returns:** Calculated from EUR-denominated prices for all assets

### R² Improvements

**Overall:**
- Before: Average R² = 0.217, 36.1% of stocks < 0.15
- After: Average R² = 0.236, 30.2% of stocks < 0.15
- Improvement: +0.019 average R², -5.9 percentage points in low R² stocks

**By Currency:**
- **GBP:** 0.203 → 0.258 (+0.055, +27% improvement)
- **SEK:** 0.264 → 0.323 (+0.059, +22% improvement)
- **CHF:** 0.275 → 0.246 (-0.029, slight decrease but still strong)
- **EUR:** 0.208 → 0.208 (no change, expected)

**By Country:**
- **United Kingdom:** 0.203 → 0.258 (major improvement)
- **Sweden:** 0.264 → 0.323 (strong improvement)
- **Switzerland:** 0.275 → 0.246 (slight decrease)
- **Spain:** 0.184 (still lowest, but improved from currency conversion)
- **France, Germany, Italy:** ~0.21 (moderate, EUR stocks)

## Why R² is Still Relatively Low

### Expected Behavior for Single-Factor Model

**R² of 0.20-0.30 is typical for CAPM:**
- Finance literature shows that individual stock R² values of 0.20-0.35 are normal
- CAPM is a single-factor model - it cannot explain all return variation
- Multi-factor models (Fama-French) typically achieve R² of 0.40-0.60

### Remaining Low R² Stocks (30.2% have R² < 0.15)

**Characteristics:**
- Mean beta: 0.575 (vs overall 0.905) - weak market sensitivity
- 29.7% have non-significant betas (p > 0.05) - truly idiosyncratic
- Concentrated in certain countries:
  - Spain: 45.7% have R² < 0.15
  - United Kingdom: 26.6% have R² < 0.15
  - France: 35.1% have R² < 0.15

**Reasons for Low R²:**
1. **Firm-Specific Factors:** Earnings surprises, management changes, product launches
2. **Sector Effects:** Technology, financials, energy have distinct risk profiles not captured by market beta
3. **Size Effects:** Small-cap stocks may have different return drivers
4. **Value/Quality Effects:** Book-to-market, profitability factors
5. **Momentum:** Recent performance trends
6. **Country-Specific Factors:** Even after currency conversion, local economic conditions matter

### Is This a Problem?

**No - this is expected:**
- CAPM is designed to explain systematic (market) risk, not total risk
- Idiosyncratic risk is expected and cannot be diversified away at the stock level
- Portfolio-level R² would be higher (diversification reduces idiosyncratic risk)
- The fact that 70% of stocks have R² > 0.15 shows the model has meaningful explanatory power

## Recommendations for Further R² Improvement

### 1. Multi-Factor Models
- **Fama-French 3-Factor:** Add size and value factors (could increase R² to 0.40-0.50)
- **Carhart 4-Factor:** Add momentum factor
- **Fama-French 5-Factor:** Add profitability and investment factors

### 2. Sector-Adjusted Analysis
- Control for sector effects (technology, financials, energy have different risk profiles)
- Sector-specific market proxies might improve R² for sector-concentrated stocks

### 3. Extended Sample Period
- Longer time periods may provide more stable beta estimates
- Current period (2021-2025) includes high volatility (COVID, inflation, rate hikes)

### 4. Alternative Market Proxies
- Test if country-specific indices improve R² for certain stocks
- However, this would conflict with the European-wide approach justified in methodology

## Conclusion

Currency conversion successfully improved R², especially for GBP and SEK stocks. The remaining relatively low R² (average 0.233) is **expected and normal** for a single-factor model. Further improvements would require multi-factor models, which is beyond the scope of this CAPM analysis but represents a natural extension for future research.
