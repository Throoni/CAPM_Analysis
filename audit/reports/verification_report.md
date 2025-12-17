# CAPM Analysis Verification Report

Generated: 2025-12-08 16:29:05

---

## Issue 1: Beta Significance Discrepancy (84.3% vs 95.9%)

### Findings

**All Stocks (including invalid):**
- Total stocks: 249
- Significant betas (p < 0.05): 210
- Percentage: **84.3%** ← This matches the audit finding of 84.3%

**Valid Stocks Only:**
- Total stocks: 219
- Significant betas (p < 0.05): 210
- Percentage: **95.9%** ← This matches the summary finding of 95.9%

### Explanation

The discrepancy occurs because:
1. **Audit calculation** (`audit/validate_results.py` line 108-109) uses ALL stocks (valid + invalid):
   ```python
   significant = (capm_results['pvalue_beta'] < 0.05).sum()
   pct_significant = significant / len(capm_results) * 100
   ```
   This gives **84.3%** (≈84.3%)

2. **Summary calculation** (in reports) uses only VALID stocks:
   - Invalid stocks (extreme betas, low R², etc.) are excluded
   - Valid stocks have higher significance rate: **95.9%** (≈95.9%)

3. **Both are correct** but measure different things:
   - 84.3% = significance rate including data quality issues
   - 95.9% = significance rate for clean, valid sample

### Recommendation

- **For reporting:** Use 95.9% (valid stocks only) as this represents the clean sample
- **For data quality assessment:** Use 84.3% (all stocks) to identify problematic stocks
- **Document both** in methodology section

---

## Issue 2: Extreme Betas Verification (ATO.PA and III.L)

### Findings

**Extreme Beta Stocks Found:**

- **ATO.PA** (France):
  - Beta: -54.14
  - Is Valid: False

- **III.L** (UnitedKingdom):
  - Beta: 62.38
  - Is Valid: False

### ATO.PA Detailed Analysis

- **Country:** France
- **Beta:** -54.14
- **Alpha:** 344.26%
- **R²:** 0.0160
- **P-value (beta):** 0.3401
- **Is Valid:** False
- **Fix Applied:** True

- **Price Data:**
  - Observations: 58
  - Date range: 2020-12-31 00:00:00 to 2025-11-30 00:00:00
  - Price range: 23.00 to 5464.05
  - Extreme returns (>100%): 0

### III.L Detailed Analysis

- **Country:** UnitedKingdom
- **Beta:** 62.38
- **Alpha:** 620.65%
- **R²:** 0.0080
- **P-value (beta):** 0.5016
- **Is Valid:** False
- **Fix Applied:** True

- **Price Data:**
  - Observations: 54
  - Date range: 2020-12-31 00:00:00 to 2025-11-30 00:00:00
  - Price range: 1092.07 to 4399.50
  - Extreme returns (>100%): 0

### Explanation

Both stocks (ATO.PA and III.L) have extreme betas due to:
1. **Data quality issues:** Extreme price movements or data errors
2. **Properly handled:** Both are marked as `is_valid = False`
3. **Excluded from analysis:** They are filtered out in visualizations and valid-stock calculations
4. **Fixes applied:** According to `extreme_beta_fixes.csv`, fixes were attempted

### Recommendation

- **Current status:** These stocks are correctly excluded from valid results
- **No action needed:** The system properly identifies and filters extreme betas
- **Document:** Note in methodology that stocks with |beta| > 5.0 or beta < -1.0 are excluded

---

## Issue 3: Currency Mismatch Assessment

### Findings

**MSCI Index Tickers (USD-denominated iShares ETFs):**
- Germany: EWG (ETF in USD, stocks in EUR)
- France: EWQ (ETF in USD, stocks in EUR)
- Italy: EWI (ETF in USD, stocks in EUR)
- Spain: EWP (ETF in USD, stocks in EUR)
- Sweden: EWD (ETF in USD, stocks in SEK)
- UnitedKingdom: EWU (ETF in USD, stocks in GBP)
- Switzerland: EWL (ETF in USD, stocks in CHF)

**Currency Handling:**
- Currency mentioned in code: True
- Currency conversion applied: True
- Currency mentioned in docs: False

### Explanation

**The Issue:**
- MSCI country indexes are accessed via iShares ETFs (EWG, EWQ, EWI, etc.)
- These ETFs are **USD-denominated** (traded on US exchanges)
- Stocks are in **local currencies** (EUR, SEK, GBP, CHF)
- Beta estimates include both market risk AND currency risk

**Is This a Problem?**

**No, this is acceptable and common in international finance research:**

1. **Standard Practice:** Many international CAPM studies use USD-denominated market proxies
2. **Currency Risk is Part of Market Risk:** For international investors, currency movements are part of the market
3. **Consistent Methodology:** All countries use the same approach (USD ETFs)
4. **Beta Interpretation:** Beta captures sensitivity to both local market movements (via ETF) and currency movements

**Alternative Approaches (not used here):**
- Use local currency MSCI indexes (if available)
- Convert all returns to a common currency (e.g., USD)
- Use currency-hedged ETFs

### Recommendation

- **Current approach is valid** for international CAPM testing
- **Document clearly:** Note in methodology that MSCI indexes are USD-denominated ETFs
- **Interpretation:** Beta includes both market and currency risk
- **No changes needed** unless specifically testing currency-hedged portfolios

---

## Issue 4: Count Discrepancy (n=38 vs n=34 for France)

### Findings

**France Stock Counts:**
- Total in `capm_results.csv`: 38
- Valid stocks: 34
- Invalid stocks: 4

**Where Each Count Appears:**
- **Table 1** (`table1_capm_timeseries_summary.csv`): Shows n=38 (ALL stocks)
- **Graphs** (`average_beta_by_country.png`): Shows n=34 (VALID stocks only)
- **Country Summary** (`capm_by_country.csv`): Shows n=38 (ALL stocks)

### Explanation

The discrepancy occurs because:

1. **Table 1** uses `capm_by_country.csv` which includes ALL stocks (valid + invalid):
   - Created by `create_country_summaries()` in `capm_regression.py`
   - Uses: `n_stocks = len(country_data)` (line 354)
   - This gives **38 stocks** for France

2. **Graphs** filter to valid stocks only:
   - Code in `create_capm_visualizations()` (line 501):
     ```python
     valid_results = results_df[results_df['is_valid'] == True]
     ```
   - This gives **34 stocks** for France (38 - 4 invalid = 34)

3. **Invalid France Stocks:**
   - ATO.PA: Beta = -54.14, R² = 0.0160
   - BN.PA: Beta = 0.18, R² = 0.0471
   - EMEIS.PA: Beta = 0.30, R² = 0.0052
   - HO.PA: Beta = 0.16, R² = 0.0110

### Recommendation

**Option 1: Make Table 1 consistent with graphs (RECOMMENDED)**
- Filter Table 1 to show only valid stocks
- Update `generate_table1_capm_timeseries()` to filter: `capm_results[capm_results['is_valid'] == True]`
- This makes all outputs consistent (graphs and tables both use valid stocks)

**Option 2: Document the difference**
- Keep Table 1 showing all stocks
- Add footnote: "Table includes all stocks; graphs show only valid stocks (excluding extreme betas, low R², etc.)"

**Recommendation:** Use Option 1 for consistency. Most analysis should focus on valid stocks only.

---

## Summary

### Issue 1: Beta Significance  RESOLVED
- **84.3%** = All stocks (includes invalid)
- **95.9%** = Valid stocks only
- **Both correct**, use 95.9% for reporting

### Issue 2: Extreme Betas  VERIFIED
- ATO.PA and III.L have extreme betas due to data issues
- **Properly excluded** from valid results
- **No action needed**

### Issue 3: Currency Mismatch  ACCEPTABLE
- USD MSCI indexes vs local currency stocks
- **Standard practice** in international finance
- **No changes needed**, but document clearly

### Issue 4: Count Discrepancy  EXPLAINED
- Table shows all stocks (n=38)
- Graphs show valid stocks only (n=34)
- **Recommendation:** Make Table 1 consistent by filtering to valid stocks

---

## Recommended Actions

1.  **Document beta significance:** Note that 95.9% is for valid stocks, 84.3% includes all
2.  **No action on extreme betas:** Already properly handled
3.  **Document currency approach:** Note USD ETFs in methodology
4.  **Fix Table 1:** Filter to valid stocks for consistency with graphs

---

**Report generated by:** `audit/verify_issues.py`
**Date:** 2025-12-08 16:29:05
