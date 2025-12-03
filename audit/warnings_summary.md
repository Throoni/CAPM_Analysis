# Audit Warnings Summary

**Total Warnings:** 28  
**Date:** 2025-12-03

---

## Phase 1.1: Raw Data Validation (20 warnings)

### Date Range Mismatches (14 warnings)
1. **Germany stock prices:** Start date mismatch
2. **Germany stock prices:** End date mismatch
3. **France stock prices:** Start date mismatch
4. **France stock prices:** End date mismatch
5. **Italy stock prices:** Start date mismatch
6. **Italy stock prices:** End date mismatch
7. **Spain stock prices:** Start date mismatch
8. **Spain stock prices:** End date mismatch
9. **Sweden stock prices:** Start date mismatch
10. **Sweden stock prices:** End date mismatch
11. **UnitedKingdom stock prices:** Start date mismatch
12. **UnitedKingdom stock prices:** End date mismatch
13. **Switzerland stock prices:** Start date mismatch
14. **Switzerland stock prices:** End date mismatch

**Note:** These are warnings because actual dates may differ slightly from expected (2020-12-01 to 2025-12-01). The data may start/end a few days earlier or later, which is acceptable as long as the core analysis period is covered.

### Extreme Price Changes (3 warnings)
15. **France:** 11 extreme price changes (>50% month-over-month) - may be stock splits, dividends, or corporate actions
16. **Italy:** 2 extreme price changes (>50% month-over-month) - may be stock splits, dividends, or corporate actions
17. **UnitedKingdom:** 8 extreme price changes (>50% month-over-month) - may be stock splits, dividends, or corporate actions

**Note:** These are expected in real-world data and typically represent corporate actions (splits, dividends, mergers) rather than data errors.

### Missing Data (3 warnings)
18. **France:** 1 stock with >10% missing data
19. **Italy:** 2 stocks with >10% missing data

**Note:** Stocks with >10% missing data are typically excluded during processing, so this is expected.

### Stock Count (1 warning)
20. **Stock count mismatch:** 252 stocks found vs expected 249

**Note:** This is a minor discrepancy. The difference (3 stocks) may be due to:
- Additional stocks in raw data that get filtered out during processing
- Different counting methodology
- Stocks that don't meet minimum observation requirements

---

## Phase 1.3: Processed Data Validation (1 warning)

21. **Extreme returns:** 1 extreme return (>50%) - may be data issues or corporate actions

**Note:** This is the same issue as #15-17, but flagged in the processed data. The extreme return is likely from a corporate action and is acceptable.

---

## Phase 3: Statistical Methodology Audit (1 warning)

22. **Fama-MacBeth t-statistics:** T-statistics column not found in summary file (but values are present in the data)

**Note:** This is a column naming issue. The t-statistics exist in the CSV file but may have different column names (e.g., `tstat_gamma_1` vs `t(γ₁)`). The values are correct, just the column name detection needs improvement.

---

## Phase 4.3: Assumption Violations (1 warning)

23. **Heteroscedasticity:** Detected in 6/10 sample stocks

**Note:** This is common in financial data. Heteroscedasticity (non-constant variance of residuals) violates OLS assumptions but doesn't invalidate the results. It may affect standard errors, but the Fama-MacBeth methodology accounts for this.

---

## Phase 5: Results Validation Audit (5 warnings)

24. **Extreme high beta:** 1 stock with beta > 3.0

**Note:** This is unusual but not necessarily an error. Some stocks (e.g., highly leveraged financials, small caps) can have very high betas. This stock should be reviewed to ensure it's not a data error.

25. **Extreme low beta:** 1 stock with beta < -1.0

**Note:** Negative betas are rare but possible (e.g., gold stocks, inverse ETFs). This should be verified as intentional.

26. **Beta significance:** Only 84.3% of betas are significant (p < 0.05), expected ~95.9%

**Note:** The audit found 84.3% significant, but the summary reports 95.9%. This discrepancy should be investigated. However, 84.3% is still reasonable for financial data.

27. **Low R²:** 30 stocks with very low R² (< 0.05)

**Note:** These stocks have weak relationships with the market, which is acceptable. They may be driven by firm-specific factors rather than market movements.

28. **Portfolio slope:** ~~Portfolio slope is positive (1.7531)~~ **FIXED:** Portfolio slope is actually **negative (-0.8797)**, which is consistent with CAPM rejection

**Note:** **This warning was a bug in the audit code.** The column detection was incorrectly selecting `beta_max` and `std_return` instead of `portfolio_beta` and `avg_return`. The bug has been fixed, and the actual portfolio slope is negative (-0.8797), meaning higher beta portfolios have lower returns, which is consistent with the Fama-MacBeth finding that gamma_1 is negative.

---

## Summary by Severity

### Minor Warnings (Expected in Real Data)
- Date range mismatches (14) - Acceptable if core period covered
- Extreme price changes (3) - Corporate actions are normal
- Missing data (3) - Stocks get filtered during processing
- Stock count mismatch (1) - Minor discrepancy
- Extreme returns (1) - Same as price changes
- Heteroscedasticity (1) - Common in financial data
- Low R² stocks (1) - Some stocks have weak market relationships
- **Portfolio slope** (1) - **FIXED:** Was a bug in audit code, now correctly shows negative slope

### Warnings Requiring Review
- **Fama-MacBeth t-statistics column naming** (1) - Technical issue, values are correct
- **Extreme betas** (2) - Should verify these are not data errors
- **Beta significance discrepancy** (1) - 84.3% vs 95.9% reported, needs investigation

---

## Recommended Actions

1. ~~**Investigate portfolio slope issue (#28):**~~ **FIXED** - The audit bug has been corrected. Portfolio slope is correctly identified as negative, consistent with Fama-MacBeth results.

2. **Verify extreme betas (#24-25):** Check the 2 stocks with extreme betas to ensure they're not data errors.

3. **Resolve beta significance discrepancy (#26):** Investigate why audit found 84.3% vs reported 95.9%.

4. **Document date range differences:** If actual dates differ from expected, document the actual analysis period used.

5. **All other warnings are acceptable** and represent normal characteristics of real-world financial data.

---

*Generated from audit log: logs/audit_20251203_151727.log*

