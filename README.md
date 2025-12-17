# CAPM_Analysis

[![CI](https://github.com/Throoni/CAPM_Analysis/workflows/CI/badge.svg)](https://github.com/Throoni/CAPM_Analysis/actions)
[![Audit](https://github.com/Throoni/CAPM_Analysis/workflows/Audit/badge.svg)](https://github.com/Throoni/CAPM_Analysis/actions)
[![Tests](https://img.shields.io/badge/tests-19%2F19%20passing-brightgreen)](tests/)
[![CodeRabbit Reviews](https://img.shields.io/coderabbit/prs/github/Throoni/CAPM_Analysis?utm_source=oss&utm_medium=github&utm_campaign=Throoni%2FCAPM_Analysis&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)](https://coderabbit.ai)

##  Mission: Investment Analysis for American Equity Investors

Empirical CAPM testing across 7 European markets with **actionable investment recommendations** for American equity investors. This analysis identifies opportunities in European equity markets, focusing on currency risk, country allocation, factor-based strategies, and portfolio construction.

**Key Finding:** CAPM is rejected in European markets - beta does not explain expected returns, creating opportunities for factor-based and active strategies.

 **[View Investment Recommendations](results/reports/Investment_Recommendations_US_Investor.md)** |  **[Execution Guide](docs/EXECUTION_GUIDE.md)** |  **[Methodology](docs/methodology/METHODOLOGY.md)**

## Risk-Free Rate Data Sources

This project requires actual 3-month government bond yields as risk-free rates. The system supports multiple data sources with automatic fallback:

### Data Source Priority (tried in order):
1. **CSV Files** - Processed risk-free rate files (primary source, available for all countries)
2. **ECB API** - For EUR countries (free, no API key required)
3. **FRED API** - For all countries (free, requires API key)
4. **WRDS** - For academic users with WRDS access
5. **Yahoo Finance** - Limited availability (fallback)

**Note:** System requires real data - CSV files are available for all 7 countries in `data/raw/riskfree_rates/`. No placeholder values are used.

### Setting Up Credentials (Recommended)

**Option 1: Using .env file (Easiest)**

1. **Copy the example file:**
   ```bash
   cp env.example .env
   ```

2. **Edit `.env` with your credentials:**
   ```bash
   # Open .env in your editor
   nano .env  # or use your preferred editor
   ```

3. **Fill in your credentials:**
   ```env
   FRED_API_KEY=your_fred_api_key_here
   WRDS_USERNAME=your_wrds_username  # Optional
   WRDS_PASSWORD=your_wrds_password  # Optional
   ```

4. **The `.env` file is automatically loaded** - no need to export variables manually!

**Option 2: Environment Variables (Manual)**

Set environment variables directly:

```bash
export FRED_API_KEY="your_api_key_here"
export WRDS_USERNAME="your_wrds_username"  # Optional
export WRDS_PASSWORD="your_wrds_password"  # Optional
```

Or add to your shell profile (`~/.zshrc` or `~/.bashrc`):
```bash
echo 'export FRED_API_KEY="your_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

### Setting Up FRED API (Recommended)

FRED API is the most reliable source for international government bond yields.

1. **Get a free FRED API key:**
   - Visit: https://fred.stlouisfed.org/docs/api/api_key.html
   - Sign in or create a free account
   - Request an API key (instant approval)

2. **Add to `.env` file or export as environment variable** (see above)

3. **Verify installation:**
   ```bash
   pip install fredapi python-dotenv
   python3 -c "from fredapi import Fred; print('FRED API ready')"
   ```

### Setting Up WRDS (Optional - Limited)

**Note:** WRDS requires Two-Factor Authentication (Duo Mobile) and is difficult to use in non-interactive/automated environments. **FRED API is strongly recommended instead.**

If you still want to try WRDS:

1. **Set up Duo Mobile 2FA** on your WRDS account (required)

2. **Add WRDS credentials to `.env` file:**
   ```env
   WRDS_USERNAME=your_wrds_username
   WRDS_PASSWORD=your_wrds_password
   ```

   Or set as environment variables:
   ```bash
   export WRDS_USERNAME="your_wrds_username"
   export WRDS_PASSWORD="your_wrds_password"
   ```

3. **Or use .pgpass file:**
   - Create/edit `~/.pgpass` file:
     ```
     wrds-pgdata.wharton.upenn.edu:9737:wrds:your_username:your_password
     ```
   - Set permissions:
     ```bash
     chmod 600 ~/.pgpass
     ```

**Limitation:** WRDS connection may fail in automated/non-interactive scripts due to 2FA requirements. FRED API is more reliable for automated use.

### Security Notes

- **Never commit `.env` file** - it's already in `.gitignore`
- **Share `env.example`** - this template file is safe to commit
- **Credentials are optional** - the code works without them (uses fallback data sources)
- **All credentials are loaded from environment variables** - no hardcoded secrets in code

### Current Status

The system automatically tries data sources in order, starting with CSV files (available for all countries). **Real data is required** - the system will fail with a clear error if no data source succeeds (no placeholder fallback).

Risk-free rate CSV files are available for all 7 countries in `data/raw/riskfree_rates/`. To use additional sources:
- **Best option**: Set up FRED API key (free, 5 minutes) for automatic updates
- **Alternative**: Use WRDS if you have access
- **Primary**: CSV files are already available and will be used automatically

### Testing Data Sources

Test your setup:
```bash
python3 -m analysis.data.riskfree_helper
```

This will test fetching risk-free rates for all countries using available data sources.

---

## Quick Start: Running the Full Analysis

**For detailed step-by-step instructions, see [Execution Guide](docs/EXECUTION_GUIDE.md).**

### Run All Analyses

```bash
# 1. Process returns
python3 -m analysis.core.returns_processing

# 2. CAPM regression
python3 -m analysis.core.capm_regression

# 3. Fama-MacBeth test
python3 -m analysis.core.fama_macbeth

# 4. Robustness checks
python3 -m analysis.core.robustness_checks

# 5. Portfolio optimization
python3 -m analysis.extensions.portfolio_optimization

# 6. Value effects
python3 -m analysis.extensions.value_effects

# 7. Market cap analysis
python3 -m analysis.extensions.market_cap_analysis

# 8. Portfolio recommendation
python3 -m analysis.extensions.portfolio_recommendation

# 9. Investment recommendations (US investors)
python3 -m analysis.extensions.investment_recommendations
```

### Key Outputs

- **Investment Recommendations:** `results/reports/Investment_Recommendations_US_Investor.md`
- **Portfolio Recommendation:** `results/reports/Portfolio_Recommendation.md`
- **CAPM Analysis Report:** `results/reports/main/CAPM_Analysis_Report.md`
- **Executive Summary:** `results/reports/main/Executive_Summary.md`

---

## Important Notes

- **Real Data Only:** System requires real data - no placeholder values are used. CSV files are available for all 7 countries.
- **Fail Hard:** If data is missing, system will fail with clear error messages (no silent placeholders).
- **Investment Focus:** Analysis is designed to provide actionable recommendations for American equity investors.
