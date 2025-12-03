# CAPM_Analysis
Empirical CAPM testing across 7 European markets

## Risk-Free Rate Data Sources

This project requires actual 3-month government bond yields as risk-free rates. The system supports multiple data sources with automatic fallback:

### Data Source Priority (tried in order):
1. **ECB API** - For EUR countries (free, no API key required)
2. **FRED API** - For all countries (free, requires API key)
3. **WRDS** - For academic users with WRDS access
4. **Yahoo Finance** - Limited availability (fallback)
5. **Placeholder** - Last resort (0.1% monthly)

### Setting Up FRED API (Recommended)

FRED API is the most reliable source for international government bond yields.

1. **Get a free FRED API key:**
   - Visit: https://fred.stlouisfed.org/docs/api/api_key.html
   - Sign in or create a free account
   - Request an API key (instant approval)

2. **Set the API key as environment variable:**
   ```bash
   export FRED_API_KEY="your_api_key_here"
   ```
   
   Or add to your shell profile (`~/.zshrc` or `~/.bashrc`):
   ```bash
   echo 'export FRED_API_KEY="your_api_key_here"' >> ~/.zshrc
   source ~/.zshrc
   ```

3. **Verify installation:**
   ```bash
   pip install fredapi
   python3 -c "from fredapi import Fred; print('FRED API ready')"
   ```

### Setting Up WRDS (Optional - Limited)

**Note:** WRDS requires Two-Factor Authentication (Duo Mobile) and is difficult to use in non-interactive/automated environments. **FRED API is strongly recommended instead.**

If you still want to try WRDS:

1. **Set up Duo Mobile 2FA** on your WRDS account (required)

2. **Set WRDS credentials as environment variables:**
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

### Current Status

The system will automatically try data sources in order. If no API keys are set, it will use placeholder values (0.1% monthly) with warnings.

To get actual risk-free rates:
- **Best option**: Set up FRED API key (free, 5 minutes)
- **Alternative**: Use WRDS if you have access
- **Fallback**: System will use placeholder with clear warnings

### Testing Data Sources

Test your setup:
```bash
python3 -m analysis.riskfree_helper
```

This will test fetching risk-free rates for all countries using available data sources.
