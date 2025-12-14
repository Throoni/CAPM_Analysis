import os
from dataclasses import dataclass

"""
config.py

Central configuration for paths and global analysis parameters.
Stage 1: file system + high-level settings.
Later stages will extend this with tickers, indices, and RF proxies.
"""


# -------------------------------------------------------------------
# PROJECT ROOT & PATHS
# -------------------------------------------------------------------

# Absolute path to project root (folder that contains .git, data/, analysis/, etc.)
# config.py is now in analysis/utils/, so go up two levels
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_RAW_PRICES_DIR = os.path.join(DATA_RAW_DIR, "prices")
DATA_RAW_RISKFREE_DIR = os.path.join(DATA_RAW_DIR, "riskfree_rates")
DATA_RAW_EXCHANGE_DIR = os.path.join(DATA_RAW_DIR, "exchange_rates")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DATA_METADATA_DIR = os.path.join(DATA_DIR, "metadata")
DATA_BASELINES_DIR = os.path.join(DATA_DIR, "baselines")

# Results directories
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# New organized structure
RESULTS_PORTFOLIO_OPT_DIR = os.path.join(RESULTS_DIR, "portfolio_optimization")
RESULTS_PORTFOLIO_LONG_ONLY_DIR = os.path.join(RESULTS_PORTFOLIO_OPT_DIR, "long_only")
RESULTS_PORTFOLIO_SHORT_SELLING_DIR = os.path.join(RESULTS_PORTFOLIO_OPT_DIR, "short_selling")
RESULTS_PORTFOLIO_REALISTIC_SHORT_DIR = os.path.join(RESULTS_PORTFOLIO_OPT_DIR, "realistic_short")

RESULTS_CAPM_ANALYSIS_DIR = os.path.join(RESULTS_DIR, "capm_analysis")
RESULTS_CAPM_TIMESERIES_DIR = os.path.join(RESULTS_CAPM_ANALYSIS_DIR, "time_series")
RESULTS_CAPM_TIMESERIES_FIGURES_DIR = os.path.join(RESULTS_CAPM_TIMESERIES_DIR, "figures")
RESULTS_CAPM_CROSSSECTIONAL_DIR = os.path.join(RESULTS_CAPM_ANALYSIS_DIR, "cross_sectional")
RESULTS_CAPM_CROSSSECTIONAL_FIGURES_DIR = os.path.join(RESULTS_CAPM_CROSSSECTIONAL_DIR, "figures")

RESULTS_FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
RESULTS_FIGURES_CAPM_DIR = os.path.join(RESULTS_FIGURES_DIR, "capm")
RESULTS_FIGURES_FM_DIR = os.path.join(RESULTS_FIGURES_DIR, "fama_macbeth")
RESULTS_FIGURES_PORTFOLIO_DIR = os.path.join(RESULTS_FIGURES_DIR, "portfolio")
RESULTS_FIGURES_VALUE_EFFECTS_DIR = os.path.join(RESULTS_FIGURES_DIR, "value_effects")

RESULTS_REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
RESULTS_REPORTS_MAIN_DIR = os.path.join(RESULTS_REPORTS_DIR, "main")
RESULTS_REPORTS_INVESTMENT_DIR = os.path.join(RESULTS_REPORTS_DIR, "investment")
RESULTS_REPORTS_THESIS_DIR = os.path.join(RESULTS_REPORTS_DIR, "thesis")

# Legacy paths (for backward compatibility - will be removed after code updates)
RESULTS_REPORTS_DATA_DIR = os.path.join(RESULTS_REPORTS_DIR, "data")
RESULTS_REPORTS_CONSTRAINED_DIR = os.path.join(RESULTS_REPORTS_DIR, "constrained_only")
RESULTS_DATA_DIR = os.path.join(RESULTS_DIR, "data")
RESULTS_TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
RESULTS_BASELINES_DIR = os.path.join(RESULTS_DIR, "baselines")
RESULTS_FIGURES_CONSTRAINED_DIR = os.path.join(RESULTS_DIR, "figures_constrained")

# Logs directory
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Ensure these directories exist (idempotent)
for path in [
    DATA_DIR,
    DATA_RAW_DIR,
    DATA_RAW_PRICES_DIR,
    DATA_RAW_RISKFREE_DIR,
    DATA_RAW_EXCHANGE_DIR,
    DATA_PROCESSED_DIR,
    DATA_METADATA_DIR,
    DATA_BASELINES_DIR,
    RESULTS_DIR,
    RESULTS_PORTFOLIO_OPT_DIR,
    RESULTS_PORTFOLIO_LONG_ONLY_DIR,
    RESULTS_PORTFOLIO_SHORT_SELLING_DIR,
    RESULTS_PORTFOLIO_REALISTIC_SHORT_DIR,
    RESULTS_CAPM_ANALYSIS_DIR,
    RESULTS_CAPM_TIMESERIES_DIR,
    RESULTS_CAPM_TIMESERIES_FIGURES_DIR,
    RESULTS_CAPM_CROSSSECTIONAL_DIR,
    RESULTS_CAPM_CROSSSECTIONAL_FIGURES_DIR,
    RESULTS_FIGURES_DIR,
    RESULTS_FIGURES_CAPM_DIR,
    RESULTS_FIGURES_FM_DIR,
    RESULTS_FIGURES_PORTFOLIO_DIR,
    RESULTS_FIGURES_VALUE_EFFECTS_DIR,
    RESULTS_REPORTS_DIR,
    RESULTS_REPORTS_MAIN_DIR,
    RESULTS_REPORTS_INVESTMENT_DIR,
    RESULTS_REPORTS_THESIS_DIR,
    # Legacy paths
    RESULTS_REPORTS_DATA_DIR,
    RESULTS_REPORTS_CONSTRAINED_DIR,
    RESULTS_DATA_DIR,
    RESULTS_TABLES_DIR,
    RESULTS_BASELINES_DIR,
    RESULTS_FIGURES_CONSTRAINED_DIR,
    LOGS_DIR,
]:
    os.makedirs(path, exist_ok=True)


# -------------------------------------------------------------------
# GLOBAL ANALYSIS SETTINGS (from your methodology)
# -------------------------------------------------------------------

@dataclass(frozen=True)
class AnalysisSettings:
    """
    Global CAPM analysis parameters (fixed by your methodology).
    """
    start_date: str = "2020-12-01"
    end_date: str = "2025-12-01"
    return_frequency: str = "M"   # Monthly
    return_type: str = "simple"   # Simple (arithmetic) returns
    rf_frequency: str = "M"       # Monthly risk-free series
    window_length: int = 60       # 60 months for beta estimation
    rolling_window: int = 36      # Optional 36-month rolling betas

ANALYSIS_SETTINGS = AnalysisSettings()


# -------------------------------------------------------------------
# COUNTRY-CURRENCY SKELETON
# -------------------------------------------------------------------

@dataclass(frozen=True)
class CountryConfig:
    name: str
    currency: str


COUNTRIES = {
    "Germany":        CountryConfig(name="Germany",        currency="EUR"),
    "France":         CountryConfig(name="France",         currency="EUR"),
    "Italy":          CountryConfig(name="Italy",          currency="EUR"),
    "Spain":          CountryConfig(name="Spain",          currency="EUR"),
    "Sweden":         CountryConfig(name="Sweden",         currency="SEK"),
    "UnitedKingdom":  CountryConfig(name="United Kingdom", currency="GBP"),
    "Switzerland":    CountryConfig(name="Switzerland",    currency="CHF"),
}

# MSCI Country Index Tickers (iShares ETFs tracking MSCI indices)
# NOTE: These are kept for reference but are no longer used in the main analysis.
# The analysis now uses MSCI Europe (IEUR) for all countries.
MSCI_INDEX_TICKERS = {
    "Germany":        "EWG",
    "France":         "EWQ",
    "Italy":          "EWI",
    "Spain":          "EWP",
    "Sweden":         "EWD",
    "UnitedKingdom": "EWU",
    "Switzerland":    "EWL",
}

# MSCI Europe Index Ticker (iShares Core MSCI Europe ETF)
# This is used as the market proxy for all countries in the analysis
# Ticker: IEUR (tracks MSCI Europe Index, includes large-, mid-, and small-cap European stocks)
# Note: While the ETF trades in USD, it tracks EUR-denominated European stocks,
# making it appropriate for European investor analysis. The underlying returns
# reflect European market movements in their local currencies.
MSCI_EUROPE_TICKER = "IEUR"

# Minimum observations required (59 months of returns from 60 months of prices)
MIN_OBSERVATIONS = 59

# -------------------------------------------------------------------
# RISK-FREE RATE DATA SOURCE CONFIGURATION
# -------------------------------------------------------------------

# Preferred data source order (will try in this order until one succeeds)
# Note: CSV files are checked first in get_riskfree_rate() before this list
# System requires real data - no placeholder fallback
RISKFREE_SOURCE_ORDER = ["csv", "ecb", "fred", "wrds", "yfinance"]

# FRED API Series Codes for 3-month government bond yields
# Format: IR3TTS01[COUNTRY]M156N where COUNTRY is country code
FRED_SERIES_CODES = {
    "Germany": "IR3TTS01DEM156N",      # Germany 3-month Treasury Bill
    "Sweden": "IR3TTS01SEM156N",      # Sweden 3-month Treasury Bill
    "UnitedKingdom": "IR3TTS01GBM156N", # UK 3-month Treasury Bill
    "Switzerland": "IR3TTS01CHM156N",  # Switzerland 3-month Treasury Bill
    # Note: EUR countries (France, Italy, Spain) use Germany's series
}

# ECB Statistical Data Warehouse Series Keys
# Format: ECB SDMX series keys for 3-month government bond yields
ECB_SERIES_KEYS = {
    "Germany": "FM.B.U2.EUR.4F.BB.U3_3M.HSTA",  # Germany 3-month government bond yield
    # Note: EUR countries (France, Italy, Spain) use Germany's series
}

# Yahoo Finance tickers for government bonds (limited availability)
YFINANCE_BOND_TICKERS = {
    "Germany": "^IRX",  # US 13-week Treasury (fallback only)
    "Sweden": "^IRX",   # US 13-week Treasury (fallback only)
    "UnitedKingdom": "^IRX",  # US 13-week Treasury (fallback only)
    "Switzerland": "^IRX",   # US 13-week Treasury (fallback only)
    # Note: Yahoo Finance has limited international bond data
}

# WRDS credentials (set via environment variables for security)
# WRDS_USERNAME and WRDS_PASSWORD should be set in environment
# Or use WRDS config file: ~/.wrdsrc


