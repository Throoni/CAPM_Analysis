"""
Mean-Variance Portfolio Optimization Module.

This module implements Markowitz (1952) mean-variance portfolio optimization
to construct efficient frontiers and identify optimal portfolios.

Key functionalities:
    - Efficient frontier calculation (both constrained and unconstrained)
    - Minimum-variance portfolio identification
    - Tangency (maximum Sharpe ratio) portfolio calculation
    - Capital Market Line construction
    - Diversification benefit analysis

Portfolio constraints:
    - Long-only (no short-selling): w_i >= 0 for all i
    - Short-selling allowed: -1 <= w_i <= 1 with gross exposure limits
    - Realistic constraints: Transaction costs, borrowing costs, position limits

The optimization uses scipy.optimize.minimize with SLSQP and trust-constr methods.
Numerical stability is ensured through:
    - Pre-computed bounds and constraints
    - Gradient-based optimization with multiple restarts
    - Post-optimization constraint verification

References
----------
Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.

Sharpe, W. F. (1966). Mutual Fund Performance. Journal of Business, 39(1), 119-138.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import platform

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from analysis.utils.config import (
    DATA_PROCESSED_DIR,
    RESULTS_DATA_DIR,
    RESULTS_FIGURES_DIR,
    RESULTS_REPORTS_DIR,
    RESULTS_REPORTS_CONSTRAINED_DIR,
    RESULTS_FIGURES_CONSTRAINED_DIR,
    RESULTS_PORTFOLIO_LONG_ONLY_DIR,
    RESULTS_PORTFOLIO_SHORT_SELLING_DIR,
    RESULTS_PORTFOLIO_REALISTIC_SHORT_DIR,
    COUNTRIES,
    ANALYSIS_SETTINGS
)

logger = logging.getLogger(__name__)

# Set style
if HAS_SEABORN:
    sns.set_style("whitegrid")
else:
    plt.style.use('default')
    # Set grid manually to match seaborn's whitegrid style
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    logger.warning("seaborn not available, using matplotlib default styles with manual grid")
plt.rcParams['figure.figsize'] = (12, 8)

# Maximum gross exposure for short-selling portfolios
# 2.0 = 200% total exposure (e.g., 150% long + 50% short)
# This prevents infinite leverage in unconstrained portfolios
MAX_GROSS_EXPOSURE = 2.0

# Realistic short-selling constraints
# These constraints prevent near-perfect hedging and create economically achievable portfolios
TRANSACTION_COST_RATE = 0.001  # 0.1% per trade (bid-ask spread + commission)
INITIAL_MARGIN_REQUIREMENT = 0.50  # 50% initial margin for short positions
MAINTENANCE_MARGIN_REQUIREMENT = 0.30  # 30% maintenance margin
BORROWING_COST_SPREAD = 0.002  # 0.2% additional cost above risk-free rate for borrowing
MAX_SHORT_POSITION_PCT = 0.10  # Maximum 10% short position in any single stock
MIN_VOLATILITY_THRESHOLD = 0.5  # Minimum realistic monthly volatility (0.5%)
MAX_REALISTIC_GROSS_EXPOSURE = 2.5  # Maximum realistic gross exposure (250%)

# Debug validation toggle
DEBUG_VALIDATE = True  # Set to False to disable strict validation checks


# ============================================================================
# STEP 1: UNIT VALIDATION AND CONVENTIONS
# ============================================================================

def validate_units(expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> None:
    """
    Validate that all inputs use consistent units (percentages).
    
    Expected magnitudes:
    - Returns: typically -10% to +10% (values like -10.0 to 10.0)
    - Risk-free rate: typically 0.01% to 1% monthly (values like 0.01 to 1.0)
    - Covariance: variance in percentage^2 (e.g., 0.01 to 100)
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns (should be in %)
    cov_matrix : pd.DataFrame
        Covariance matrix (should be in %^2)
    risk_free_rate : float
        Risk-free rate (should be in %)
    """
    if not DEBUG_VALIDATE:
        return
    
    # Check expected returns magnitude (should be in %)
    ret_min, ret_max = expected_returns.min(), expected_returns.max()
    if abs(ret_min) > 50 or abs(ret_max) > 50:
        logger.warning(f"  Expected returns seem large: min={ret_min:.2f}%, max={ret_max:.2f}%")
        logger.warning("   If these are in decimal form (0.01 = 1%), they should be multiplied by 100")
    
    # Check risk-free rate magnitude
    if abs(risk_free_rate) > 10:
        logger.warning(f"  Risk-free rate seems large: {risk_free_rate:.4f}%")
        logger.warning("   If this is in decimal form (0.01 = 1%), it should be multiplied by 100")
    
    # Check covariance diagonal (variance) magnitude
    variances = np.diag(cov_matrix.values)
    var_min, var_max = variances.min(), variances.max()
    if var_max > 1000 or var_min < 0:
        logger.warning(f"  Variances seem unusual: min={var_min:.4f}, max={var_max:.4f}")
        logger.warning("   Variances should be in %^2 (e.g., 4.0 for 2% volatility)")
    
    # Check for negative variances (should not happen)
    if np.any(variances < 0):
        logger.error(" Negative variances detected in covariance matrix!")
        raise ValueError("Covariance matrix contains negative variances")


def repair_gross_exposure_feasible(weights: np.ndarray, max_gross: float, tol: float = 1e-6) -> np.ndarray:
    """
    Repair weights to satisfy gross exposure constraint while maintaining sum(w)=1.
    
    Uses convex combination with equal weights: w_new = alpha*w + (1-alpha)*w_eq
    where alpha is chosen via binary search to satisfy sum(abs(w_new)) <= max_gross.
    
    Parameters
    ----------
    weights : np.ndarray
        Current weights (should sum to 1)
    max_gross : float
        Maximum allowed gross exposure
    tol : float
        Tolerance for constraint satisfaction
    
    Returns
    -------
    np.ndarray
        Repaired weights that satisfy both sum(w)=1 and sum(abs(w)) <= max_gross
    """
    n = len(weights)
    w_eq = np.ones(n) / n  # Equal weights (feasible base)
    
    current_gross = np.sum(np.abs(weights))
    if current_gross <= max_gross + tol:
        return weights.copy()
    
    # Binary search for alpha in [0, 1] such that gross exposure <= max_gross
    alpha_low, alpha_high = 0.0, 1.0
    best_weights = weights.copy()
    
    for _ in range(50):  # Max 50 iterations
        alpha = (alpha_low + alpha_high) / 2
        w_new = alpha * weights + (1 - alpha) * w_eq
        
        # Ensure sum = 1 (should be exact due to construction, but verify)
        w_new = w_new / np.sum(w_new)
        
        gross_new = np.sum(np.abs(w_new))
        
        if gross_new <= max_gross + tol:
            best_weights = w_new.copy()
            alpha_low = alpha
            if abs(gross_new - max_gross) < tol:
                break
        else:
            alpha_high = alpha
    
    return best_weights


def validate_portfolio_weights(weights: np.ndarray, allow_short: bool, max_gross: Optional[float] = None,
                                target_return: Optional[float] = None, expected_returns: Optional[pd.Series] = None,
                                tol: float = 1e-6) -> Tuple[bool, list]:
    """
    Validate portfolio weights satisfy all constraints.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    allow_short : bool
        Whether short-selling is allowed
    max_gross : float, optional
        Maximum gross exposure (if short-selling)
    target_return : float, optional
        Target return (if equality constraint exists)
    expected_returns : pd.Series, optional
        Expected returns (needed if target_return is specified)
    tol : float
        Tolerance for constraint checks
    
    Returns
    -------
    Tuple[bool, list]
        (is_valid, list_of_violations)
    """
    violations = []
    
    # Check sum = 1
    weight_sum = np.sum(weights)
    if abs(weight_sum - 1.0) > tol:
        violations.append(f"Sum constraint violated: sum(w) = {weight_sum:.8f} (expected 1.0)")
    
    # Check bounds
    if not allow_short:
        if np.any(weights < -tol):
            violations.append(f"Long-only constraint violated: {np.sum(weights < -tol)} negative weights")
    else:
        # For short-selling, check gross exposure
        if max_gross is not None:
            gross_exp = np.sum(np.abs(weights))
            if gross_exp > max_gross + tol:
                violations.append(f"Gross exposure constraint violated: {gross_exp:.6f} > {max_gross}")
    
    # Check target return (if specified)
    if target_return is not None and expected_returns is not None:
        # Handle both Series and numpy array
        if isinstance(expected_returns, pd.Series):
            expected_returns_np = expected_returns.values
        else:
            expected_returns_np = expected_returns
        actual_return = portfolio_return(weights, expected_returns_np)
        if abs(actual_return - target_return) > tol * max(1, abs(target_return)):
            violations.append(f"Target return constraint violated: {actual_return:.6f}% != {target_return:.6f}%")
    
    return len(violations) == 0, violations


def select_top_15_stocks_by_market_cap(panel_df: pd.DataFrame, n_stocks: int = 15) -> pd.Series:
    """
    Select the top N stocks by market capitalization, ensuring at least one stock from each country.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Returns panel with ticker, country, and other columns
    n_stocks : int
        Number of stocks to select (default: 15)
    
    Returns
    -------
    pd.Series
        Series of selected tickers
    """
    logger.info(f"Selecting top {n_stocks} stocks by market capitalization...")
    
    # Load CAPM results to get country information
    capm_results_file = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
    if not os.path.exists(capm_results_file):
        logger.error("CAPM results not found. Cannot select stocks by market cap.")
        return pd.Series(dtype=str)
    
    capm_results = pd.read_csv(capm_results_file)
    valid_stocks = capm_results[capm_results['is_valid']].copy()
    
    # Get unique countries
    unique_countries = valid_stocks['country'].unique()
    logger.info(f"  Found {len(unique_countries)} countries: {', '.join(unique_countries)}")
    
    # Import market cap functions
    from analysis.extensions.market_cap_analysis import load_market_caps_for_country
    
    # Load market caps for all countries
    all_market_caps = []
    for country in unique_countries:
        country_caps = load_market_caps_for_country(country, valid_stocks)
        if len(country_caps) > 0:
            country_caps['country'] = country
            all_market_caps.append(country_caps)
    
    if len(all_market_caps) == 0:
        logger.warning("No market cap data available. Using all valid stocks.")
        return pd.Series(valid_stocks['ticker'].unique()[:n_stocks])
    
    market_caps_df = pd.concat(all_market_caps, ignore_index=True)
    market_caps_df = market_caps_df.sort_values('market_cap_millions', ascending=False)
    
    # First, select the largest stock from each country to ensure representation
    selected_tickers = []
    selected_countries = set()
    
    for country in unique_countries:
        country_stocks = market_caps_df[market_caps_df['country'] == country]
        if len(country_stocks) > 0:
            largest_in_country = country_stocks.iloc[0]
            selected_tickers.append(largest_in_country['ticker'])
            selected_countries.add(country)
            logger.info(f"  Selected {largest_in_country['ticker']} from {country} (market cap: {largest_in_country['market_cap_millions']:.0f}M)")
    
    # Then, select remaining stocks from the largest overall, excluding already selected
    remaining_slots = n_stocks - len(selected_tickers)
    if remaining_slots > 0:
        remaining_stocks = market_caps_df[~market_caps_df['ticker'].isin(selected_tickers)]
        top_remaining = remaining_stocks.head(remaining_slots)
        
        # Vectorized selection instead of iterrows (more efficient)
        for idx in top_remaining.index:
            ticker = top_remaining.loc[idx, 'ticker']
            country = top_remaining.loc[idx, 'country']
            market_cap = top_remaining.loc[idx, 'market_cap_millions']
            selected_tickers.append(ticker)
            selected_countries.add(country)  # Track country representation
            logger.info(f"  Selected {ticker} from {country} (market cap: {market_cap:.0f}M)")
    
    selected_series = pd.Series(selected_tickers)
    logger.info(f" Selected {len(selected_series)} stocks: {', '.join(selected_series.tolist())}")
    logger.info(f"  Countries represented: {len(selected_countries)}/{len(unique_countries)}")
    
    return selected_series


def calculate_expected_returns_and_covariance(panel_df: pd.DataFrame, selected_tickers: Optional[pd.Series] = None) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calculate expected returns (mean historical returns) and covariance matrix.
    
    Only includes valid stocks (stocks with is_valid=True in CAPM results).
    This ensures consistency across all analyses.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Returns panel with columns: [date, country, ticker, stock_return, ...]
    selected_tickers : pd.Series, optional
        If provided, only use these tickers
    
    Returns
    -------
    tuple
        (expected_returns, covariance_matrix)
    """
    logger.info("Calculating expected returns and covariance matrix...")
    
    # Load CAPM results to filter by valid stocks only
    capm_results_file = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
    if not os.path.exists(capm_results_file):
        logger.error(f"CAPM results not found: {capm_results_file}")
        raise FileNotFoundError(f"CAPM results file not found: {capm_results_file}")
    
    capm_results = pd.read_csv(capm_results_file)
    # Normalize is_valid to boolean (handle string/boolean mix)
    if capm_results['is_valid'].dtype == 'object':
        capm_results['is_valid'] = capm_results['is_valid'].astype(str).str.lower().isin(['true', '1', 'yes'])
    valid_tickers = set(capm_results[capm_results['is_valid']]['ticker'].unique())
    
    # Filter panel to only include valid stocks
    panel_df_filtered = panel_df[panel_df['ticker'].isin(valid_tickers)].copy()
    
    # If specific tickers are provided, filter to those
    if selected_tickers is not None:
        selected_set = set(selected_tickers.tolist())
        panel_df_filtered = panel_df_filtered[panel_df_filtered['ticker'].isin(selected_set)].copy()
        logger.info(f"  Filtered to {len(selected_set)} selected stocks")
    
    logger.info(f"  Using {panel_df_filtered['ticker'].nunique()} stocks (from {panel_df['ticker'].nunique()} total)")
    
    # Pivot to wide format: stocks as columns, dates as rows
    returns_wide = panel_df_filtered.pivot_table(
        index='date',
        columns='ticker',
        values='stock_return',
        aggfunc='mean'
    )
    
    # Calculate expected returns (mean)
    expected_returns = returns_wide.mean()
    
    # Calculate covariance matrix
    covariance_matrix = returns_wide.cov()
    
    logger.info(f"  Expected returns: {len(expected_returns)} stocks")
    logger.info(f"  Covariance matrix: {covariance_matrix.shape}")
    logger.info(f"  Mean return range: [{expected_returns.min():.2f}%, {expected_returns.max():.2f}%]")
    logger.info(f"  Mean return: {expected_returns.mean():.2f}%")
    
    return expected_returns, covariance_matrix


def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculate portfolio variance: w' * Σ * w
    
    Optimized: accepts numpy array directly to avoid repeated .values conversion.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    cov_matrix : np.ndarray
        Covariance matrix (numpy array, not DataFrame)
    
    Returns
    -------
    float
        Portfolio variance
    """
    # Use einsum for better performance: w' * Σ * w = Σ_ij w_i * Σ_ij * w_j
    # This is equivalent to np.dot(weights, np.dot(cov_matrix, weights)) but faster
    return np.einsum('i,ij,j->', weights, cov_matrix, weights)


def portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """
    Calculate portfolio expected return: w' * μ
    
    Optimized: accepts numpy array directly to avoid repeated .values conversion.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    expected_returns : np.ndarray
        Expected returns (numpy array, not Series)
    
    Returns
    -------
    float
        Portfolio expected return
    """
    return np.dot(weights, expected_returns)


def portfolio_return_with_costs(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    risk_free_rate: float
) -> float:
    """
    Calculate portfolio return adjusted for transaction costs and borrowing costs.
    
    Optimized: accepts numpy array directly and uses vectorized operations.
    
    For short positions:
    - Short return = -expected_return (lose when stock goes up)
    - Borrowing cost = (risk_free_rate + borrowing_cost_spread) × |short_exposure|
    
    Transaction costs reduce returns by: cost_rate × gross_exposure
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (can be negative for short positions)
    expected_returns : np.ndarray
        Expected returns (numpy array, not Series)
    risk_free_rate : float
        Risk-free rate (monthly, percentage)
    
    Returns
    -------
    float
        Adjusted portfolio return (percentage)
    """
    # Calculate gross exposure (cached for reuse)
    abs_weights = np.abs(weights)
    gross_exposure = np.sum(abs_weights)
    
    # Separate long and short positions (vectorized)
    long_weights = np.maximum(weights, 0)
    short_weights = np.minimum(weights, 0)
    
    # Long positions earn expected return
    long_return = np.dot(long_weights, expected_returns)
    
    # Short positions: lose expected return (negative of expected return)
    # When you short, you profit when stock goes down, lose when it goes up
    # Expected short return = -expected_return (you lose the expected return)
    short_return = np.dot(short_weights, expected_returns)
    
    # Borrowing cost: pay additional cost above risk-free rate for borrowing shares
    # Reuse abs_weights for short exposure calculation
    short_exposure = np.sum(abs_weights[weights < 0]) if np.any(weights < 0) else 0.0
    borrowing_cost = (risk_free_rate + BORROWING_COST_SPREAD) * short_exposure
    
    # Transaction costs: apply to gross exposure (both long and short trades)
    transaction_costs = TRANSACTION_COST_RATE * gross_exposure
    
    # Adjusted return: long return + short return - borrowing cost - transaction costs
    # Note: short_return is already negative (from negative weights × positive returns)
    adjusted_return = long_return + short_return - borrowing_cost - transaction_costs
    
    return adjusted_return


def portfolio_variance_with_costs(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> float:
    """
    Calculate portfolio variance adjusted for transaction costs and market impact.
    
    Optimized: accepts numpy array directly and caches calculations.
    
    Transaction costs add variance due to execution uncertainty.
    Market impact increases variance for large positions.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    cov_matrix : np.ndarray
        Covariance matrix (numpy array, not DataFrame)
    
    Returns
    -------
    float
        Adjusted portfolio variance
    """
    # Base variance
    base_variance = portfolio_variance(weights, cov_matrix)
    
    # Transaction cost variance: proportional to gross exposure
    # Cache abs(weights) to avoid recalculating
    abs_weights = np.abs(weights)
    gross_exposure = np.sum(abs_weights)
    transaction_cost_variance = (TRANSACTION_COST_RATE * gross_exposure) ** 2
    
    # Market impact variance: larger positions have more impact
    # Use squared weights directly (more efficient than weights ** 2)
    market_impact_variance = 0.0001 * np.dot(weights, weights)  # w' * w = sum(w^2)
    
    # Adjusted variance
    adjusted_variance = base_variance + transaction_cost_variance + market_impact_variance
    
    return adjusted_variance


def find_minimum_variance_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    allow_short: bool = False
) -> Tuple[np.ndarray, float, float]:
    """
    Find the minimum-variance portfolio.
    
    Optimized: caches numpy arrays to avoid repeated conversions.
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns
    cov_matrix : pd.DataFrame
        Covariance matrix
    allow_short : bool
        Whether to allow short selling
    
    Returns
    -------
    tuple
        (weights, expected_return, volatility)
    """
    logger.info("Finding minimum-variance portfolio...")
    
    n = len(expected_returns)
    
    # Cache numpy arrays to avoid repeated .values conversion
    cov_matrix_np = cov_matrix.values
    expected_returns_np = expected_returns.values
    
    # Objective: minimize portfolio variance (use cached numpy array)
    def objective(weights):
        return portfolio_variance(weights, cov_matrix_np)
    
    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Add gross exposure constraint for short-selling case
    # Use slightly tighter constraint to account for numerical tolerance
    if allow_short:
        constraints.append({
            'type': 'ineq', 
            'fun': lambda w: (MAX_GROSS_EXPOSURE - 1e-4) - np.sum(np.abs(w))  # Tighter by 1e-4
        })
    
    # Bounds: no short selling if not allowed
    if allow_short:
        bounds = [(-1, 1) for _ in range(n)]
    else:
        bounds = [(0, 1) for _ in range(n)]
    
    # Initial guess: for short-selling, get unconstrained solution first, then project onto constraint
    if allow_short:
        # First get unconstrained solution as starting point
        constraints_unconstrained = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        x0_unconstrained = np.ones(n) / n
        result_unconstrained = minimize(objective, x0_unconstrained, method='SLSQP',
                                       bounds=bounds, constraints=constraints_unconstrained,
                                       options={'ftol': 1e-4, 'maxiter': 200})
        
        if result_unconstrained.success:
            # Project onto constraint boundary if needed
            unconstrained_weights = result_unconstrained.x
            gross_exp = np.sum(np.abs(unconstrained_weights))
            if gross_exp > MAX_GROSS_EXPOSURE:
                # Scale down to satisfy constraint while maintaining sum=1
                # This is a simple projection: scale all weights proportionally
                scale_factor = MAX_GROSS_EXPOSURE / gross_exp
                x0 = unconstrained_weights * scale_factor
                # Renormalize to ensure sum = 1 (may need adjustment)
                x0 = x0 / np.sum(x0)
                logger.debug(f"  Projected initial guess from unconstrained solution (gross exposure: {gross_exp:.4f} → {np.sum(np.abs(x0)):.4f})")
            else:
                x0 = unconstrained_weights
        else:
            # Fallback to equal weights if unconstrained fails
            x0 = np.ones(n) / n
    else:
        # For long-only, use equal weights
        x0 = np.ones(n) / n
    
    # Optimize: use trust-constr for short-selling (handles constraints better)
    if allow_short:
        result = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'gtol': 1e-4})
        if not result.success:
            logger.warning("trust-constr did not converge, trying SLSQP as fallback...")
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                             options={'ftol': 1e-4, 'maxiter': 1000})
    else:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'ftol': 1e-5, 'maxiter': 500})
    
    weights = result.x
    
    # Post-optimization constraint enforcement
    if not allow_short:
        # For long-only: enforce w_i ≥ 0 constraint strictly
        # Clip any negative weights (from numerical precision errors) to 0
        weights = np.maximum(weights, 0)
        # Renormalize to sum to 1
        weight_sum = np.sum(weights)
        if weight_sum > 1e-10:
            weights = weights / weight_sum
        else:
            # Fallback: equal weights if all weights are too small
            weights = np.ones(len(weights)) / len(weights)
            logger.warning("  All weights too small in minimum-variance portfolio, using equal weights")
        
        # Verify constraint satisfaction (should be impossible after clipping, but double-check)
        if np.any(weights < -1e-8):
            logger.warning("    Negative weights detected in long-only minimum-variance portfolio, clipping to 0")
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
    
    # Post-optimization constraint enforcement for short-selling case
    if allow_short:
        # Use proper feasibility repair that maintains sum(w)=1
        weights = repair_gross_exposure_feasible(weights, MAX_GROSS_EXPOSURE, tol=1e-6)
        
        final_gross_exp = np.sum(np.abs(weights))
        if final_gross_exp > MAX_GROSS_EXPOSURE + 1e-5:
            logger.warning(f"    Constraint violation after repair: gross exposure {final_gross_exp:.6f} > {MAX_GROSS_EXPOSURE}")
            # Re-optimize with repaired weights as starting point
            logger.info("  Re-optimizing with repaired weights as starting point...")
            result_retry = minimize(objective, weights, method='trust-constr', bounds=bounds, constraints=constraints,
                                  options={'maxiter': 500, 'gtol': 1e-4})
            if result_retry.success:
                weights = repair_gross_exposure_feasible(result_retry.x, MAX_GROSS_EXPOSURE, tol=1e-6)
    
    # Validate weights
    if DEBUG_VALIDATE:
        is_valid, violations = validate_portfolio_weights(weights, allow_short, 
                                                          max_gross=MAX_GROSS_EXPOSURE if allow_short else None)
        if not is_valid:
            logger.warning(f"    Weight validation failed: {violations}")
    
    port_return = portfolio_return(weights, expected_returns)
    port_vol = np.sqrt(portfolio_variance(weights, cov_matrix))
    
    logger.info(f"  Minimum-variance portfolio:")
    logger.info(f"    Expected return: {port_return:.4f}%")
    logger.info(f"    Volatility: {port_vol:.4f}%")
    logger.info(f"    Sharpe ratio: {port_return / port_vol:.4f}")
    if allow_short:
        logger.info(f"    Gross exposure: {np.sum(np.abs(weights)):.4f}")
    
    return weights, port_return, port_vol


def find_tangency_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    allow_short: bool = False
) -> Tuple[np.ndarray, float, float]:
    """
    Find the tangency portfolio (optimal risky portfolio).
    Maximizes Sharpe ratio: (μ_p - r_f) / σ_p
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns
    cov_matrix : pd.DataFrame
        Covariance matrix
    risk_free_rate : float
        Risk-free rate (monthly, in percentage)
    allow_short : bool
        Whether to allow short selling
    
    Returns
    -------
    tuple
        (weights, expected_return, volatility)
    """
    logger.info("Finding tangency portfolio (optimal risky portfolio)...")
    
    n = len(expected_returns)
    
    # Excess returns
    excess_returns = expected_returns - risk_free_rate
    
    # Cache numpy arrays to avoid repeated .values conversion
    cov_matrix_np = cov_matrix.values
    expected_returns_np = expected_returns.values
    
    # Objective: minimize negative Sharpe ratio (maximize Sharpe ratio)
    def objective(weights):
        port_return = portfolio_return(weights, expected_returns_np)
        port_vol = np.sqrt(portfolio_variance(weights, cov_matrix_np))
        if port_vol == 0:
            return 1e10
        sharpe = (port_return - risk_free_rate) / port_vol
        return -sharpe  # Minimize negative Sharpe = maximize Sharpe
    
    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Add gross exposure constraint for short-selling case
    # Use slightly tighter constraint to account for numerical tolerance
    if allow_short:
        constraints.append({
            'type': 'ineq', 
            'fun': lambda w: (MAX_GROSS_EXPOSURE - 1e-4) - np.sum(np.abs(w))  # Tighter by 1e-4
        })
    
    # Bounds
    if allow_short:
        bounds = [(-1, 1) for _ in range(n)]
    else:
        bounds = [(0, 1) for _ in range(n)]
    
    # Initial guess: for short-selling, get unconstrained solution first, then project onto constraint
    if allow_short:
        # First get unconstrained tangency solution as starting point
        constraints_unconstrained = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        x0_unconstrained = np.ones(n) / n
        result_unconstrained = minimize(objective, x0_unconstrained, method='SLSQP',
                                       bounds=bounds, constraints=constraints_unconstrained,
                                       options={'ftol': 1e-4, 'maxiter': 200})
        
        if result_unconstrained.success:
            # Project onto constraint boundary if needed
            unconstrained_weights = result_unconstrained.x
            gross_exp = np.sum(np.abs(unconstrained_weights))
            if gross_exp > MAX_GROSS_EXPOSURE:
                # Scale down to satisfy constraint
                scale_factor = MAX_GROSS_EXPOSURE / gross_exp
                x0 = unconstrained_weights * scale_factor
                # Renormalize to ensure sum = 1
                x0 = x0 / np.sum(x0)
                logger.debug(f"  Projected initial guess from unconstrained tangency (gross exposure: {gross_exp:.4f} → {np.sum(np.abs(x0)):.4f})")
            else:
                x0 = unconstrained_weights
        else:
            # Fallback to equal weights if unconstrained fails
            x0 = np.ones(n) / n
    else:
        # For long-only, use equal weights
        x0 = np.ones(n) / n
    
    # Optimize: use trust-constr for short-selling (handles constraints better)
    if allow_short:
        result = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'gtol': 1e-4})
        if not result.success:
            logger.warning("trust-constr did not converge, trying SLSQP as fallback...")
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                             options={'ftol': 1e-4, 'maxiter': 1000})
    else:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'ftol': 1e-5, 'maxiter': 500})
    
    weights = result.x
    
    # Post-optimization constraint enforcement
    if not allow_short:
        # For long-only: enforce w_i ≥ 0 constraint strictly
        # Clip any negative weights (from numerical precision errors) to 0
        weights = np.maximum(weights, 0)
        # Renormalize to sum to 1
        weight_sum = np.sum(weights)
        if weight_sum > 1e-10:
            weights = weights / weight_sum
        else:
            # Fallback: equal weights if all weights are too small
            weights = np.ones(len(weights)) / len(weights)
            logger.warning("  All weights too small in tangency portfolio, using equal weights")
        
        # Verify constraint satisfaction (should be impossible after clipping, but double-check)
        if np.any(weights < -1e-8):
            logger.warning("    Negative weights detected in long-only tangency portfolio, clipping to 0")
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
    
    # Post-optimization constraint enforcement for short-selling case
    if allow_short:
        # Use proper feasibility repair that maintains sum(w)=1
        weights = repair_gross_exposure_feasible(weights, MAX_GROSS_EXPOSURE, tol=1e-6)
        
        final_gross_exp = np.sum(np.abs(weights))
        if final_gross_exp > MAX_GROSS_EXPOSURE + 1e-5:
            logger.warning(f"    Constraint violation after repair: gross exposure {final_gross_exp:.6f} > {MAX_GROSS_EXPOSURE}")
            # Re-optimize with repaired weights as starting point
            logger.info("  Re-optimizing tangency with repaired weights as starting point...")
            result_retry = minimize(objective, weights, method='trust-constr', bounds=bounds, constraints=constraints,
                                  options={'maxiter': 500, 'gtol': 1e-4})
            if result_retry.success:
                weights = repair_gross_exposure_feasible(result_retry.x, MAX_GROSS_EXPOSURE, tol=1e-6)
    
    port_return = portfolio_return(weights, expected_returns_np)
    port_vol = np.sqrt(portfolio_variance(weights, cov_matrix_np))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
    
    logger.info(f"  Tangency portfolio:")
    logger.info(f"    Expected return: {port_return:.4f}%")
    logger.info(f"    Volatility: {port_vol:.4f}%")
    logger.info(f"    Sharpe ratio: {sharpe:.4f}")
    if allow_short:
        logger.info(f"    Gross exposure: {np.sum(np.abs(weights)):.4f}")
    
    return weights, port_return, port_vol


def calculate_efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
    allow_short: bool = False,
    tangency_return: Optional[float] = None
) -> pd.DataFrame:
    """
    Calculate efficient frontier.
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns
    cov_matrix : pd.DataFrame
        Covariance matrix
    n_points : int
        Number of points on efficient frontier
    allow_short : bool
        Whether to allow short selling
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [return, volatility]
        Note: Sharpe ratio is not included as it requires risk-free rate which is not passed to this function.
    """
    logger.info(f"Calculating efficient frontier ({n_points} points)...")
    
    # Get minimum-variance portfolio to find the minimum return on the frontier
    _, min_var_return, min_var_vol = find_minimum_variance_portfolio(
        expected_returns, cov_matrix, allow_short
    )
    
    # For short-selling case with gross exposure constraint, use tangency return if provided
    if allow_short:
        if tangency_return is not None:
            # Use the tangency portfolio return as upper bound (already calculated)
            max_return = tangency_return
            logger.info(f"  Using tangency portfolio return as upper bound: {max_return:.4f}%")
        else:
            # Fallback: use a simple conservative estimate
            sorted_returns = expected_returns.sort_values(ascending=False)
            max_return_estimate = min_var_return + (sorted_returns.iloc[0] - min_var_return) * 1.2
            max_return = min(max_return_estimate, expected_returns.max() * 1.1)
            logger.info(f"  Using conservative maximum return estimate: {max_return:.4f}%")
    else:
        # For long-only, maximum return is the maximum individual stock return
        max_return = expected_returns.max()
    
    # Ensure max_return is at least min_var_return
    if max_return < min_var_return:
        max_return = min_var_return + 0.1  # Small buffer
    
    # Target returns between min_var_return and max_return
    # Use fewer points for short-selling case to speed up optimization
    if allow_short:
        n_points_actual = min(n_points, 15)  # Limit to 15 points for short-selling to avoid getting stuck
        target_returns = np.linspace(min_var_return, max_return, n_points_actual)
        logger.info(f"  Using {n_points_actual} points for short-selling frontier (reduced for speed)")
    else:
        target_returns = np.linspace(min_var_return, max_return, n_points)
    
    n = len(expected_returns)
    
    # Cache numpy arrays to avoid repeated .values conversion in loops
    cov_matrix_np = cov_matrix.values
    expected_returns_np = expected_returns.values
    
    # Pre-compute bounds to avoid recreating in every iteration
    if allow_short:
        bounds = [(-1, 1)] * n
    else:
        bounds = [(0, 1)] * n
    
    frontier_returns = []
    frontier_vols = []
    
    # Get initial weights from minimum-variance portfolio for better starting point
    min_var_weights, _, _ = find_minimum_variance_portfolio(
        expected_returns, cov_matrix, allow_short
    )
    x0 = min_var_weights.copy()
    
    for i, target_return in enumerate(target_returns):
        # Objective: minimize variance (use cached numpy array)
        def objective(weights):
            return portfolio_variance(weights, cov_matrix_np)
        
        # Constraints: weights sum to 1, target return
        # Use closure to capture target_return properly
        def return_constraint(weights, target=target_return):
            return portfolio_return(weights, expected_returns_np) - target
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': return_constraint}
        ]
        
        # Add gross exposure constraint for short-selling case
        # Use slightly tighter constraint to account for numerical tolerance
        if allow_short:
            constraints.append({
                'type': 'ineq', 
                'fun': lambda w: (MAX_GROSS_EXPOSURE - 1e-4) - np.sum(np.abs(w))  # Tighter by 1e-4
            })
        
        # Optimize: use trust-constr for short-selling (handles constraints better)
        if allow_short:
            result = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=constraints,
                             options={'maxiter': 500, 'gtol': 1e-4})
            if not result.success:
                # Fallback to SLSQP with more iterations
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                                 options={'ftol': 1e-4, 'maxiter': 500})
        else:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                             options={'ftol': 1e-5, 'maxiter': 300})
        
        if result.success:
            weights = result.x
            
            # Post-optimization constraint enforcement
            # CRITICAL: Do NOT renormalize or clip aggressively, as this breaks the target return constraint
            # Only fix tiny numerical errors that don't meaningfully affect the target return
            
            if not allow_short:
                # For long-only: only clip truly tiny negatives (< 1e-10) that are numerical noise
                # Do NOT renormalize unless absolutely necessary, as it breaks target return constraint
                tiny_negatives = weights < -1e-10
                if np.any(tiny_negatives):
                    logger.debug(f"  Clipping {np.sum(tiny_negatives)} tiny negative weights at target return {target_return:.4f}%")
                    weights = np.maximum(weights, 0)
                    # Only renormalize if sum changed significantly
                    weight_sum = np.sum(weights)
                    if abs(weight_sum - 1.0) > 1e-6:
                        weights = weights / weight_sum
                        # Re-optimize to restore target return constraint
                        logger.debug(f"  Re-optimizing to restore target return after clipping...")
                        result_retry = minimize(objective, weights, method='SLSQP', bounds=bounds, constraints=constraints,
                                              options={'ftol': 1e-5, 'maxiter': 200})
                        if result_retry.success:
                            weights = result_retry.x
                        else:
                            # If re-optimization fails, mark as NaN
                            frontier_returns.append(np.nan)
                            frontier_vols.append(np.nan)
                            continue
            
            # Post-optimization constraint enforcement for short-selling case
            if allow_short:
                # Use proper feasibility repair that maintains sum(w)=1
                weights = repair_gross_exposure_feasible(weights, MAX_GROSS_EXPOSURE, tol=1e-6)
                
                final_gross_exp = np.sum(np.abs(weights))
                if final_gross_exp > MAX_GROSS_EXPOSURE + 1e-5:
                    # Re-optimize with repaired weights as starting point
                    logger.debug(f"  Re-optimizing at target return {target_return:.4f}% to satisfy gross exposure...")
                    result_retry = minimize(objective, weights, method='trust-constr', bounds=bounds, constraints=constraints,
                                          options={'maxiter': 300, 'gtol': 1e-4})
                    if result_retry.success:
                        weights = repair_gross_exposure_feasible(result_retry.x, MAX_GROSS_EXPOSURE, tol=1e-6)
                        final_gross_exp = np.sum(np.abs(weights))
                        if final_gross_exp > MAX_GROSS_EXPOSURE + 1e-5:
                            # Still violated, skip this point
                            frontier_returns.append(np.nan)
                            frontier_vols.append(np.nan)
                            continue
                    else:
                        # Re-optimization failed, skip this point
                        frontier_returns.append(np.nan)
                        frontier_vols.append(np.nan)
                        continue
            
            # Validate constraints before accepting (use cached numpy arrays)
            port_return = portfolio_return(weights, expected_returns_np)
            port_vol = np.sqrt(portfolio_variance(weights, cov_matrix_np))
            
            if DEBUG_VALIDATE:
                is_valid, violations = validate_portfolio_weights(
                    weights, allow_short,
                    max_gross=MAX_GROSS_EXPOSURE if allow_short else None,
                    target_return=target_return,
                    expected_returns=pd.Series(expected_returns_np, index=expected_returns.index)  # Convert back for validation
                )
                if not is_valid:
                    logger.debug(f"  Constraint violations at target return {target_return:.4f}%: {violations}")
                    # If target return is significantly violated, skip this point
                    if any("Target return" in v for v in violations):
                        frontier_returns.append(np.nan)
                        frontier_vols.append(np.nan)
                        continue
            
            frontier_returns.append(port_return)
            frontier_vols.append(port_vol)
            # Use current solution as starting point for next optimization (sequential approach)
            x0 = weights.copy()
        else:
            # If optimization fails, append NaN and try next point
            frontier_returns.append(np.nan)
            frontier_vols.append(np.nan)
    
    # Filter out NaN values before creating DataFrame (vectorized for efficiency)
    # Convert to numpy arrays for faster filtering
    frontier_returns_arr = np.array(frontier_returns, dtype=np.float64)
    frontier_vols_arr = np.array(frontier_vols, dtype=np.float64)
    valid_mask = ~(np.isnan(frontier_returns_arr) | np.isnan(frontier_vols_arr))
    frontier_returns_clean = frontier_returns_arr[valid_mask].tolist()
    frontier_vols_clean = frontier_vols_arr[valid_mask].tolist()
    
    # For short-selling case, if we have very few points, add min-var and tangency portfolios
    if allow_short and len(frontier_returns_clean) < 3 and tangency_return is not None:
        logger.warning(f"  Only {len(frontier_returns_clean)} frontier points calculated, adding min-var and tangency portfolios")
        # Get min-var and tangency portfolios explicitly
        min_var_weights, min_var_ret, min_var_vol = find_minimum_variance_portfolio(
            expected_returns, cov_matrix, allow_short=True
        )
        # Tangency portfolio should already be calculated, but we need its volatility
        # For now, just ensure we have at least min-var point
        if min_var_ret not in frontier_returns_clean:
            frontier_returns_clean.insert(0, min_var_ret)
            frontier_vols_clean.insert(0, min_var_vol)
    
    frontier_df = pd.DataFrame({
        'return': frontier_returns_clean,
        'volatility': frontier_vols_clean
    })
    
    n_points_attempted = len(target_returns)
    logger.info(f"  Efficient frontier: {len(frontier_df)} points (out of {n_points_attempted} attempted)")
    if len(frontier_df) < n_points_attempted:
        logger.info(f"    {n_points_attempted - len(frontier_df)} points failed optimization (likely infeasible target returns or iteration limits)")
        if allow_short and len(frontier_df) < 3:
            logger.warning("    Consider using simplified approach (min-var and tangency portfolios only)")
    
    return frontier_df


def find_minimum_variance_portfolio_realistic_short(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float
) -> Tuple[np.ndarray, float, float, Dict]:
    """
    Find minimum-variance portfolio with realistic short-selling constraints.
    
    Applies transaction costs, margin requirements, borrowing costs, and position limits
    to create an economically achievable portfolio.
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns
    cov_matrix : pd.DataFrame
        Covariance matrix
    risk_free_rate : float
        Risk-free rate (monthly, percentage)
    
    Returns
    -------
    tuple
        (weights, expected_return, volatility, error_flags)
        error_flags: dict with flags and explanations
    """
    logger.info("Finding minimum-variance portfolio with realistic short-selling constraints...")
    
    n = len(expected_returns)
    error_flags = {
        'unrealistic_volatility': False,
        'excessive_leverage': False,
        'large_short_position': False,
        'optimization_failed': False,
        'explanations': []
    }
    
    # Cache numpy arrays to avoid repeated conversions
    cov_matrix_np = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
    expected_returns_np = expected_returns.values if isinstance(expected_returns, pd.Series) else expected_returns
    
    # Objective: minimize adjusted portfolio variance (use cached numpy array)
    def objective(weights):
        return portfolio_variance_with_costs(weights, cov_matrix_np)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        {
            'type': 'ineq',
            'fun': lambda w: MAX_REALISTIC_GROSS_EXPOSURE - np.sum(np.abs(w))  # Gross exposure limit
        }
    ]
    
    # Position limits: no single short position > MAX_SHORT_POSITION_PCT
    # Pre-build constraint functions to avoid lambda capture issues
    for i in range(n):
        # Use default argument to properly capture loop variable
        def make_constraint(idx):
            return lambda w: MAX_SHORT_POSITION_PCT + w[idx]
        constraints.append({
            'type': 'ineq',
            'fun': make_constraint(i)  # w[i] >= -MAX_SHORT_POSITION_PCT
        })
    
    # Bounds: allow short-selling but with realistic limits
    bounds = [(-MAX_SHORT_POSITION_PCT, 1) for _ in range(n)]
    
    # Initial guess: equal weights
    x0 = np.ones(n) / n
    
    # Optimize
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                     options={'ftol': 1e-6, 'maxiter': 1000})
    
    if not result.success:
        logger.warning("Optimization did not converge, trying alternative method...")
        result = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'gtol': 1e-6})
    
    if not result.success:
        error_flags['optimization_failed'] = True
        error_flags['explanations'].append("Optimization failed to converge - constraints may be too restrictive")
        logger.error("Optimization failed with realistic constraints")
        return np.ones(n) / n, 0.0, 0.0, error_flags
    
    weights = result.x
    
    # Calculate adjusted return and volatility
    port_return = portfolio_return_with_costs(weights, expected_returns_np, risk_free_rate)
    port_variance = portfolio_variance_with_costs(weights, cov_matrix_np)
    port_vol = np.sqrt(port_variance)
    
    # Calculate effective risk-free rate for Sharpe ratio
    long_exposure = np.sum(np.maximum(weights, 0))
    short_exposure = np.sum(np.abs(np.minimum(weights, 0)))
    total_exposure = long_exposure + short_exposure
    
    if total_exposure > 0:
        effective_rf = (long_exposure * risk_free_rate + short_exposure * (risk_free_rate + BORROWING_COST_SPREAD)) / total_exposure
    else:
        effective_rf = risk_free_rate
    
    sharpe_ratio = (port_return - effective_rf) / port_vol if port_vol > 0 else 0
    
    # Check for unrealistic results
    if port_vol < MIN_VOLATILITY_THRESHOLD:
        error_flags['unrealistic_volatility'] = True
        error_flags['explanations'].append(
            f"Unrealistic volatility ({port_vol:.4f}% < {MIN_VOLATILITY_THRESHOLD}%): "
            "Transaction costs and borrowing costs prevent near-perfect hedging in practice"
        )
    
    gross_exposure = np.sum(np.abs(weights))
    if gross_exposure > MAX_REALISTIC_GROSS_EXPOSURE:
        error_flags['excessive_leverage'] = True
        error_flags['explanations'].append(
            f"Excessive leverage (gross exposure {gross_exposure:.2f} > {MAX_REALISTIC_GROSS_EXPOSURE}): "
            "Margin requirements limit achievable leverage"
        )
    
    max_short = np.min(weights) if np.any(weights < 0) else 0
    if abs(max_short) > MAX_SHORT_POSITION_PCT:
        error_flags['large_short_position'] = True
        error_flags['explanations'].append(
            f"Large short position ({abs(max_short):.2%} > {MAX_SHORT_POSITION_PCT:.0%}): "
            "Liquidity constraints prevent large short positions"
        )
    
    logger.info(f"  Realistic minimum-variance portfolio:")
    logger.info(f"    Expected return: {port_return:.4f}%")
    logger.info(f"    Volatility: {port_vol:.4f}%")
    logger.info(f"    Effective risk-free rate: {effective_rf:.4f}%")
    logger.info(f"    Sharpe ratio: {sharpe_ratio:.4f}")
    logger.info(f"    Gross exposure: {gross_exposure:.4f}")
    
    if error_flags['explanations']:
        for explanation in error_flags['explanations']:
            logger.warning(f"    {explanation}")
    
    return weights, port_return, port_vol, error_flags


def find_tangency_portfolio_realistic_short(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float
) -> Tuple[np.ndarray, float, float, Dict]:
    """
    Find tangency portfolio with realistic short-selling constraints.
    
    Maximizes Sharpe ratio adjusted for transaction costs and borrowing costs.
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns
    cov_matrix : pd.DataFrame
        Covariance matrix
    risk_free_rate : float
        Risk-free rate (monthly, percentage)
    
    Returns
    -------
    tuple
        (weights, expected_return, volatility, error_flags)
    """
    logger.info("Finding tangency portfolio with realistic short-selling constraints...")
    
    n = len(expected_returns)
    error_flags = {
        'unrealistic_volatility': False,
        'excessive_leverage': False,
        'large_short_position': False,
        'optimization_failed': False,
        'explanations': []
    }
    
    # Cache numpy arrays to avoid repeated conversions
    cov_matrix_np = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
    expected_returns_np = expected_returns.values if isinstance(expected_returns, pd.Series) else expected_returns
    
    # Objective: maximize Sharpe ratio (minimize negative Sharpe)
    def objective(weights):
        port_return = portfolio_return_with_costs(weights, expected_returns_np, risk_free_rate)
        port_variance = portfolio_variance_with_costs(weights, cov_matrix_np)
        port_vol = np.sqrt(port_variance)
        if port_vol == 0:
            return 1e10
        
        # Calculate effective risk-free rate accounting for borrowing costs on short positions
        # For short positions, the effective risk-free rate is higher due to borrowing costs
        # Cache abs() calculation for efficiency
        abs_weights = np.abs(weights)
        long_exposure = np.sum(np.maximum(weights, 0))
        short_exposure = np.sum(abs_weights[weights < 0]) if np.any(weights < 0) else 0.0
        total_exposure = long_exposure + short_exposure
        
        if total_exposure > 0:
            # Weighted average risk-free rate: long positions use base rate, short positions use base + spread
            effective_rf = (long_exposure * risk_free_rate + short_exposure * (risk_free_rate + BORROWING_COST_SPREAD)) / total_exposure
        else:
            effective_rf = risk_free_rate
        
        sharpe = (port_return - effective_rf) / port_vol
        return -sharpe  # Minimize negative Sharpe = maximize Sharpe
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {
            'type': 'ineq',
            'fun': lambda w: MAX_REALISTIC_GROSS_EXPOSURE - np.sum(np.abs(w))
        }
    ]
    
    # Position limits
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, idx=i: MAX_SHORT_POSITION_PCT + w[idx]
        })
    
    # Bounds
    bounds = [(-MAX_SHORT_POSITION_PCT, 1) for _ in range(n)]
    
    # Initial guess
    x0 = np.ones(n) / n
    
    # Optimize
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                     options={'ftol': 1e-6, 'maxiter': 1000})
    
    if not result.success:
        result = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'gtol': 1e-6})
    
    if not result.success:
        error_flags['optimization_failed'] = True
        error_flags['explanations'].append("Optimization failed to converge")
        return np.ones(n) / n, 0.0, 0.0, error_flags
    
    weights = result.x
    
    # Calculate adjusted return and volatility
    port_return = portfolio_return_with_costs(weights, expected_returns_np, risk_free_rate)
    port_variance = portfolio_variance_with_costs(weights, cov_matrix_np)
    port_vol = np.sqrt(port_variance)
    
    # Calculate effective risk-free rate for Sharpe ratio
    long_exposure = np.sum(np.maximum(weights, 0))
    short_exposure = np.sum(np.abs(np.minimum(weights, 0)))
    total_exposure = long_exposure + short_exposure
    
    if total_exposure > 0:
        effective_rf = (long_exposure * risk_free_rate + short_exposure * (risk_free_rate + BORROWING_COST_SPREAD)) / total_exposure
    else:
        effective_rf = risk_free_rate
    
    sharpe_ratio = (port_return - effective_rf) / port_vol if port_vol > 0 else 0
    
    # Check for unrealistic results
    if port_vol < MIN_VOLATILITY_THRESHOLD:
        error_flags['unrealistic_volatility'] = True
        error_flags['explanations'].append(
            f"Unrealistic volatility ({port_vol:.4f}% < {MIN_VOLATILITY_THRESHOLD}%): "
            "Borrowing costs and transaction costs prevent perfect hedging"
        )
    
    gross_exposure = np.sum(np.abs(weights))
    if gross_exposure > MAX_REALISTIC_GROSS_EXPOSURE:
        error_flags['excessive_leverage'] = True
        error_flags['explanations'].append(
            f"Excessive leverage ({gross_exposure:.2f} > {MAX_REALISTIC_GROSS_EXPOSURE}): "
            "Margin requirements limit leverage"
        )
    
    logger.info(f"  Realistic tangency portfolio:")
    logger.info(f"    Expected return: {port_return:.4f}%")
    logger.info(f"    Volatility: {port_vol:.4f}%")
    logger.info(f"    Effective risk-free rate: {effective_rf:.4f}%")
    logger.info(f"    Sharpe ratio: {sharpe_ratio:.4f}")
    
    return weights, port_return, port_vol, error_flags


def calculate_efficient_frontier_realistic_short(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    n_points: int = 30  # Reduced from 50 to prevent long computation times
) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate efficient frontier with realistic short-selling constraints.
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns
    cov_matrix : pd.DataFrame
        Covariance matrix
    risk_free_rate : float
        Risk-free rate (monthly, percentage)
    n_points : int
        Number of points on efficient frontier
    
    Returns
    -------
    tuple
        (frontier_df, error_flags)
        frontier_df: DataFrame with columns [return, volatility, sharpe]
        error_flags: dict with flags for unrealistic portfolios
    """
    logger.info(f"Calculating realistic short-selling efficient frontier ({n_points} points)...")
    
    # Get minimum-variance portfolio
    min_var_weights, min_var_return, min_var_vol, _ = find_minimum_variance_portfolio_realistic_short(
        expected_returns, cov_matrix, risk_free_rate
    )
    
    # Get tangency portfolio for upper bound
    tangency_weights, tangency_return, tangency_vol, _ = find_tangency_portfolio_realistic_short(
        expected_returns, cov_matrix, risk_free_rate
    )
    
    # Target returns between min and max
    max_return = max(tangency_return, expected_returns.max())
    target_returns = np.linspace(min_var_return, max_return, n_points)
    
    n = len(expected_returns)
    frontier_returns = []
    frontier_vols = []
    frontier_sharpes = []
    error_flags_list = []
    
    x0 = min_var_weights.copy()
    
    # Cache numpy arrays once (outside loop for efficiency)
    cov_matrix_np = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
    expected_returns_np = expected_returns.values if isinstance(expected_returns, pd.Series) else expected_returns
    
    # Pre-compute bounds (more efficient than creating in loop)
    bounds = [(-MAX_SHORT_POSITION_PCT, 1)] * n
    
    logger.info(f"  Calculating {len(target_returns)} frontier points...")
    successful_points = 0
    for i, target_return in enumerate(target_returns):
        progress_pct = int((i + 1) / len(target_returns) * 100)
        progress_bar = "█" * (progress_pct // 5) + "░" * (20 - progress_pct // 5)
        logger.info(f"    [{progress_bar}] {progress_pct}% - Point {i+1}/{len(target_returns)}: target return = {target_return:.4f}%")
        
        # Objective: minimize variance (use cached numpy array)
        def objective(weights):
            return portfolio_variance_with_costs(weights, cov_matrix_np)
        
        # Constraints (use cached numpy array)
        def return_constraint(weights, target=target_return):
            return portfolio_return_with_costs(weights, expected_returns_np, risk_free_rate) - target
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': return_constraint},
            {
                'type': 'ineq',
                'fun': lambda w: MAX_REALISTIC_GROSS_EXPOSURE - np.sum(np.abs(w))
            }
        ]
        
        # Position limits
        for j in range(n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, idx=j: MAX_SHORT_POSITION_PCT + w[idx]
            })
        
        bounds = [(-MAX_SHORT_POSITION_PCT, 1) for _ in range(n)]
        
        # Optimize with timeout protection
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                             options={'ftol': 1e-5, 'maxiter': 300})  # Reduced maxiter for faster computation
        except Exception as e:
            logger.warning(f"  Optimization failed for target return {target_return:.4f}%: {e}")
            frontier_returns.append(np.nan)
            frontier_vols.append(np.nan)
            frontier_sharpes.append(np.nan)
            error_flags_list.append({'optimization_failed': True})
            continue
        
        if result.success:
            weights = result.x
            port_return = portfolio_return_with_costs(weights, expected_returns_np, risk_free_rate)
            port_variance = portfolio_variance_with_costs(weights, cov_matrix_np)
            port_vol = np.sqrt(port_variance)
            
            # Calculate effective risk-free rate for Sharpe ratio
            # Cache abs() calculation for efficiency
            abs_weights = np.abs(weights)
            long_exposure = np.sum(np.maximum(weights, 0))
            short_exposure = np.sum(abs_weights[weights < 0]) if np.any(weights < 0) else 0.0
            total_exposure = long_exposure + short_exposure
            
            if total_exposure > 0:
                effective_rf = (long_exposure * risk_free_rate + short_exposure * (risk_free_rate + BORROWING_COST_SPREAD)) / total_exposure
            else:
                effective_rf = risk_free_rate
            
            sharpe = (port_return - effective_rf) / port_vol if port_vol > 0 else 0
            
            # Check for errors (reuse abs_weights for efficiency)
            gross_exp = np.sum(abs_weights)
            error_flags = {
                'unrealistic_volatility': port_vol < MIN_VOLATILITY_THRESHOLD,
                'excessive_leverage': gross_exp > MAX_REALISTIC_GROSS_EXPOSURE,
                'large_short_position': np.any(weights < -MAX_SHORT_POSITION_PCT)
            }
            
            frontier_returns.append(port_return)
            frontier_vols.append(port_vol)
            frontier_sharpes.append(sharpe)
            error_flags_list.append(error_flags)
            
            x0 = weights.copy()
        else:
            # Skip failed optimizations
            frontier_returns.append(np.nan)
            frontier_vols.append(np.nan)
            frontier_sharpes.append(np.nan)
            error_flags_list.append({'optimization_failed': True})
    
    # Create DataFrame
    frontier_df = pd.DataFrame({
        'return': frontier_returns,
        'volatility': frontier_vols,
        'sharpe': frontier_sharpes
    }).dropna()
    
    # Aggregate error flags
    aggregate_flags = {
        'unrealistic_volatility_count': sum(1 for f in error_flags_list if f.get('unrealistic_volatility', False)),
        'excessive_leverage_count': sum(1 for f in error_flags_list if f.get('excessive_leverage', False)),
        'large_short_position_count': sum(1 for f in error_flags_list if f.get('large_short_position', False)),
        'optimization_failed_count': sum(1 for f in error_flags_list if f.get('optimization_failed', False))
    }
    
    logger.info(f"  Efficient frontier: {len(frontier_df)} points (out of {n_points} attempted)")
    if aggregate_flags['unrealistic_volatility_count'] > 0:
        logger.warning(f"    {aggregate_flags['unrealistic_volatility_count']} portfolios with unrealistic volatility")
    if aggregate_flags['optimization_failed_count'] > 0:
        logger.warning(f"    {aggregate_flags['optimization_failed_count']} optimization failures")
    
    return frontier_df, aggregate_flags


def calculate_diversification_benefits(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    portfolio_weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Calculate diversification benefits.
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns
    cov_matrix : pd.DataFrame
        Covariance matrix
    portfolio_weights : np.ndarray, optional
        Portfolio weights (if None, uses equal-weighted)
    
    Returns
    -------
    dict
        Dictionary with diversification metrics
    """
    logger.info("Calculating diversification benefits...")
    
    if portfolio_weights is None:
        # Equal-weighted portfolio
        n = len(expected_returns)
        portfolio_weights = np.ones(n) / n
    
    # Portfolio statistics
    port_return = portfolio_return(portfolio_weights, expected_returns)
    port_variance = portfolio_variance(portfolio_weights, cov_matrix)
    port_vol = np.sqrt(port_variance)
    
    # Average individual stock variance
    avg_stock_variance = np.diag(cov_matrix.values).mean()
    avg_stock_vol = np.sqrt(avg_stock_variance)
    
    # Diversification ratio
    diversification_ratio = avg_stock_vol / port_vol
    
    # Variance reduction
    variance_reduction = (avg_stock_variance - port_variance) / avg_stock_variance * 100
    
    logger.info(f"  Average individual stock volatility: {avg_stock_vol:.4f}%")
    logger.info(f"  Portfolio volatility: {port_vol:.4f}%")
    logger.info(f"  Diversification ratio: {diversification_ratio:.4f}")
    logger.info(f"  Variance reduction: {variance_reduction:.2f}%")
    
    return {
        'avg_stock_volatility': avg_stock_vol,
        'portfolio_volatility': port_vol,
        'diversification_ratio': diversification_ratio,
        'variance_reduction_pct': variance_reduction,
        'portfolio_return': port_return
    }


def plot_efficient_frontier(
    frontier_df_constrained: pd.DataFrame,
    frontier_df_unconstrained: pd.DataFrame,
    min_var_port_constrained: Tuple[float, float],
    min_var_port_unconstrained: Tuple[float, float],
    tangency_port_constrained: Tuple[float, float],
    tangency_port_unconstrained: Tuple[float, float],
    equal_weighted_port: Tuple[float, float],
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    market_index_return: float,
    market_index_vol: float,
    risk_free_rate: float,
    output_path: str
) -> None:
    """
    Plot efficient frontier with all individual stocks and market index (long-only only).
    
    This graph shows only constrained (long-only) portfolios, as unconstrained portfolios
    achieve economically unachievable results. The plot includes:
    - X-axis: Volatility (standard deviation)
    - Y-axis: Expected return
    - All individual stocks as scatter points
    - Constrained efficient frontier (long-only, solid line)
    - Market index (MSCI Europe) point
    - Constrained minimum-variance and tangency portfolios
    - Interpretation: If efficient frontier overlaps well with market index, CAPM holds.
                     If not, CAPM does not hold (opposite).
    
    Note: Unconstrained parameters are kept for backward compatibility but are ignored.
    
    Parameters
    ----------
    frontier_df_constrained : pd.DataFrame
        Constrained efficient frontier data (long-only) with 'volatility' and 'return' columns
    frontier_df_unconstrained : pd.DataFrame
        Unconstrained efficient frontier data (ignored, kept for backward compatibility)
    min_var_port_constrained : tuple
        (return, volatility) for constrained minimum-variance portfolio (long-only)
    min_var_port_unconstrained : tuple
        Unconstrained minimum-variance portfolio (ignored, kept for backward compatibility)
    tangency_port_constrained : tuple
        (return, volatility) for constrained tangency portfolio (long-only)
    tangency_port_unconstrained : tuple
        Unconstrained tangency portfolio (ignored, kept for backward compatibility)
    equal_weighted_port : tuple
        (return, volatility) for equal-weighted portfolio
    expected_returns : pd.Series
        Expected returns for all individual stocks
    cov_matrix : pd.DataFrame
        Covariance matrix for calculating individual stock volatilities
    market_index_return : float
        Expected return of market index (MSCI Europe)
    market_index_vol : float
        Volatility of market index (MSCI Europe)
    risk_free_rate : float
        Risk-free rate (German 3-month Bund, EUR, for all countries)
    output_path : str
        Path to save plot
    """
    logger.info("Creating enhanced efficient frontier plot with all stocks and market index...")
    
    # Verify units: returns are in percentage, covariance is in percentage²
    # Individual stock volatilities: sqrt(diag(Σ)) gives percentage
    individual_volatilities = np.sqrt(np.diag(cov_matrix.values))
    logger.info(f"  Unit check - Individual stock volatilities: min={individual_volatilities.min():.2f}%, max={individual_volatilities.max():.2f}%")
    logger.info(f"  Unit check - Expected returns: min={expected_returns.min():.2f}%, max={expected_returns.max():.2f}%")
    logger.info(f"  Unit check - Constrained frontier volatility: min={frontier_df_constrained['volatility'].min():.2f}%, max={frontier_df_constrained['volatility'].max():.2f}%")
    logger.info(f"  Unit check - Market index: return={market_index_return:.2f}%, vol={market_index_vol:.2f}%")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot ALL individual stocks
    ax.scatter(individual_volatilities, expected_returns.values, 
              alpha=0.4, s=30, color='gray', label=f'Individual Stocks (n={len(expected_returns)})',
              edgecolors='black', linewidths=0.3)
    
    # Plot constrained efficient frontier (long-only, solid line)
    ax.plot(frontier_df_constrained['volatility'], frontier_df_constrained['return'], 
            'b-', linewidth=2.5, label='Efficient Frontier (Long-Only)', zorder=5)
    
    # Plot market index (MSCI Europe)
    ax.scatter(market_index_vol, market_index_return, 
              s=200, marker='*', color='red', edgecolors='darkred', linewidths=2,
              label='Market Index (MSCI Europe)', zorder=6)
    
    # Plot constrained minimum-variance portfolio (long-only, red circle)
    ax.plot(min_var_port_constrained[1], min_var_port_constrained[0], 
            'ro', markersize=12, label='Min-Variance Portfolio (Long-Only)', zorder=5)
    
    # Plot constrained tangency portfolio (long-only, green circle)
    ax.plot(tangency_port_constrained[1], tangency_port_constrained[0], 
            'go', markersize=12, label='Optimal Risky Portfolio (Tangency, Long-Only)', zorder=5)
    
    # Plot equal-weighted portfolio (purple diamond)
    ax.plot(equal_weighted_port[1], equal_weighted_port[0], 
            'mD', markersize=12, label='Equal-Weighted Portfolio', zorder=5)
    
    # Plot risk-free rate
    ax.axhline(y=risk_free_rate, color='k', linestyle='--', linewidth=1, 
              label=f'Risk-Free Rate ({risk_free_rate:.2f}%)', zorder=3)
    
    # Plot capital market line (from risk-free rate through constrained tangency portfolio)
    if tangency_port_constrained[1] > 0:
        cml_slope = (tangency_port_constrained[0] - risk_free_rate) / tangency_port_constrained[1]
        max_vol = max(
            frontier_df_constrained['volatility'].max() if len(frontier_df_constrained) > 0 else 0,
            individual_volatilities.max()
        )
        x_cml = np.linspace(0, max_vol * 1.1, 100)
        y_cml = risk_free_rate + cml_slope * x_cml
        ax.plot(x_cml, y_cml, 'g--', linewidth=1.5, alpha=0.7, label='Capital Market Line', zorder=4)
    
    # Add interpretation text about CAPM
    # Calculate distance between market index and constrained efficient frontier (more realistic)
    # Find closest point on constrained efficient frontier to market index
    if len(frontier_df_constrained) > 0:
        frontier_distances = np.sqrt(
            (frontier_df_constrained['volatility'] - market_index_vol)**2 + 
            (frontier_df_constrained['return'] - market_index_return)**2
        )
        min_distance = frontier_distances.min()
        closest_idx = frontier_distances.idxmin()
        closest_point_vol = frontier_df_constrained.loc[closest_idx, 'volatility']
        closest_point_ret = frontier_df_constrained.loc[closest_idx, 'return']
    else:
        min_distance = np.inf
        closest_point_vol = 0
        closest_point_ret = 0
    
    # Interpretation: If market index is close to efficient frontier, CAPM holds
    # Threshold: if distance is less than 5% of market volatility, consider it "close"
    distance_threshold = market_index_vol * 0.05
    if min_distance < distance_threshold:
        capm_status = "CAPM HOLDS: Market index overlaps well with efficient frontier"
        capm_color = 'green'
    else:
        capm_status = "CAPM DOES NOT HOLD: Market index does not overlap with efficient frontier"
        capm_color = 'red'
    
    # Add text box with interpretation
    textstr = f'CAPM Interpretation:\n{capm_status}\n\nDistance to frontier: {min_distance:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor=capm_color, linewidth=2)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, color=capm_color, fontweight='bold')
    
    ax.set_xlabel('Volatility (Standard Deviation, %)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Mean-Variance Efficient Frontier (Long-Only) with All Stocks and Market Index', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f" Saved: {output_path}")
    logger.info(f"  Market index distance to efficient frontier: {min_distance:.4f}")
    logger.info(f"  CAPM status: {capm_status}")


def plot_efficient_frontier_constrained_only(
    frontier_df_constrained: pd.DataFrame,
    min_var_port_constrained: Tuple[float, float],
    tangency_port_constrained: Tuple[float, float],
    equal_weighted_port: Tuple[float, float],
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    market_index_return: float,
    market_index_vol: float,
    risk_free_rate: float,
    output_path: str
) -> None:
    """
    Plot efficient frontier with constrained (long-only) portfolios only.
    
    This function creates a plot showing only the constrained efficient frontier,
    which represents economically achievable investment opportunities. Unconstrained
    (short-selling) portfolios are excluded as they are not economically meaningful.
    
    This graph shows:
    - X-axis: Volatility (standard deviation)
    - Y-axis: Expected return
    - All individual stocks as scatter points
    - Constrained efficient frontier (long-only, solid line)
    - Market index (MSCI Europe) point
    - Constrained minimum-variance and tangency portfolios
    - Interpretation: If efficient frontier overlaps well with market index, CAPM holds.
    
    Parameters
    ----------
    frontier_df_constrained : pd.DataFrame
        Constrained efficient frontier data (long-only) with 'volatility' and 'return' columns
    min_var_port_constrained : tuple
        (return, volatility) for constrained minimum-variance portfolio (long-only)
    tangency_port_constrained : tuple
        (return, volatility) for constrained tangency portfolio (long-only)
    equal_weighted_port : tuple
        (return, volatility) for equal-weighted portfolio
    expected_returns : pd.Series
        Expected returns for all individual stocks
    cov_matrix : pd.DataFrame
        Covariance matrix for calculating individual stock volatilities
    market_index_return : float
        Expected return of market index (MSCI Europe)
    market_index_vol : float
        Volatility of market index (MSCI Europe)
    risk_free_rate : float
        Risk-free rate (German 3-month Bund, EUR, for all countries)
    output_path : str
        Path to save plot
    """
    logger.info("Creating constrained-only efficient frontier plot...")
    
    # Verify units: returns are in percentage, covariance is in percentage²
    individual_volatilities = np.sqrt(np.diag(cov_matrix.values))
    logger.info(f"  Unit check - Individual stock volatilities: min={individual_volatilities.min():.2f}%, max={individual_volatilities.max():.2f}%")
    logger.info(f"  Unit check - Expected returns: min={expected_returns.min():.2f}%, max={expected_returns.max():.2f}%")
    logger.info(f"  Unit check - Constrained frontier volatility: min={frontier_df_constrained['volatility'].min():.2f}%, max={frontier_df_constrained['volatility'].max():.2f}%")
    logger.info(f"  Unit check - Market index: return={market_index_return:.2f}%, vol={market_index_vol:.2f}%")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot ALL individual stocks
    ax.scatter(individual_volatilities, expected_returns.values, 
              alpha=0.4, s=30, color='gray', label=f'Individual Stocks (n={len(expected_returns)})',
              edgecolors='black', linewidths=0.3)
    
    # Plot constrained efficient frontier (long-only, solid line)
    ax.plot(frontier_df_constrained['volatility'], frontier_df_constrained['return'], 
            'b-', linewidth=2.5, label='Efficient Frontier (Long-Only)', zorder=5)
    
    # Plot market index (MSCI Europe)
    ax.scatter(market_index_vol, market_index_return, 
              s=200, marker='*', color='red', edgecolors='darkred', linewidths=2,
              label='Market Index (MSCI Europe)', zorder=6)
    
    # Plot constrained minimum-variance portfolio (long-only, red circle)
    ax.plot(min_var_port_constrained[1], min_var_port_constrained[0], 
            'ro', markersize=12, label='Min-Variance Portfolio (Long-Only)', zorder=5)
    
    # Plot constrained tangency portfolio (long-only, green circle)
    ax.plot(tangency_port_constrained[1], tangency_port_constrained[0], 
            'go', markersize=12, label='Optimal Risky Portfolio (Tangency, Long-Only)', zorder=5)
    
    # Plot equal-weighted portfolio (purple diamond)
    ax.plot(equal_weighted_port[1], equal_weighted_port[0], 
            'mD', markersize=12, label='Equal-Weighted Portfolio', zorder=5)
    
    # Plot risk-free rate
    ax.axhline(y=risk_free_rate, color='k', linestyle='--', linewidth=1, 
              label=f'Risk-Free Rate ({risk_free_rate:.4f}%)', zorder=3)
    
    # Plot capital market line (from risk-free rate through constrained tangency portfolio)
    if tangency_port_constrained[1] > 0:
        cml_slope = (tangency_port_constrained[0] - risk_free_rate) / tangency_port_constrained[1]
        max_vol = max(
            frontier_df_constrained['volatility'].max() if len(frontier_df_constrained) > 0 else 0,
            individual_volatilities.max()
        )
        x_cml = np.linspace(0, max_vol * 1.1, 100)
        y_cml = risk_free_rate + cml_slope * x_cml
        ax.plot(x_cml, y_cml, 'g--', linewidth=1.5, alpha=0.7, label='Capital Market Line', zorder=4)
    
    # Add interpretation text about CAPM
    # Calculate distance between market index and constrained efficient frontier
    if len(frontier_df_constrained) > 0:
        frontier_distances = np.sqrt(
            (frontier_df_constrained['volatility'] - market_index_vol)**2 + 
            (frontier_df_constrained['return'] - market_index_return)**2
        )
        min_distance = frontier_distances.min()
        closest_idx = frontier_distances.idxmin()
        closest_point_vol = frontier_df_constrained.loc[closest_idx, 'volatility']
        closest_point_ret = frontier_df_constrained.loc[closest_idx, 'return']
    else:
        min_distance = np.inf
        closest_point_vol = 0
        closest_point_ret = 0
    
    # Interpretation: If market index is close to efficient frontier, CAPM holds
    distance_threshold = market_index_vol * 0.05
    if min_distance < distance_threshold:
        capm_status = "CAPM HOLDS: Market index overlaps well with efficient frontier"
        capm_color = 'green'
    else:
        capm_status = "CAPM DOES NOT HOLD: Market index does not overlap with efficient frontier"
        capm_color = 'red'
    
    # Add text box with interpretation
    textstr = f'CAPM Interpretation:\n{capm_status}\n\nDistance to frontier: {min_distance:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor=capm_color, linewidth=2)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, color=capm_color, fontweight='bold')
    
    ax.set_xlabel('Volatility (Standard Deviation, %)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Mean-Variance Efficient Frontier (Constrained, Long-Only)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f" Saved constrained-only plot: {output_path}")
    logger.info(f"  Market index distance to efficient frontier: {min_distance:.4f}")
    logger.info(f"  CAPM status: {capm_status}")


def plot_efficient_frontier_realistic_short(
    frontier_df_realistic: pd.DataFrame,
    frontier_df_unconstrained: pd.DataFrame,
    min_var_port_realistic: Tuple[float, float],
    min_var_port_unconstrained: Tuple[float, float],
    tangency_port_realistic: Tuple[float, float],
    tangency_port_unconstrained: Tuple[float, float],
    equal_weighted_port: Tuple[float, float],
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    market_index_return: float,
    market_index_vol: float,
    risk_free_rate: float,
    error_flags: Dict,
    output_path: str
) -> None:
    """
    Plot efficient frontier with realistic short-selling constraints.
    
    Shows comparison between realistic (with constraints) and theoretical (unconstrained)
    short-selling frontiers, with error flags and explanations.
    
    Parameters
    ----------
    frontier_df_realistic : pd.DataFrame
        Realistic efficient frontier with constraints
    frontier_df_unconstrained : pd.DataFrame
        Theoretical unconstrained efficient frontier
    min_var_port_realistic : tuple
        (return, volatility) for realistic minimum-variance portfolio
    min_var_port_unconstrained : tuple
        (return, volatility) for theoretical minimum-variance portfolio
    tangency_port_realistic : tuple
        (return, volatility) for realistic tangency portfolio
    tangency_port_unconstrained : tuple
        (return, volatility) for theoretical tangency portfolio
    equal_weighted_port : tuple
        (return, volatility) for equal-weighted portfolio
    expected_returns : pd.Series
        Expected returns for all stocks
    cov_matrix : pd.DataFrame
        Covariance matrix
    market_index_return : float
        Expected return of market index
    market_index_vol : float
        Volatility of market index
    risk_free_rate : float
        Risk-free rate
    error_flags : dict
        Dictionary with error flags and explanations
    output_path : str
        Path to save plot
    """
    logger.info("Creating realistic short-selling efficient frontier plot...")
    
    individual_volatilities = np.sqrt(np.diag(cov_matrix.values))
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot individual stocks
    ax.scatter(individual_volatilities, expected_returns.values,
              alpha=0.3, s=20, color='gray', label=f'Individual Stocks (n={len(expected_returns)})',
              edgecolors='black', linewidths=0.2)
    
    # Plot realistic efficient frontier (with constraints)
    if len(frontier_df_realistic) > 0:
        ax.plot(frontier_df_realistic['volatility'], frontier_df_realistic['return'],
                'b-', linewidth=2.5, label='Efficient Frontier (Realistic Short-Selling)', zorder=5)
        
        # Flag unrealistic portfolios
        unrealistic = frontier_df_realistic[frontier_df_realistic['volatility'] < MIN_VOLATILITY_THRESHOLD]
        if len(unrealistic) > 0:
            ax.scatter(unrealistic['volatility'], unrealistic['return'],
                      s=100, marker='x', color='red', linewidths=3,
                      label=f'Unrealistic Volatility (<{MIN_VOLATILITY_THRESHOLD}%)', zorder=6)
    
    # Plot theoretical unconstrained frontier (for comparison)
    if len(frontier_df_unconstrained) > 0:
        ax.plot(frontier_df_unconstrained['volatility'], frontier_df_unconstrained['return'],
                'b--', linewidth=1.5, alpha=0.5, label='Efficient Frontier (Theoretical, Unconstrained)', zorder=4)
    
    # Plot market index
    ax.scatter(market_index_vol, market_index_return,
              s=200, marker='*', color='red', edgecolors='darkred', linewidths=2,
              label='Market Index (MSCI Europe)', zorder=6)
    
    # Plot portfolios
    ax.plot(min_var_port_realistic[1], min_var_port_realistic[0],
            'ro', markersize=12, label='Min-Variance (Realistic)', zorder=5)
    ax.plot(min_var_port_unconstrained[1], min_var_port_unconstrained[0],
            'rs', markersize=10, alpha=0.6, label='Min-Variance (Theoretical)', zorder=4)
    
    ax.plot(tangency_port_realistic[1], tangency_port_realistic[0],
            'go', markersize=12, label='Tangency (Realistic)', zorder=5)
    ax.plot(tangency_port_unconstrained[1], tangency_port_unconstrained[0],
            'gs', markersize=10, alpha=0.6, label='Tangency (Theoretical)', zorder=4)
    
    ax.plot(equal_weighted_port[1], equal_weighted_port[0],
            'mD', markersize=12, label='Equal-Weighted', zorder=5)
    
    # Risk-free rate
    ax.axhline(y=risk_free_rate, color='k', linestyle='--', linewidth=1,
              label=f'Risk-Free Rate ({risk_free_rate:.4f}%)', zorder=3)
    
    # Capital market line (from realistic tangency)
    if tangency_port_realistic[1] > 0:
        cml_slope = (tangency_port_realistic[0] - risk_free_rate) / tangency_port_realistic[1]
        max_vol = max(
            frontier_df_realistic['volatility'].max() if len(frontier_df_realistic) > 0 else 0,
            individual_volatilities.max()
        )
        x_cml = np.linspace(0, max_vol * 1.1, 100)
        y_cml = risk_free_rate + cml_slope * x_cml
        ax.plot(x_cml, y_cml, 'g--', linewidth=1.5, alpha=0.7, label='Capital Market Line', zorder=4)
    
    # Add error flags and explanations
    error_text = []
    if error_flags.get('unrealistic_volatility_count', 0) > 0:
        error_text.append(f" {error_flags['unrealistic_volatility_count']} portfolios with unrealistic volatility")
    if error_flags.get('excessive_leverage_count', 0) > 0:
        error_text.append(f" {error_flags['excessive_leverage_count']} portfolios with excessive leverage")
    if error_flags.get('optimization_failed_count', 0) > 0:
        error_text.append(f" {error_flags['optimization_failed_count']} optimization failures")
    
    if error_text:
        error_str = '\n'.join(error_text)
        props = dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2)
        ax.text(0.02, 0.02, f'Error Flags:\n{error_str}', transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', bbox=props, color='red', fontweight='bold')
    
    # Add constraint explanations
    constraint_text = (
        "Realistic Constraints Applied:\n"
        f"• Transaction costs: {TRANSACTION_COST_RATE*100:.2f}% of gross exposure\n"
        f"• Borrowing cost spread: {BORROWING_COST_SPREAD*100:.2f}% above risk-free rate\n"
        f"• Max short position: {MAX_SHORT_POSITION_PCT*100:.0f}% per stock\n"
        f"• Max gross exposure: {MAX_REALISTIC_GROSS_EXPOSURE*100:.0f}%\n"
        f"• Min realistic volatility: {MIN_VOLATILITY_THRESHOLD}%"
    )
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='blue', linewidth=1)
    ax.text(0.98, 0.98, constraint_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=props, color='blue')
    
    ax.set_xlabel('Volatility (Standard Deviation, %)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Efficient Frontier: Realistic vs. Theoretical Short-Selling', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f" Saved realistic short-selling plot: {output_path}")


def plot_efficient_frontier_top15_short_selling(
    frontier_df: pd.DataFrame,
    min_var_port: Tuple[float, float],
    tangency_port: Tuple[float, float],
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    output_path: str
) -> None:
    """
    Plot efficient frontier for top 15 stocks with short-selling allowed.
    
    Parameters
    ----------
    frontier_df : pd.DataFrame
        Efficient frontier data with 'volatility' and 'return' columns
    min_var_port : tuple
        (return, volatility) for minimum-variance portfolio
    tangency_port : tuple
        (return, volatility) for tangency portfolio
    expected_returns : pd.Series
        Expected returns for all stocks
    cov_matrix : pd.DataFrame
        Covariance matrix
    risk_free_rate : float
        Risk-free rate
    output_path : str
        Path to save plot
    """
    logger.info("Creating efficient frontier plot for top 15 stocks (short-selling allowed)...")
    
    individual_volatilities = np.sqrt(np.diag(cov_matrix.values))
    
    _, ax = plt.subplots(figsize=(14, 10))
    
    # Plot individual stocks
    ax.scatter(individual_volatilities, expected_returns.values, 
              alpha=0.6, s=50, color='gray', label=f'Individual Stocks (n={len(expected_returns)})',
              edgecolors='black', linewidths=0.5, zorder=3)
    
    # Plot efficient frontier (with short-selling)
    if len(frontier_df) > 0:
        ax.plot(frontier_df['volatility'], frontier_df['return'], 
                'b-', linewidth=2.5, label='Efficient Frontier (Short-Selling Allowed)', zorder=5)
    
    # Plot minimum-variance portfolio
    ax.plot(min_var_port[1], min_var_port[0], 
            'ro', markersize=12, label='Min-Variance Portfolio', zorder=6)
    
    # Plot tangency portfolio
    ax.plot(tangency_port[1], tangency_port[0], 
            'go', markersize=12, label='Tangency Portfolio', zorder=6)
    
    # Plot risk-free rate
    ax.axhline(y=risk_free_rate, color='k', linestyle='--', linewidth=1, 
              label=f'Risk-Free Rate ({risk_free_rate:.4f}%)', zorder=3)
    
    # Plot capital market line
    if tangency_port[1] > 0:
        cml_slope = (tangency_port[0] - risk_free_rate) / tangency_port[1]
        max_vol = max(
            frontier_df['volatility'].max() if len(frontier_df) > 0 else 0,
            individual_volatilities.max()
        )
        x_cml = np.linspace(0, max_vol * 1.1, 100)
        y_cml = risk_free_rate + cml_slope * x_cml
        ax.plot(x_cml, y_cml, 'g--', linewidth=1.5, alpha=0.7, label='Capital Market Line', zorder=4)
    
    ax.set_xlabel('Volatility (Standard Deviation, %)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Efficient Frontier: Top 15 Stocks by Market Cap (Short-Selling Allowed)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f" Saved top 15 stocks efficient frontier plot: {output_path}")


def run_portfolio_optimization() -> Dict:
    """
    Run complete portfolio optimization analysis.
    
    **Important Note on Short-Selling Constraints:**
    
    The minimum-variance portfolio is computed allowing short-selling (unconstrained)
    to demonstrate the theoretical mathematical result. However, this yields
    economically unachievable results (volatility approaches ~0.002%), which fully
    inflates the Sharpe ratio due to the unconstrained ability to short sell and
    achieve near-perfect hedging.
    
    In practice, such low volatility is not achievable due to:
    - Regulatory restrictions on short selling
    - Transaction costs (bid-ask spreads, commissions)
    - Margin requirements and borrowing costs
    - Liquidity constraints (not all stocks can be shorted in large quantities)
    - Market impact of large short positions
    
    Therefore, the efficient frontier and tangency portfolio are computed WITHOUT
    short-selling (long-only constraint). While this is not consistent with CAPM's
    purely theoretical assumptions (which assume frictionless markets), the objective
    of this analysis is to yield economically sound investment advice, thus motivating
    the choice of using the constrained efficient frontier.
    
    Returns
    -------
    dict
        Dictionary with all optimization results
    """
    logger.info("="*70)
    logger.info("PORTFOLIO OPTIMIZATION ANALYSIS")
    logger.info("="*70)
    
    # Load returns panel
    panel_file = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
    if not os.path.exists(panel_file):
        logger.error("Returns panel not found. Run returns processing first.")
        return {}
    
    panel_df = pd.read_csv(panel_file, parse_dates=['date'])
    logger.info(f"Loaded returns panel: {len(panel_df)} rows, {panel_df['ticker'].nunique()} stocks")
    
    # Calculate expected returns and covariance
    expected_returns, cov_matrix = calculate_expected_returns_and_covariance(panel_df)
    
    # Get average risk-free rate
    # NOTE: All countries use German 3-month Bund (EUR) as the risk-free rate.
    # This is consistent across all calculations in the analysis.
    avg_rf_rate = panel_df['riskfree_rate'].mean()
    logger.info(f"Average risk-free rate: {avg_rf_rate:.4f}% (monthly, German 3-month Bund, EUR)")
    
    # Find minimum-variance portfolio (unconstrained - with short selling)
    # NOTE: This is computed with short-selling to show the theoretical result,
    # but the extremely low volatility (~0.002%) is not economically achievable.
    # The unconstrained solution enables near-perfect hedging through negative
    # weights, which fully inflates the Sharpe ratio. This is reported for
    # completeness but not used for investment recommendations.
    logger.info("Calculating unconstrained minimum-variance portfolio (with short selling)...")
    min_var_weights_unconstrained, min_var_return_unconstrained, min_var_vol_unconstrained = find_minimum_variance_portfolio(
        expected_returns, cov_matrix, allow_short=True
    )
    
    # Warn if volatility is unrealistically low (theoretical result from short selling)
    if min_var_vol_unconstrained < 0.1 and min_var_vol_unconstrained > 0:
        logger.warning(f"  Unconstrained minimum-variance portfolio has unrealistically low volatility ({min_var_vol_unconstrained:.6f}%)")
        logger.warning("   This is a theoretical result from short selling - not achievable in practice.")
        logger.warning("   Sharpe ratio will be set to NaN as it's not meaningful when volatility approaches zero.")
        logger.warning("   In practice, transaction costs, margin requirements, and liquidity constraints")
        logger.warning("   would prevent such low volatility (expected: 0.5-1.5% monthly).")
    
    # Find minimum-variance portfolio (constrained - long-only)
    logger.info("Calculating constrained minimum-variance portfolio (long-only)...")
    min_var_weights_constrained, min_var_return_constrained, min_var_vol_constrained = find_minimum_variance_portfolio(
        expected_returns, cov_matrix, allow_short=False
    )
    
    # Find tangency portfolio (constrained - long-only, more realistic for investment)
    # NOTE: Computed WITHOUT short-selling (long-only constraint) to yield
    # economically meaningful results. While CAPM assumes frictionless markets
    # with unlimited short-selling, real-world constraints (transaction costs,
    # regulatory restrictions, liquidity) make the unconstrained solution
    # economically unachievable. The constrained solution provides actionable
    # investment advice.
    logger.info("Calculating constrained tangency portfolio (long-only)...")
    tangency_weights_constrained, tangency_return_constrained, tangency_vol_constrained = find_tangency_portfolio(
        expected_returns, cov_matrix, avg_rf_rate, allow_short=False
    )
    
    # Find tangency portfolio (unconstrained - with short selling)
    logger.info("Calculating unconstrained tangency portfolio (with short selling)...")
    tangency_weights_unconstrained, tangency_return_unconstrained, tangency_vol_unconstrained = find_tangency_portfolio(
        expected_returns, cov_matrix, avg_rf_rate, allow_short=True
    )
    
    # Calculate efficient frontier (unconstrained - with short selling)
    # Pass tangency return as upper bound to avoid recalculating
    logger.info("Calculating unconstrained efficient frontier (with short selling)...")
    frontier_df_unconstrained = calculate_efficient_frontier(
        expected_returns, cov_matrix, n_points=50, allow_short=True,
        tangency_return=tangency_return_unconstrained
    )
    
    # Calculate efficient frontier (constrained - long-only)
    # NOTE: Computed WITHOUT short-selling for the same reasons as the tangency
    # portfolio. The constrained efficient frontier represents realistic investment
    # opportunities that can actually be implemented in practice.
    logger.info("Calculating constrained efficient frontier (long-only)...")
    frontier_df_constrained = calculate_efficient_frontier(
        expected_returns, cov_matrix, n_points=50, allow_short=False
    )
    
    # Calculate diversification benefits (includes equal-weighted portfolio)
    div_benefits = calculate_diversification_benefits(expected_returns, cov_matrix)
    
    # Calculate equal-weighted portfolio explicitly for plotting
    n_stocks = len(expected_returns)
    equal_weights = np.ones(n_stocks) / n_stocks
    equal_weighted_return = portfolio_return(equal_weights, expected_returns)
    equal_weighted_vol = np.sqrt(portfolio_variance(equal_weights, cov_matrix))
    
    # Calculate market index (MSCI Europe) return and volatility from panel
    # Market index returns are in the 'msci_index_return' column (already in percentage form)
    # Get unique market returns (same for all stocks in each month)
    market_returns = panel_df.groupby('date')['msci_index_return'].first().dropna()
    market_index_return = market_returns.mean()  # Expected return (mean historical, already in %)
    market_index_vol = market_returns.std()  # Volatility (already in percentage form, no need to multiply by 100)
    logger.info(f"Market index (MSCI Europe): Return={market_index_return:.4f}%, Volatility={market_index_vol:.4f}%")
    
    # Create plot with all stocks and market index
    plot_path = os.path.join(RESULTS_FIGURES_DIR, "efficient_frontier.png")
    plot_efficient_frontier(
        frontier_df_constrained,
        frontier_df_unconstrained,
        (min_var_return_constrained, min_var_vol_constrained),
        (min_var_return_unconstrained, min_var_vol_unconstrained),
        (tangency_return_constrained, tangency_vol_constrained),
        (tangency_return_unconstrained, tangency_vol_unconstrained),
        (equal_weighted_return, equal_weighted_vol),
        expected_returns,
        cov_matrix,
        market_index_return,
        market_index_vol,
        avg_rf_rate,
        plot_path
    )
    
    # Save results
    results_df = pd.DataFrame({
        'portfolio': [
            'Minimum-Variance (Constrained)', 
            'Minimum-Variance (Unconstrained)', 
            'Optimal Risky (Tangency, Constrained)', 
            'Optimal Risky (Tangency, Unconstrained)',
            'Equal-Weighted'
        ],
        'expected_return': [
            min_var_return_constrained,
            min_var_return_unconstrained,
            tangency_return_constrained,
            tangency_return_unconstrained,
            equal_weighted_return
        ],
        'volatility': [
            min_var_vol_constrained,
            min_var_vol_unconstrained,
            tangency_vol_constrained,
            tangency_vol_unconstrained,
            equal_weighted_vol
        ],
        'sharpe_ratio': [
            # Constrained min-var
            (min_var_return_constrained - avg_rf_rate) / min_var_vol_constrained if min_var_vol_constrained > 0 else 0,
            # Unconstrained min-var: when volatility is extremely low (<0.1%),
            # the Sharpe ratio becomes meaningless. Set to NaN to indicate not meaningful.
            # In practice, such low volatility is impossible due to transaction costs and constraints.
            (min_var_return_unconstrained / min_var_vol_unconstrained if min_var_vol_unconstrained >= 0.1 else np.nan) if min_var_vol_unconstrained > 0 else 0,
            # Constrained tangency portfolio
            (tangency_return_constrained - avg_rf_rate) / tangency_vol_constrained if tangency_vol_constrained > 0 else 0,
            # Unconstrained tangency portfolio
            (tangency_return_unconstrained - avg_rf_rate) / tangency_vol_unconstrained if tangency_vol_unconstrained > 0 else 0,
            # Equal-weighted
            (equal_weighted_return - avg_rf_rate) / equal_weighted_vol if equal_weighted_vol > 0 else 0
        ]
    })
    
    results_file = os.path.join(RESULTS_REPORTS_DIR, "portfolio_optimization_results.csv")
    results_df.to_csv(results_file, index=False)
    logger.info(f" Saved: {results_file}")
    
    # Save diversification metrics
    div_df = pd.DataFrame([div_benefits])
    div_file = os.path.join(RESULTS_REPORTS_DIR, "diversification_benefits.csv")
    div_df.to_csv(div_file, index=False)
    logger.info(f" Saved: {div_file}")
    
    # Save constrained-only results to separate folder (for paper)
    # This contains only economically achievable portfolios (long-only, no short-selling)
    constrained_results_df = pd.DataFrame({
        'portfolio': [
            'Minimum-Variance (Constrained)', 
            'Optimal Risky (Tangency, Constrained)', 
            'Equal-Weighted'
        ],
        'expected_return': [
            min_var_return_constrained,
            tangency_return_constrained,
            equal_weighted_return
        ],
        'volatility': [
            min_var_vol_constrained,
            tangency_vol_constrained,
            equal_weighted_vol
        ],
        'sharpe_ratio': [
            (min_var_return_constrained - avg_rf_rate) / min_var_vol_constrained if min_var_vol_constrained > 0 else 0,
            (tangency_return_constrained - avg_rf_rate) / tangency_vol_constrained if tangency_vol_constrained > 0 else 0,
            (equal_weighted_return - avg_rf_rate) / equal_weighted_vol if equal_weighted_vol > 0 else 0
        ]
    })
    constrained_results_file = os.path.join(RESULTS_REPORTS_CONSTRAINED_DIR, "portfolio_optimization_results.csv")
    constrained_results_df.to_csv(constrained_results_file, index=False)
    logger.info(f" Saved constrained-only results: {constrained_results_file}")
    
    # Plot constrained-only efficient frontier
    constrained_plot_path = os.path.join(RESULTS_FIGURES_CONSTRAINED_DIR, "efficient_frontier.png")
    plot_efficient_frontier_constrained_only(
        frontier_df_constrained,
        (min_var_return_constrained, min_var_vol_constrained),
        (tangency_return_constrained, tangency_vol_constrained),
        (equal_weighted_return, equal_weighted_vol),
        expected_returns,
        cov_matrix,
        market_index_return,
        market_index_vol,
        avg_rf_rate,
        constrained_plot_path
    )
    
    # Calculate realistic short-selling portfolios (with transaction costs, margin, borrowing costs)
    logger.info("="*70)
    logger.info("CALCULATING REALISTIC SHORT-SELLING PORTFOLIOS")
    logger.info("="*70)
    logger.info("Applying realistic constraints:")
    logger.info(f"  Transaction costs: {TRANSACTION_COST_RATE*100:.2f}% of gross exposure")
    logger.info(f"  Borrowing cost spread: {BORROWING_COST_SPREAD*100:.2f}% above risk-free rate")
    logger.info(f"  Max short position: {MAX_SHORT_POSITION_PCT*100:.0f}% per stock")
    logger.info(f"  Max gross exposure: {MAX_REALISTIC_GROSS_EXPOSURE*100:.0f}%")
    logger.info(f"  Min realistic volatility: {MIN_VOLATILITY_THRESHOLD}%")
    
    # Find realistic minimum-variance portfolio
    min_var_weights_realistic, min_var_return_realistic, min_var_vol_realistic, min_var_errors = \
        find_minimum_variance_portfolio_realistic_short(expected_returns, cov_matrix, avg_rf_rate)
    
    # Find realistic tangency portfolio
    tangency_weights_realistic, tangency_return_realistic, tangency_vol_realistic, tangency_errors = \
        find_tangency_portfolio_realistic_short(expected_returns, cov_matrix, avg_rf_rate)
    
    # Calculate realistic efficient frontier (reduced points for faster computation)
    logger.info("Calculating realistic short-selling efficient frontier (this may take a few minutes)...")
    frontier_df_realistic, frontier_errors = calculate_efficient_frontier_realistic_short(
        expected_returns, cov_matrix, avg_rf_rate, n_points=20  # Reduced to 20 for faster computation
    )
    
    # Combine error flags
    all_error_flags = {
        'min_var_errors': min_var_errors,
        'tangency_errors': tangency_errors,
        'frontier_errors': frontier_errors,
        'unrealistic_volatility_count': frontier_errors.get('unrealistic_volatility_count', 0),
        'excessive_leverage_count': frontier_errors.get('excessive_leverage_count', 0),
        'large_short_position_count': frontier_errors.get('large_short_position_count', 0),
        'optimization_failed_count': frontier_errors.get('optimization_failed_count', 0)
    }
    
    # Calculate effective risk-free rates for realistic short-selling portfolios
    # (accounting for borrowing costs on short positions)
    min_var_long_exposure = np.sum(np.maximum(min_var_weights_realistic, 0))
    min_var_short_exposure = np.sum(np.abs(np.minimum(min_var_weights_realistic, 0)))
    min_var_total_exposure = min_var_long_exposure + min_var_short_exposure
    min_var_effective_rf = (
        (min_var_long_exposure * avg_rf_rate + min_var_short_exposure * (avg_rf_rate + BORROWING_COST_SPREAD)) / min_var_total_exposure
        if min_var_total_exposure > 0 else avg_rf_rate
    )
    
    tangency_long_exposure = np.sum(np.maximum(tangency_weights_realistic, 0))
    tangency_short_exposure = np.sum(np.abs(np.minimum(tangency_weights_realistic, 0)))
    tangency_total_exposure = tangency_long_exposure + tangency_short_exposure
    tangency_effective_rf = (
        (tangency_long_exposure * avg_rf_rate + tangency_short_exposure * (avg_rf_rate + BORROWING_COST_SPREAD)) / tangency_total_exposure
        if tangency_total_exposure > 0 else avg_rf_rate
    )
    
    # Save realistic short-selling results
    realistic_results_df = pd.DataFrame({
        'portfolio': [
            'Minimum-Variance (Realistic Short-Selling)',
            'Optimal Risky (Tangency, Realistic Short-Selling)',
            'Minimum-Variance (Theoretical, Unconstrained)',
            'Optimal Risky (Tangency, Theoretical, Unconstrained)'
        ],
        'expected_return': [
            min_var_return_realistic,
            tangency_return_realistic,
            min_var_return_unconstrained,
            tangency_return_unconstrained
        ],
        'volatility': [
            min_var_vol_realistic,
            tangency_vol_realistic,
            min_var_vol_unconstrained,
            tangency_vol_unconstrained
        ],
        'sharpe_ratio': [
            (min_var_return_realistic - min_var_effective_rf) / min_var_vol_realistic if min_var_vol_realistic > 0 else 0,
            (tangency_return_realistic - tangency_effective_rf) / tangency_vol_realistic if tangency_vol_realistic > 0 else 0,
            np.nan if min_var_vol_unconstrained < 0.1 else (min_var_return_unconstrained - avg_rf_rate) / min_var_vol_unconstrained if min_var_vol_unconstrained > 0 else 0,
            (tangency_return_unconstrained - avg_rf_rate) / tangency_vol_unconstrained if tangency_vol_unconstrained > 0 else 0
        ],
        'gross_exposure': [
            np.sum(np.abs(min_var_weights_realistic)),
            np.sum(np.abs(tangency_weights_realistic)),
            np.sum(np.abs(min_var_weights_unconstrained)),
            np.sum(np.abs(tangency_weights_unconstrained))
        ],
        'has_errors': [
            bool(min_var_errors.get('explanations', [])),
            bool(tangency_errors.get('explanations', [])),
            min_var_vol_unconstrained < MIN_VOLATILITY_THRESHOLD,
            tangency_vol_unconstrained < MIN_VOLATILITY_THRESHOLD
        ]
    })
    
    realistic_results_file = os.path.join(RESULTS_REPORTS_DIR, "portfolio_optimization_realistic_short.csv")
    realistic_results_df.to_csv(realistic_results_file, index=False)
    logger.info(f" Saved realistic short-selling results: {realistic_results_file}")
    
    # Plot realistic short-selling efficient frontier
    realistic_plot_path = os.path.join(RESULTS_FIGURES_DIR, "efficient_frontier_realistic_short.png")
    plot_efficient_frontier_realistic_short(
        frontier_df_realistic,
        frontier_df_unconstrained,
        (min_var_return_realistic, min_var_vol_realistic),
        (min_var_return_unconstrained, min_var_vol_unconstrained),
        (tangency_return_realistic, tangency_vol_realistic),
        (tangency_return_unconstrained, tangency_vol_unconstrained),
        (equal_weighted_return, equal_weighted_vol),
        expected_returns,
        cov_matrix,
        market_index_return,
        market_index_vol,
        avg_rf_rate,
        all_error_flags,
        realistic_plot_path
    )
    
    # NEW: Calculate efficient frontier for top 15 stocks with short-selling
    # Skip this if it's taking too long - it's optional
    logger.info("="*70)
    logger.info("CALCULATING EFFICIENT FRONTIER FOR TOP 15 STOCKS (SHORT-SELLING ALLOWED)")
    logger.info("="*70)
    logger.info("Note: This step may be skipped if market cap fetching is slow")
    
    # Select top 15 stocks by market cap (with timeout to prevent hanging)
    logger.info("Selecting top 15 stocks by market cap (this may take a moment if fetching market cap data)...")
    top15_tickers = pd.Series(dtype=str)
    
    try:
        # Use platform-agnostic timeout (signal.SIGALRM only works on Unix)
        if platform.system() != 'Windows':
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Market cap selection timed out after 30 seconds")
            
            # Set timeout for market cap selection (30 seconds - shorter to avoid hanging)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            try:
                top15_tickers = select_top_15_stocks_by_market_cap(panel_df, n_stocks=15)
            finally:
                signal.alarm(0)  # Cancel timeout
        else:
            # Windows doesn't support SIGALRM, use ThreadPoolExecutor with timeout instead
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(select_top_15_stocks_by_market_cap, panel_df, 15)
            try:
                top15_tickers = future.result(timeout=30)
            except FutureTimeoutError:
                logger.warning("  Market cap selection timed out after 30 seconds")
                logger.warning("Skipping top 15 stocks efficient frontier calculation.")
                top15_tickers = pd.Series(dtype=str)
            finally:
                executor.shutdown(wait=False)
        
    except (TimeoutError, KeyboardInterrupt) as e:
        logger.warning(f"  Market cap selection timed out or was interrupted: {e}")
        logger.warning("Skipping top 15 stocks efficient frontier calculation.")
        top15_tickers = pd.Series(dtype=str)
    except Exception as e:
        logger.warning(f"  Error selecting top 15 stocks: {e}")
        logger.warning("Skipping top 15 stocks efficient frontier calculation.")
        top15_tickers = pd.Series(dtype=str)
    
    if len(top15_tickers) > 0:
        # Calculate expected returns and covariance for top 15 stocks
        expected_returns_top15, cov_matrix_top15 = calculate_expected_returns_and_covariance(
            panel_df, selected_tickers=top15_tickers
        )
        
        # Calculate efficient frontier with short-selling
        logger.info("Calculating efficient frontier with short-selling for top 15 stocks...")
        # Use fewer points for speed (short-selling case already limits to 15 points internally)
        frontier_df_top15 = calculate_efficient_frontier(
            expected_returns_top15, cov_matrix_top15, n_points=15, allow_short=True
        )
        
        # Find minimum-variance portfolio (with short-selling)
        min_var_weights_top15, min_var_return_top15, min_var_vol_top15 = find_minimum_variance_portfolio(
            expected_returns_top15, cov_matrix_top15, allow_short=True
        )
        
        # Find tangency portfolio (with short-selling)
        tangency_weights_top15, tangency_return_top15, tangency_vol_top15 = find_tangency_portfolio(
            expected_returns_top15, cov_matrix_top15, avg_rf_rate, allow_short=True
        )
        
        # Plot efficient frontier
        top15_plot_path = os.path.join(RESULTS_FIGURES_DIR, "efficient_frontier_top15_short_selling.png")
        plot_efficient_frontier_top15_short_selling(
            frontier_df_top15,
            (min_var_return_top15, min_var_vol_top15),
            (tangency_return_top15, tangency_vol_top15),
            expected_returns_top15,
            cov_matrix_top15,
            avg_rf_rate,
            top15_plot_path
        )
        
        # Save results
        top15_results_df = pd.DataFrame({
            'portfolio': ['Minimum-Variance', 'Tangency'],
            'expected_return': [min_var_return_top15, tangency_return_top15],
            'volatility': [min_var_vol_top15, tangency_vol_top15],
            'sharpe_ratio': [
                (min_var_return_top15 - avg_rf_rate) / min_var_vol_top15 if min_var_vol_top15 > 0 else 0,
                (tangency_return_top15 - avg_rf_rate) / tangency_vol_top15 if tangency_vol_top15 > 0 else 0
            ]
        })
        top15_results_file = os.path.join(RESULTS_REPORTS_DIR, "portfolio_optimization_top15_short_selling.csv")
        top15_results_df.to_csv(top15_results_file, index=False)
        logger.info(f" Saved top 15 stocks results: {top15_results_file}")
        
        # Save frontier data
        top15_frontier_file = os.path.join(RESULTS_REPORTS_DIR, "efficient_frontier_top15_short_selling.csv")
        frontier_df_top15.to_csv(top15_frontier_file, index=False)
        logger.info(f" Saved top 15 stocks frontier data: {top15_frontier_file}")
        
        logger.info(" Completed efficient frontier calculation for top 15 stocks with short-selling")
    else:
        logger.warning("Could not select top 15 stocks. Skipping top 15 efficient frontier calculation.")
    
    return {
        'min_var_portfolio_constrained': {
            'weights': min_var_weights_constrained,
            'return': min_var_return_constrained,
            'volatility': min_var_vol_constrained
        },
        'min_var_portfolio_unconstrained': {
            'weights': min_var_weights_unconstrained,
            'return': min_var_return_unconstrained,
            'volatility': min_var_vol_unconstrained
        },
        'tangency_portfolio_constrained': {
            'weights': tangency_weights_constrained,
            'return': tangency_return_constrained,
            'volatility': tangency_vol_constrained
        },
        'tangency_portfolio_unconstrained': {
            'weights': tangency_weights_unconstrained,
            'return': tangency_return_unconstrained,
            'volatility': tangency_vol_unconstrained
        },
        'equal_weighted_portfolio': {
            'weights': equal_weights,
            'return': equal_weighted_return,
            'volatility': equal_weighted_vol
        },
        'efficient_frontier_constrained': frontier_df_constrained,
        'efficient_frontier_unconstrained': frontier_df_unconstrained,
        'min_var_portfolio_realistic_short': {
            'weights': min_var_weights_realistic,
            'return': min_var_return_realistic,
            'volatility': min_var_vol_realistic,
            'error_flags': min_var_errors
        },
        'tangency_portfolio_realistic_short': {
            'weights': tangency_weights_realistic,
            'return': tangency_return_realistic,
            'volatility': tangency_vol_realistic,
            'error_flags': tangency_errors
        },
        'efficient_frontier_realistic_short': frontier_df_realistic,
        'error_flags': all_error_flags,
        'diversification_benefits': div_benefits
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    results = run_portfolio_optimization()
    
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION COMPLETE")
    print("="*70)

