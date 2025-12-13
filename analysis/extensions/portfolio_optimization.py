"""
portfolio_optimization.py

Mean-variance portfolio optimization: efficient frontier, minimum-variance portfolio,
and optimal risky portfolio (tangency portfolio).
This addresses assignment requirement: "Illustrate investment opportunities on a mean-variance diagram."
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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


def calculate_expected_returns_and_covariance(panel_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calculate expected returns (mean historical returns) and covariance matrix.
    
    Only includes valid stocks (stocks with is_valid=True in CAPM results).
    This ensures consistency across all analyses.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Returns panel with columns: [date, country, ticker, stock_return, ...]
    
    Returns
    -------
    tuple
        (expected_returns, covariance_matrix)
    """
    logger.info("Calculating expected returns and covariance matrix...")
    
    # Load CAPM results to filter by valid stocks only
    from analysis.utils.config import RESULTS_DATA_DIR
    capm_results = pd.read_csv(os.path.join(RESULTS_DATA_DIR, "capm_results.csv"))
    valid_tickers = set(capm_results[capm_results['is_valid']]['ticker'].unique())
    
    # Filter panel to only include valid stocks
    panel_df_filtered = panel_df[panel_df['ticker'].isin(valid_tickers)].copy()
    logger.info(f"  Filtered to {len(valid_tickers)} valid stocks (from {panel_df['ticker'].nunique()} total)")
    
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


def portfolio_variance(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """
    Calculate portfolio variance: w' * Σ * w
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    cov_matrix : pd.DataFrame
        Covariance matrix
    
    Returns
    -------
    float
        Portfolio variance
    """
    return np.dot(weights, np.dot(cov_matrix.values, weights))


def portfolio_return(weights: np.ndarray, expected_returns: pd.Series) -> float:
    """
    Calculate portfolio expected return: w' * μ
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    expected_returns : pd.Series
        Expected returns
    
    Returns
    -------
    float
        Portfolio expected return
    """
    return np.dot(weights, expected_returns.values)


def portfolio_return_with_costs(
    weights: np.ndarray,
    expected_returns: pd.Series,
    risk_free_rate: float
) -> float:
    """
    Calculate portfolio return adjusted for transaction costs and borrowing costs.
    
    For short positions:
    - Short return = -expected_return (lose when stock goes up)
    - Borrowing cost = (risk_free_rate + borrowing_cost_spread) × |short_exposure|
    
    Transaction costs reduce returns by: cost_rate × gross_exposure
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (can be negative for short positions)
    expected_returns : pd.Series
        Expected returns
    risk_free_rate : float
        Risk-free rate (monthly, percentage)
    
    Returns
    -------
    float
        Adjusted portfolio return (percentage)
    """
    # Calculate gross exposure
    gross_exposure = np.sum(np.abs(weights))
    
    # Separate long and short positions
    long_weights = np.maximum(weights, 0)
    short_weights = np.minimum(weights, 0)
    
    # Long positions earn expected return
    long_return = np.dot(long_weights, expected_returns.values)
    
    # Short positions: lose expected return (negative of expected return)
    # When you short, you profit when stock goes down, lose when it goes up
    # Expected short return = -expected_return (you lose the expected return)
    short_return = np.dot(short_weights, expected_returns.values)
    
    # Borrowing cost: pay additional cost above risk-free rate for borrowing shares
    short_exposure = np.sum(np.abs(short_weights))
    borrowing_cost = (risk_free_rate + BORROWING_COST_SPREAD) * short_exposure
    
    # Transaction costs: apply to gross exposure (both long and short trades)
    transaction_costs = TRANSACTION_COST_RATE * gross_exposure
    
    # Adjusted return: long return + short return - borrowing cost - transaction costs
    # Note: short_return is already negative (from negative weights × positive returns)
    adjusted_return = long_return + short_return - borrowing_cost - transaction_costs
    
    return adjusted_return


def portfolio_variance_with_costs(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame
) -> float:
    """
    Calculate portfolio variance adjusted for transaction costs and market impact.
    
    Transaction costs add variance due to execution uncertainty.
    Market impact increases variance for large positions.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    cov_matrix : pd.DataFrame
        Covariance matrix
    
    Returns
    -------
    float
        Adjusted portfolio variance
    """
    # Base variance
    base_variance = portfolio_variance(weights, cov_matrix)
    
    # Transaction cost variance: proportional to gross exposure
    gross_exposure = np.sum(np.abs(weights))
    transaction_cost_variance = (TRANSACTION_COST_RATE * gross_exposure) ** 2
    
    # Market impact variance: larger positions have more impact
    # Approximate as proportional to squared position sizes
    position_sizes_sq = weights ** 2
    market_impact_variance = 0.0001 * np.sum(position_sizes_sq)  # Small impact factor
    
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
    
    # Objective: minimize portfolio variance
    def objective(weights):
        return portfolio_variance(weights, cov_matrix)
    
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
    
    # Post-optimization constraint projection for short-selling case
    if allow_short:
        # Iteratively project until constraint is satisfied
        max_iterations = 10
        for iteration in range(max_iterations):
            gross_exposure = np.sum(np.abs(weights))
            if gross_exposure <= MAX_GROSS_EXPOSURE + 1e-8:
                break
            # Project onto constraint boundary
            scale_factor = MAX_GROSS_EXPOSURE / gross_exposure
            weights = weights * scale_factor
            # Renormalize to ensure sum = 1
            weights = weights / np.sum(weights)
            if iteration == 0:
                logger.debug(f"  Post-optimization projection: gross exposure {gross_exposure:.6f} → {np.sum(np.abs(weights)):.6f}")
        
        final_gross_exp = np.sum(np.abs(weights))
        if final_gross_exp > MAX_GROSS_EXPOSURE + 1e-6:
            logger.warning(f"  ⚠️  Constraint violation after projection: gross exposure {final_gross_exp:.6f} > {MAX_GROSS_EXPOSURE}")
    
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
    
    # Objective: minimize negative Sharpe ratio (maximize Sharpe ratio)
    def objective(weights):
        port_return = portfolio_return(weights, expected_returns)
        port_vol = np.sqrt(portfolio_variance(weights, cov_matrix))
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
    
    # Post-optimization constraint projection for short-selling case
    # Hard cap: directly enforce the constraint by scaling
    if allow_short:
        gross_exposure = np.sum(np.abs(weights))
        if gross_exposure > MAX_GROSS_EXPOSURE + 1e-8:
            # Hard cap: scale to exactly satisfy gross exposure constraint
            scale_factor = MAX_GROSS_EXPOSURE / gross_exposure
            weights = weights * scale_factor
            # After scaling, sum will be scale_factor, need to adjust to 1
            current_sum = np.sum(weights)
            if abs(current_sum - 1.0) > 1e-8:
                # Adjust proportionally to maintain relative weights while fixing sum
                # This is a heuristic: distribute the difference based on current weights
                diff = 1.0 - current_sum
                # Add/subtract proportionally (preserving signs)
                weight_magnitudes = np.abs(weights)
                if np.sum(weight_magnitudes) > 1e-10:
                    adjustment = diff * weight_magnitudes / np.sum(weight_magnitudes)
                    weights = weights + np.sign(weights) * adjustment
                else:
                    # Fallback: equal weights
                    weights = np.ones(len(weights)) / len(weights)
            
            # Final hard cap if still violated (shouldn't happen, but safety check)
            final_gross_exp = np.sum(np.abs(weights))
            if final_gross_exp > MAX_GROSS_EXPOSURE + 1e-6:
                # Last resort: scale again and accept slight sum violation
                scale_factor = MAX_GROSS_EXPOSURE / final_gross_exp
                weights = weights * scale_factor
                logger.debug(f"  Post-optimization hard cap (final): gross exposure → {np.sum(np.abs(weights)):.6f}, sum → {np.sum(weights):.6f}")
            else:
                logger.debug(f"  Post-optimization projection: gross exposure {gross_exposure:.6f} → {final_gross_exp:.6f}")
        
        # Final verification
        final_gross_exp = np.sum(np.abs(weights))
        if final_gross_exp > MAX_GROSS_EXPOSURE + 1e-5:
            logger.warning(f"  ⚠️  Constraint violation after projection: gross exposure {final_gross_exp:.6f} > {MAX_GROSS_EXPOSURE}")
    
    port_return = portfolio_return(weights, expected_returns)
    port_vol = np.sqrt(portfolio_variance(weights, cov_matrix))
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
        DataFrame with columns: [return, volatility, sharpe]
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
    frontier_returns = []
    frontier_vols = []
    
    # Get initial weights from minimum-variance portfolio for better starting point
    min_var_weights, _, _ = find_minimum_variance_portfolio(
        expected_returns, cov_matrix, allow_short
    )
    x0 = min_var_weights.copy()
    
    for i, target_return in enumerate(target_returns):
        # Objective: minimize variance
        def objective(weights):
            return portfolio_variance(weights, cov_matrix)
        
        # Constraints: weights sum to 1, target return
        # Use closure to capture target_return properly
        def return_constraint(weights, target=target_return):
            return portfolio_return(weights, expected_returns) - target
        
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
        
        # Bounds
        if allow_short:
            bounds = [(-1, 1) for _ in range(n)]
        else:
            bounds = [(0, 1) for _ in range(n)]
        
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
            
            # Post-optimization constraint projection for short-selling case
            # Hard cap: directly enforce the constraint by scaling
            if allow_short:
                gross_exposure = np.sum(np.abs(weights))
                if gross_exposure > MAX_GROSS_EXPOSURE + 1e-8:
                    # Hard cap: scale to exactly satisfy gross exposure constraint
                    scale_factor = MAX_GROSS_EXPOSURE / gross_exposure
                    weights = weights * scale_factor
                    # After scaling, sum will be scale_factor, need to adjust to 1
                    current_sum = np.sum(weights)
                    if abs(current_sum - 1.0) > 1e-8:
                        # Adjust proportionally to maintain relative weights while fixing sum
                        diff = 1.0 - current_sum
                        weight_magnitudes = np.abs(weights)
                        if np.sum(weight_magnitudes) > 1e-10:
                            adjustment = diff * weight_magnitudes / np.sum(weight_magnitudes)
                            weights = weights + np.sign(weights) * adjustment
                        else:
                            # Fallback: equal weights
                            weights = np.ones(len(weights)) / len(weights)
            
            port_return = portfolio_return(weights, expected_returns)
            port_vol = np.sqrt(portfolio_variance(weights, cov_matrix))
            
            # Verify constraints are satisfied
            if allow_short:
                final_gross_exp = np.sum(np.abs(weights))
                if final_gross_exp > MAX_GROSS_EXPOSURE + 1e-6:
                    # Last resort: scale again and accept slight sum violation
                    scale_factor = MAX_GROSS_EXPOSURE / final_gross_exp
                    weights = weights * scale_factor
                    # If still violated after hard cap, skip this point
                    final_gross_exp = np.sum(np.abs(weights))
                    if final_gross_exp > MAX_GROSS_EXPOSURE + 1e-6:
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
    
    # Filter out NaN values before creating DataFrame
    valid_indices = ~(np.isnan(frontier_returns) | np.isnan(frontier_vols))
    frontier_returns_clean = [r for i, r in enumerate(frontier_returns) if valid_indices[i]]
    frontier_vols_clean = [v for i, v in enumerate(frontier_vols) if valid_indices[i]]
    
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
    
    # Objective: minimize adjusted portfolio variance
    def objective(weights):
        return portfolio_variance_with_costs(weights, cov_matrix)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        {
            'type': 'ineq',
            'fun': lambda w: MAX_REALISTIC_GROSS_EXPOSURE - np.sum(np.abs(w))  # Gross exposure limit
        }
    ]
    
    # Position limits: no single short position > MAX_SHORT_POSITION_PCT
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, idx=i: MAX_SHORT_POSITION_PCT + w[idx]  # w[i] >= -MAX_SHORT_POSITION_PCT
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
    port_return = portfolio_return_with_costs(weights, expected_returns, risk_free_rate)
    port_variance = portfolio_variance_with_costs(weights, cov_matrix)
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
            logger.warning(f"  ⚠️  {explanation}")
    
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
    
    # Objective: maximize Sharpe ratio (minimize negative Sharpe)
    def objective(weights):
        port_return = portfolio_return_with_costs(weights, expected_returns, risk_free_rate)
        port_variance = portfolio_variance_with_costs(weights, cov_matrix)
        port_vol = np.sqrt(port_variance)
        if port_vol == 0:
            return 1e10
        
        # Calculate effective risk-free rate accounting for borrowing costs on short positions
        # For short positions, the effective risk-free rate is higher due to borrowing costs
        long_exposure = np.sum(np.maximum(weights, 0))
        short_exposure = np.sum(np.abs(np.minimum(weights, 0)))
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
    port_return = portfolio_return_with_costs(weights, expected_returns, risk_free_rate)
    port_variance = portfolio_variance_with_costs(weights, cov_matrix)
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
    
    logger.info(f"  Calculating {len(target_returns)} frontier points...")
    successful_points = 0
    for i, target_return in enumerate(target_returns):
        progress_pct = int((i + 1) / len(target_returns) * 100)
        progress_bar = "█" * (progress_pct // 5) + "░" * (20 - progress_pct // 5)
        logger.info(f"    [{progress_bar}] {progress_pct}% - Point {i+1}/{len(target_returns)}: target return = {target_return:.4f}%")
        # Objective: minimize variance
        def objective(weights):
            return portfolio_variance_with_costs(weights, cov_matrix)
        
        # Constraints
        def return_constraint(weights, target=target_return):
            return portfolio_return_with_costs(weights, expected_returns, risk_free_rate) - target
        
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
            port_return = portfolio_return_with_costs(weights, expected_returns, risk_free_rate)
            port_variance = portfolio_variance_with_costs(weights, cov_matrix)
            port_vol = np.sqrt(port_variance)
            
            # Calculate effective risk-free rate for Sharpe ratio
            long_exposure = np.sum(np.maximum(weights, 0))
            short_exposure = np.sum(np.abs(np.minimum(weights, 0)))
            total_exposure = long_exposure + short_exposure
            
            if total_exposure > 0:
                effective_rf = (long_exposure * risk_free_rate + short_exposure * (risk_free_rate + BORROWING_COST_SPREAD)) / total_exposure
            else:
                effective_rf = risk_free_rate
            
            sharpe = (port_return - effective_rf) / port_vol if port_vol > 0 else 0
            
            # Check for errors
            error_flags = {
                'unrealistic_volatility': port_vol < MIN_VOLATILITY_THRESHOLD,
                'excessive_leverage': np.sum(np.abs(weights)) > MAX_REALISTIC_GROSS_EXPOSURE,
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
        logger.warning(f"  ⚠️  {aggregate_flags['unrealistic_volatility_count']} portfolios with unrealistic volatility")
    if aggregate_flags['optimization_failed_count'] > 0:
        logger.warning(f"  ⚠️  {aggregate_flags['optimization_failed_count']} optimization failures")
    
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
    
    logger.info(f"✅ Saved: {output_path}")
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
    
    logger.info(f"✅ Saved constrained-only plot: {output_path}")
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
        error_text.append(f"⚠️ {error_flags['unrealistic_volatility_count']} portfolios with unrealistic volatility")
    if error_flags.get('excessive_leverage_count', 0) > 0:
        error_text.append(f"⚠️ {error_flags['excessive_leverage_count']} portfolios with excessive leverage")
    if error_flags.get('optimization_failed_count', 0) > 0:
        error_text.append(f"⚠️ {error_flags['optimization_failed_count']} optimization failures")
    
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
    
    logger.info(f"✅ Saved realistic short-selling plot: {output_path}")


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
        logger.warning(f"⚠️  Unconstrained minimum-variance portfolio has unrealistically low volatility ({min_var_vol_unconstrained:.6f}%)")
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
    logger.info(f"✅ Saved: {results_file}")
    
    # Save diversification metrics
    div_df = pd.DataFrame([div_benefits])
    div_file = os.path.join(RESULTS_REPORTS_DIR, "diversification_benefits.csv")
    div_df.to_csv(div_file, index=False)
    logger.info(f"✅ Saved: {div_file}")
    
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
    logger.info(f"✅ Saved constrained-only results: {constrained_results_file}")
    
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
    logger.info(f"✅ Saved realistic short-selling results: {realistic_results_file}")
    
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

