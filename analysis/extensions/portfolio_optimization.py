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
import seaborn as sns

from analysis.utils.config import (
    DATA_PROCESSED_DIR,
    RESULTS_DATA_DIR,
    RESULTS_FIGURES_DIR,
    RESULTS_REPORTS_DIR,
    ANALYSIS_SETTINGS
)

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Maximum gross exposure for short-selling portfolios
# 2.0 = 200% total exposure (e.g., 150% long + 50% short)
# This prevents infinite leverage in unconstrained portfolios
MAX_GROSS_EXPOSURE = 2.0


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
    Plot efficient frontier with all individual stocks and market index.
    
    This graph shows:
    - X-axis: Volatility (standard deviation)
    - Y-axis: Expected return
    - All individual stocks as scatter points
    - Constrained efficient frontier (long-only, solid line)
    - Unconstrained efficient frontier (with short selling, dashed line)
    - Market index (MSCI Europe) point
    - Interpretation: If efficient frontier overlaps well with market index, CAPM holds.
                     If not, CAPM does not hold (opposite).
    
    Parameters
    ----------
    frontier_df_constrained : pd.DataFrame
        Constrained efficient frontier data (long-only) with 'volatility' and 'return' columns
    frontier_df_unconstrained : pd.DataFrame
        Unconstrained efficient frontier data (with short selling) with 'volatility' and 'return' columns
    min_var_port_constrained : tuple
        (return, volatility) for constrained minimum-variance portfolio (long-only)
    min_var_port_unconstrained : tuple
        (return, volatility) for unconstrained minimum-variance portfolio (with short selling)
    tangency_port_constrained : tuple
        (return, volatility) for constrained tangency portfolio (long-only)
    tangency_port_unconstrained : tuple
        (return, volatility) for unconstrained tangency portfolio (with short selling)
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
        Risk-free rate
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
    logger.info(f"  Unit check - Unconstrained frontier volatility: min={frontier_df_unconstrained['volatility'].min():.2f}%, max={frontier_df_unconstrained['volatility'].max():.2f}%")
    logger.info(f"  Unit check - Market index: return={market_index_return:.2f}%, vol={market_index_vol:.2f}%")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot ALL individual stocks
    ax.scatter(individual_volatilities, expected_returns.values, 
              alpha=0.4, s=30, color='gray', label=f'Individual Stocks (n={len(expected_returns)})',
              edgecolors='black', linewidths=0.3)
    
    # Plot constrained efficient frontier (long-only, solid line)
    ax.plot(frontier_df_constrained['volatility'], frontier_df_constrained['return'], 
            'b-', linewidth=2.5, label='Efficient Frontier (Long-Only)', zorder=5)
    
    # Plot unconstrained efficient frontier (with short selling, dashed line)
    # Only plot if we have valid points (may be empty if optimizations failed)
    if len(frontier_df_unconstrained) > 0:
        ax.plot(frontier_df_unconstrained['volatility'], frontier_df_unconstrained['return'], 
                'b--', linewidth=2.0, alpha=0.8, label='Efficient Frontier (With Short Selling)', zorder=5)
    else:
        # If no frontier points, just show a note
        logger.warning("  Unconstrained efficient frontier has no points - may be due to gross exposure constraint")
    
    # Plot market index (MSCI Europe)
    ax.scatter(market_index_vol, market_index_return, 
              s=200, marker='*', color='red', edgecolors='darkred', linewidths=2,
              label='Market Index (MSCI Europe)', zorder=6)
    
    # Plot constrained minimum-variance portfolio (long-only, red circle)
    ax.plot(min_var_port_constrained[1], min_var_port_constrained[0], 
            'ro', markersize=12, label='Min-Variance Portfolio (Long-Only)', zorder=5)
    
    # Plot unconstrained minimum-variance portfolio (with short selling, red square)
    ax.plot(min_var_port_unconstrained[1], min_var_port_unconstrained[0], 
            'rs', markersize=12, label='Min-Variance Portfolio (With Short Selling)', zorder=5)
    
    # Plot constrained tangency portfolio (long-only, green circle)
    ax.plot(tangency_port_constrained[1], tangency_port_constrained[0], 
            'go', markersize=12, label='Optimal Risky Portfolio (Tangency, Long-Only)', zorder=5)
    
    # Plot unconstrained tangency portfolio (with short selling, green square)
    ax.plot(tangency_port_unconstrained[1], tangency_port_unconstrained[0], 
            'gs', markersize=12, label='Optimal Risky Portfolio (Tangency, With Short Selling)', zorder=5)
    
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
            frontier_df_unconstrained['volatility'].max() if len(frontier_df_unconstrained) > 0 else 0,
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
    ax.set_title('Mean-Variance Efficient Frontier with All Stocks and Market Index', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved: {output_path}")
    logger.info(f"  Market index distance to efficient frontier: {min_distance:.4f}")
    logger.info(f"  CAPM status: {capm_status}")


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
    avg_rf_rate = panel_df['riskfree_rate'].mean()
    logger.info(f"Average risk-free rate: {avg_rf_rate:.4f}% (monthly)")
    
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

