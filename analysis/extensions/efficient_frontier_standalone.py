"""
Efficient Frontier Generation Module.

This module provides standalone functionality to compute and visualize mean-variance
efficient frontiers under different portfolio constraints.

Two frontier types are generated:
    1. Constrained (Long-Only): No short-selling allowed (w_i >= 0)
       - Represents realistic investment opportunities for most investors
       - Required for regulated investment funds and pension plans
    
    2. Unconstrained (Short-Selling Allowed): Weights can be negative
       - Represents theoretical optimal portfolios
       - Subject to gross exposure limits to prevent infinite leverage

The efficient frontier represents the set of portfolios that offer:
    - Maximum expected return for a given level of risk, or
    - Minimum risk for a given level of expected return

Key outputs:
    - Efficient frontier curves (both constrained and unconstrained)
    - Minimum-variance portfolio coordinates
    - Tangency portfolio coordinates (maximum Sharpe ratio)
    - Capital Market Line visualization

References
----------
Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional
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
    RESULTS_PORTFOLIO_LONG_ONLY_DIR,
    RESULTS_PORTFOLIO_SHORT_SELLING_DIR,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_GROSS_EXPOSURE = 2.0  # Maximum gross exposure for unconstrained portfolios


def portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """
    Calculate portfolio expected return.
    
    Optimized: accepts numpy array directly to avoid repeated .values conversion.
    """
    return np.dot(weights, expected_returns)


def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculate portfolio variance.
    
    Optimized: uses einsum for better performance and accepts numpy array directly.
    """
    # Use einsum for better performance: w' * Σ * w = Σ_ij w_i * Σ_ij * w_j
    # This is equivalent to np.dot(weights, np.dot(cov_matrix, weights)) but faster
    return np.einsum('i,ij,j->', weights, cov_matrix, weights)


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
        Expected returns for each asset
    cov_matrix : pd.DataFrame
        Covariance matrix
    allow_short : bool
        If True, allow short-selling (negative weights)
        If False, enforce long-only constraint (weights >= 0)
    
    Returns
    -------
    Tuple[np.ndarray, float, float]
        (weights, expected_return, volatility)
    """
    n = len(expected_returns)
    
    # Cache numpy arrays to avoid repeated .values conversion
    cov_matrix_np = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
    expected_returns_np = expected_returns.values if isinstance(expected_returns, pd.Series) else expected_returns
    
    # Objective: minimize variance (use cached numpy array)
    def objective(weights):
        return portfolio_variance(weights, cov_matrix_np)
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    # Bounds: depends on allow_short
    if allow_short:
        # Allow negative weights, but limit gross exposure
        bounds = [(-MAX_GROSS_EXPOSURE, MAX_GROSS_EXPOSURE) for _ in range(n)]
    else:
        # Long-only: weights must be >= 0
        bounds = [(0, 1) for _ in range(n)]
    
    # Initial guess: equal weights
    x0 = np.ones(n) / n
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        logger.warning(f"  Optimization failed: {result.message}")
        # Fallback to equal weights
        if allow_short:
            weights = np.ones(n) / n
        else:
            weights = np.ones(n) / n
    else:
        weights = result.x
    
    # Post-optimization constraint enforcement
    if not allow_short:
        # For long-only: enforce w_i ≥ 0 constraint strictly
        weights = np.maximum(weights, 0)
        weight_sum = np.sum(weights)
        if weight_sum > 1e-10:
            weights = weights / weight_sum
        else:
            weights = np.ones(n) / n
            logger.warning("  All weights too small, using equal weights")
        
        # Verify constraint satisfaction
        if np.any(weights < -1e-8):
            logger.warning("    Negative weights detected, clipping to 0")
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
    else:
        # For short-selling: enforce gross exposure constraint
        # Cache abs() calculation
        abs_weights = np.abs(weights)
        gross_exposure = np.sum(abs_weights)
        if gross_exposure > MAX_GROSS_EXPOSURE + 1e-8:
            scale_factor = MAX_GROSS_EXPOSURE / gross_exposure
            weights = weights * scale_factor
            weights = weights / np.sum(weights)
    
    port_return = portfolio_return(weights, expected_returns_np)
    port_vol = np.sqrt(portfolio_variance(weights, cov_matrix_np))
    
    return weights, port_return, port_vol


def calculate_efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
    allow_short: bool = False,
    min_var_return: Optional[float] = None,
    min_var_vol: Optional[float] = None
) -> pd.DataFrame:
    """
    Calculate the efficient frontier.
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns for each asset
    cov_matrix : pd.DataFrame
        Covariance matrix
    n_points : int
        Number of points on the frontier
    allow_short : bool
        If True, allow short-selling
        If False, enforce long-only constraint
    min_var_return : float, optional
        Minimum-variance portfolio return (if already calculated)
    min_var_vol : float, optional
        Minimum-variance portfolio volatility (if already calculated)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['return', 'volatility']
    """
    # Find minimum-variance portfolio if not provided
    if min_var_return is None or min_var_vol is None:
        min_var_weights, min_var_return, min_var_vol = find_minimum_variance_portfolio(
            expected_returns, cov_matrix, allow_short
        )
    
    # Determine maximum return
    if allow_short:
        # For short-selling, use a conservative estimate
        sorted_returns = expected_returns.sort_values(ascending=False)
        max_return_estimate = min_var_return + (sorted_returns.iloc[0] - min_var_return) * 1.2
        max_return = min(max_return_estimate, expected_returns.max() * 1.1)
    else:
        # For long-only, maximum return is the maximum individual stock return
        max_return = expected_returns.max()
    
    # Ensure max_return is at least min_var_return
    if max_return < min_var_return:
        max_return = min_var_return + 0.1
    
    # Target returns between min_var_return and max_return
    if allow_short:
        n_points_actual = min(n_points, 15)  # Limit points for short-selling
        logger.info(f"  Using {n_points_actual} points for short-selling frontier")
    else:
        n_points_actual = n_points
    
    target_returns = np.linspace(min_var_return, max_return, n_points_actual)
    
    n = len(expected_returns)
    
    # Cache numpy arrays to avoid repeated .values conversion in loops
    cov_matrix_np = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
    expected_returns_np = expected_returns.values if isinstance(expected_returns, pd.Series) else expected_returns
    
    # Pre-compute bounds to avoid recreating in every iteration
    if allow_short:
        bounds = [(-MAX_GROSS_EXPOSURE, MAX_GROSS_EXPOSURE)] * n
    else:
        bounds = [(0, 1)] * n
    
    frontier_returns = []
    frontier_vols = []
    
    # Get initial weights from minimum-variance portfolio
    min_var_weights, _, _ = find_minimum_variance_portfolio(
        expected_returns, cov_matrix, allow_short
    )
    x0 = min_var_weights.copy()  # Copy needed for mutable array
    
    for i, target_return in enumerate(target_returns):
        # Objective: minimize variance (use cached numpy array)
        def objective(weights):
            return portfolio_variance(weights, cov_matrix_np)
        
        # Constraints: weights sum to 1, target return (use cached numpy array)
        def return_constraint(weights, target=target_return):
            return portfolio_return(weights, expected_returns_np) - target
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': return_constraint}
        ]
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if result.success:
                weights = result.x
                
                # Post-optimization constraint enforcement
                if not allow_short:
                    weights = np.maximum(weights, 0)
                    weight_sum = np.sum(weights)
                    if weight_sum > 1e-10:
                        weights = weights / weight_sum
                    else:
                        weights = np.ones(n) / n
                    
                    if np.any(weights < -1e-8):
                        logger.warning(f"    Negative weights at target return {target_return:.4f}%, clipping")
                        weights = np.maximum(weights, 0)
                        weights = weights / np.sum(weights)
                else:
                    # Enforce gross exposure constraint
                    gross_exposure = np.sum(np.abs(weights))
                    if gross_exposure > MAX_GROSS_EXPOSURE + 1e-8:
                        scale_factor = MAX_GROSS_EXPOSURE / gross_exposure
                        weights = weights * scale_factor
                        weights = weights / np.sum(weights)
                
                port_return = portfolio_return(weights, expected_returns_np)
                port_vol = np.sqrt(portfolio_variance(weights, cov_matrix_np))
                
                frontier_returns.append(port_return)
                frontier_vols.append(port_vol)
                # Use current solution as starting point for next optimization
                # In-place update to avoid allocation (weights is already a copy from result.x)
                x0[:] = weights
            else:
                # If optimization fails, skip this point
                frontier_returns.append(np.nan)
                frontier_vols.append(np.nan)
        except Exception as e:
            logger.warning(f"  Optimization failed for target return {target_return:.4f}%: {e}")
            frontier_returns.append(np.nan)
            frontier_vols.append(np.nan)
    
    # Create DataFrame
    frontier_df = pd.DataFrame({
        'return': frontier_returns,
        'volatility': frontier_vols
    })
    
    # Remove NaN values
    frontier_df = frontier_df.dropna()
    
    return frontier_df


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    min_var_port: Tuple[float, float],
    tangency_port: Optional[Tuple[float, float]],
    equal_weighted_port: Tuple[float, float],
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    output_path: str,
    title_suffix: str = ""
) -> None:
    """
    Plot the efficient frontier.
    
    Parameters
    ----------
    frontier_df : pd.DataFrame
        Efficient frontier data with 'return' and 'volatility' columns
    min_var_port : Tuple[float, float]
        (return, volatility) for minimum-variance portfolio
    tangency_port : Tuple[float, float], optional
        (return, volatility) for tangency portfolio
    equal_weighted_port : Tuple[float, float]
        (return, volatility) for equal-weighted portfolio
    expected_returns : pd.Series
        Expected returns for individual assets
    cov_matrix : pd.DataFrame
        Covariance matrix
    risk_free_rate : float
        Risk-free rate (annualized percentage)
    output_path : str
        Path to save the plot
    title_suffix : str
        Suffix to add to the plot title
    """
    # Set style
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    else:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    plt.rcParams['figure.figsize'] = (12, 8)
    
    fig, ax = plt.subplots()
    
    # Plot efficient frontier
    ax.plot(
        frontier_df['volatility'],
        frontier_df['return'],
        'b-',
        linewidth=2,
        label='Efficient Frontier',
        zorder=3
    )
    
    # Plot individual assets
    asset_vols = np.sqrt(np.diag(cov_matrix))
    ax.scatter(
        asset_vols,
        expected_returns,
        c='gray',
        alpha=0.5,
        s=50,
        label='Individual Assets',
        zorder=1
    )
    
    # Plot minimum-variance portfolio
    ax.scatter(
        min_var_port[1],
        min_var_port[0],
        c='red',
        s=200,
        marker='*',
        label='Minimum-Variance Portfolio',
        zorder=5,
        edgecolors='black',
        linewidths=1
    )
    
    # Plot tangency portfolio if provided
    if tangency_port is not None:
        ax.scatter(
            tangency_port[1],
            tangency_port[0],
            c='green',
            s=200,
            marker='*',
            label='Tangency Portfolio',
            zorder=5,
            edgecolors='black',
            linewidths=1
        )
        
        # Plot capital market line
        sharpe = (tangency_port[0] - risk_free_rate) / tangency_port[1] if tangency_port[1] > 0 else 0
        x_cml = np.linspace(0, max(frontier_df['volatility'].max(), tangency_port[1] * 1.2), 100)
        y_cml = risk_free_rate + sharpe * x_cml
        ax.plot(
            x_cml,
            y_cml,
            'g--',
            linewidth=1.5,
            alpha=0.7,
            label=f'Capital Market Line (Sharpe: {sharpe:.3f})',
            zorder=2
        )
    
    # Plot equal-weighted portfolio
    ax.scatter(
        equal_weighted_port[1],
        equal_weighted_port[0],
        c='orange',
        s=150,
        marker='o',
        label='Equal-Weighted Portfolio',
        zorder=4,
        edgecolors='black',
        linewidths=1
    )
    
    # Labels and title
    ax.set_xlabel('Volatility (Standard Deviation, %)', fontsize=12)
    ax.set_ylabel('Expected Return (%)', fontsize=12)
    
    # Determine constraint type from title_suffix
    if "Constrained" in title_suffix or "Long-Only" in title_suffix:
        constraint_text = "No Short-Selling"
    else:
        constraint_text = "Short-Selling Allowed"
    
    ax.set_title(
        f'Mean-Variance Efficient Frontier\n({constraint_text})',
        fontsize=14,
        fontweight='bold'
    )
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add risk-free rate annotation
    ax.text(
        0.02, 0.98,
        f'Risk-Free Rate: {risk_free_rate:.4f}%',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f" Saved efficient frontier plot: {output_path}")


def calculate_expected_returns_and_covariance(panel_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calculate expected returns and covariance matrix from panel data.
    
    Only includes valid stocks (stocks with is_valid=True in CAPM results).
    This ensures consistency across all analyses.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with columns: ticker, date, stock_return
    
    Returns
    -------
    Tuple[pd.Series, pd.DataFrame]
        (expected_returns, cov_matrix)
    """
    logger.info("Calculating expected returns and covariance matrix...")
    
    # Load CAPM results to filter by valid stocks only
    capm_results_file = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
    if not os.path.exists(capm_results_file):
        logger.warning("CAPM results not found. Using all stocks in panel.")
        valid_tickers = set(panel_df['ticker'].unique())
    else:
        capm_results = pd.read_csv(capm_results_file)
        valid_tickers = set(capm_results[capm_results['is_valid']]['ticker'].unique())
    
    # Filter panel to only include valid stocks
    panel_df_filtered = panel_df[panel_df['ticker'].isin(valid_tickers)].copy()
    
    logger.info(f"  Using {panel_df_filtered['ticker'].nunique()} stocks (from {panel_df['ticker'].nunique()} total)")
    
    # Pivot to get returns matrix (tickers as columns, dates as rows)
    # Use 'stock_return' column (already in percentage form)
    returns_matrix = panel_df_filtered.pivot_table(
        index='date',
        columns='ticker',
        values='stock_return',
        aggfunc='mean'
    )
    
    # Calculate expected returns (mean monthly returns, already in percentage)
    expected_returns = returns_matrix.mean()
    
    # Calculate covariance matrix (monthly, in percentage squared)
    # stock_return is already in percentage, so cov is in percentage^2
    cov_matrix = returns_matrix.cov()
    
    logger.info(f"  Expected returns: {len(expected_returns)} stocks")
    logger.info(f"  Covariance matrix: {cov_matrix.shape}")
    logger.info(f"  Mean return range: [{expected_returns.min():.2f}%, {expected_returns.max():.2f}%]")
    logger.info(f"  Mean return: {expected_returns.mean():.2f}%")
    
    return expected_returns, cov_matrix


def find_tangency_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    allow_short: bool = False
) -> Tuple[np.ndarray, float, float]:
    """
    Find the tangency portfolio (optimal risky portfolio).
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns for each asset
    cov_matrix : pd.DataFrame
        Covariance matrix
    risk_free_rate : float
        Risk-free rate (annualized percentage)
    allow_short : bool
        If True, allow short-selling
        If False, enforce long-only constraint
    
    Returns
    -------
    Tuple[np.ndarray, float, float]
        (weights, expected_return, volatility)
    """
    n = len(expected_returns)
    
    # Cache numpy arrays to avoid repeated .values conversion
    cov_matrix_np = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
    expected_returns_np = expected_returns.values if isinstance(expected_returns, pd.Series) else expected_returns
    
    # Objective: maximize Sharpe ratio = (return - rf) / volatility
    def objective(weights):
        port_return = portfolio_return(weights, expected_returns_np)
        port_vol = np.sqrt(portfolio_variance(weights, cov_matrix_np))
        if port_vol < 1e-10:
            return -1e10  # Penalty for zero volatility
        sharpe = (port_return - risk_free_rate) / port_vol
        return -sharpe  # Minimize negative Sharpe (maximize Sharpe)
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    # Pre-compute bounds (more efficient than creating in loop)
    if allow_short:
        bounds = [(-MAX_GROSS_EXPOSURE, MAX_GROSS_EXPOSURE)] * n
    else:
        bounds = [(0, 1)] * n
    
    # Initial guess: equal weights
    x0 = np.ones(n) / n
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        logger.warning(f"  Tangency optimization failed: {result.message}")
        weights = np.ones(n) / n
    else:
        weights = result.x
    
    # Post-optimization constraint enforcement
    if not allow_short:
        weights = np.maximum(weights, 0)
        weight_sum = np.sum(weights)
        if weight_sum > 1e-10:
            weights = weights / weight_sum
        else:
            weights = np.ones(n) / n
    else:
        # Cache abs() calculation
        abs_weights = np.abs(weights)
        gross_exposure = np.sum(abs_weights)
        if gross_exposure > MAX_GROSS_EXPOSURE + 1e-8:
            scale_factor = MAX_GROSS_EXPOSURE / gross_exposure
            weights = weights * scale_factor
            weights = weights / np.sum(weights)
    
    port_return = portfolio_return(weights, expected_returns_np)
    port_vol = np.sqrt(portfolio_variance(weights, cov_matrix_np))
    
    return weights, port_return, port_vol


def calculate_equal_weighted_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame
) -> Tuple[float, float]:
    """
    Calculate equal-weighted portfolio return and volatility.
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns for each asset
    cov_matrix : pd.DataFrame
        Covariance matrix
    
    Returns
    -------
    Tuple[float, float]
        (expected_return, volatility)
    """
    n = len(expected_returns)
    weights = np.ones(n) / n
    
    # Cache numpy arrays to avoid repeated .values conversion
    cov_matrix_np = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
    expected_returns_np = expected_returns.values if isinstance(expected_returns, pd.Series) else expected_returns
    
    port_return = portfolio_return(weights, expected_returns_np)
    port_vol = np.sqrt(portfolio_variance(weights, cov_matrix_np))
    
    return port_return, port_vol


def main():
    """Main function to generate both efficient frontiers."""
    logger.info("="*70)
    logger.info("EFFICIENT FRONTIER GENERATION (STANDALONE)")
    logger.info("="*70)
    
    # Load panel data
    panel_file = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
    if not os.path.exists(panel_file):
        logger.error(f"Panel data not found: {panel_file}")
        logger.error("Please run the returns processing first.")
        sys.exit(1)
    
    logger.info(f"Loading panel data from: {panel_file}")
    panel_df = pd.read_csv(panel_file, parse_dates=['date'])
    logger.info(f"  Loaded {len(panel_df)} observations")
    logger.info(f"  Date range: {panel_df['date'].min()} to {panel_df['date'].max()}")
    logger.info(f"  Number of unique tickers: {panel_df['ticker'].nunique()}")
    
    # Calculate expected returns and covariance
    logger.info("Calculating expected returns and covariance matrix...")
    expected_returns, cov_matrix = calculate_expected_returns_and_covariance(panel_df)
    logger.info(f"  Number of assets: {len(expected_returns)}")
    logger.info(f"  Expected returns range: {expected_returns.min():.4f}% to {expected_returns.max():.4f}%")
    
    # Get risk-free rate
    risk_free_rate_file = os.path.join(RESULTS_DATA_DIR, "riskfree_rate_summary.csv")
    if os.path.exists(risk_free_rate_file):
        rf_summary = pd.read_csv(risk_free_rate_file)
        avg_rf_rate = rf_summary['monthly_rate_pct'].mean()
        logger.info(f"  Average risk-free rate: {avg_rf_rate:.4f}% (monthly)")
    else:
        logger.warning("Risk-free rate summary not found, using 0%")
        avg_rf_rate = 0.0
    
    # ========================================================================
    # 1. CONSTRAINED EFFICIENT FRONTIER (Long-Only, No Short-Selling)
    # ========================================================================
    logger.info("")
    logger.info("="*70)
    logger.info("CALCULATING CONSTRAINED EFFICIENT FRONTIER (Long-Only)")
    logger.info("="*70)
    
    # Find minimum-variance portfolio
    logger.info("Finding minimum-variance portfolio (constrained)...")
    min_var_weights_constrained, min_var_return_constrained, min_var_vol_constrained = \
        find_minimum_variance_portfolio(expected_returns, cov_matrix, allow_short=False)
    logger.info(f"  Return: {min_var_return_constrained:.4f}%, Volatility: {min_var_vol_constrained:.4f}%")
    
    # Find tangency portfolio
    logger.info("Finding tangency portfolio (constrained)...")
    tangency_weights_constrained, tangency_return_constrained, tangency_vol_constrained = \
        find_tangency_portfolio(expected_returns, cov_matrix, avg_rf_rate, allow_short=False)
    logger.info(f"  Return: {tangency_return_constrained:.4f}%, Volatility: {tangency_vol_constrained:.4f}%")
    sharpe_constrained = (tangency_return_constrained - avg_rf_rate) / tangency_vol_constrained if tangency_vol_constrained > 0 else 0
    logger.info(f"  Sharpe ratio: {sharpe_constrained:.4f}")
    
    # Calculate efficient frontier
    logger.info("Calculating efficient frontier (constrained, 50 points)...")
    frontier_df_constrained = calculate_efficient_frontier(
        expected_returns, cov_matrix, n_points=50, allow_short=False,
        min_var_return=min_var_return_constrained, min_var_vol=min_var_vol_constrained
    )
    logger.info(f"  Generated {len(frontier_df_constrained)} frontier points")
    
    # Calculate equal-weighted portfolio
    equal_weighted_return, equal_weighted_vol = calculate_equal_weighted_portfolio(
        expected_returns, cov_matrix
    )
    logger.info(f"  Equal-weighted: Return: {equal_weighted_return:.4f}%, Volatility: {equal_weighted_vol:.4f}%")
    
    # Save results
    results_constrained = pd.DataFrame({
        'portfolio': [
            'Minimum-Variance (Constrained)',
            'Tangency (Constrained)',
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
            sharpe_constrained,
            (equal_weighted_return - avg_rf_rate) / equal_weighted_vol if equal_weighted_vol > 0 else 0
        ]
    })
    
    results_file_constrained = os.path.join(RESULTS_PORTFOLIO_LONG_ONLY_DIR, "portfolio_optimization_results.csv")
    results_constrained.to_csv(results_file_constrained, index=False)
    logger.info(f" Saved constrained results: {results_file_constrained}")
    
    # Save frontier data
    frontier_file_constrained = os.path.join(RESULTS_PORTFOLIO_LONG_ONLY_DIR, "efficient_frontier.csv")
    frontier_df_constrained.to_csv(frontier_file_constrained, index=False)
    logger.info(f" Saved constrained frontier data: {frontier_file_constrained}")
    
    # Plot - Constrained (No Short-Selling)
    plot_path_constrained = os.path.join(RESULTS_FIGURES_DIR, "portfolio", "efficient_frontier_no_shortselling.png")
    os.makedirs(os.path.dirname(plot_path_constrained), exist_ok=True)
    plot_efficient_frontier(
        frontier_df_constrained,
        (min_var_return_constrained, min_var_vol_constrained),
        (tangency_return_constrained, tangency_vol_constrained),
        (equal_weighted_return, equal_weighted_vol),
        expected_returns,
        cov_matrix,
        avg_rf_rate,
        plot_path_constrained,
        title_suffix=" (Long-Only, No Short-Selling)"
    )
    
    # ========================================================================
    # 2. UNCONSTRAINED EFFICIENT FRONTIER (Short-Selling Allowed)
    # ========================================================================
    logger.info("")
    logger.info("="*70)
    logger.info("CALCULATING UNCONSTRAINED EFFICIENT FRONTIER (Short-Selling Allowed)")
    logger.info("="*70)
    
    # Find minimum-variance portfolio
    logger.info("Finding minimum-variance portfolio (unconstrained)...")
    min_var_weights_unconstrained, min_var_return_unconstrained, min_var_vol_unconstrained = \
        find_minimum_variance_portfolio(expected_returns, cov_matrix, allow_short=True)
    logger.info(f"  Return: {min_var_return_unconstrained:.4f}%, Volatility: {min_var_vol_unconstrained:.4f}%")
    
    # Find tangency portfolio
    logger.info("Finding tangency portfolio (unconstrained)...")
    tangency_weights_unconstrained, tangency_return_unconstrained, tangency_vol_unconstrained = \
        find_tangency_portfolio(expected_returns, cov_matrix, avg_rf_rate, allow_short=True)
    logger.info(f"  Return: {tangency_return_unconstrained:.4f}%, Volatility: {tangency_vol_unconstrained:.4f}%")
    sharpe_unconstrained = (tangency_return_unconstrained - avg_rf_rate) / tangency_vol_unconstrained if tangency_vol_unconstrained > 0 else 0
    logger.info(f"  Sharpe ratio: {sharpe_unconstrained:.4f}")
    
    # Calculate efficient frontier
    logger.info("Calculating efficient frontier (unconstrained, 15 points)...")
    frontier_df_unconstrained = calculate_efficient_frontier(
        expected_returns, cov_matrix, n_points=15, allow_short=True,
        min_var_return=min_var_return_unconstrained, min_var_vol=min_var_vol_unconstrained
    )
    logger.info(f"  Generated {len(frontier_df_unconstrained)} frontier points")
    
    # Save results
    results_unconstrained = pd.DataFrame({
        'portfolio': [
            'Minimum-Variance (Unconstrained)',
            'Tangency (Unconstrained)',
            'Equal-Weighted'
        ],
        'expected_return': [
            min_var_return_unconstrained,
            tangency_return_unconstrained,
            equal_weighted_return
        ],
        'volatility': [
            min_var_vol_unconstrained,
            tangency_vol_unconstrained,
            equal_weighted_vol
        ],
        'sharpe_ratio': [
            (min_var_return_unconstrained - avg_rf_rate) / min_var_vol_unconstrained if min_var_vol_unconstrained > 0 else 0,
            sharpe_unconstrained,
            (equal_weighted_return - avg_rf_rate) / equal_weighted_vol if equal_weighted_vol > 0 else 0
        ],
        'gross_exposure': [
            np.sum(np.abs(min_var_weights_unconstrained)),
            np.sum(np.abs(tangency_weights_unconstrained)),
            1.0
        ]
    })
    
    results_file_unconstrained = os.path.join(RESULTS_PORTFOLIO_SHORT_SELLING_DIR, "portfolio_optimization_results.csv")
    results_unconstrained.to_csv(results_file_unconstrained, index=False)
    logger.info(f" Saved unconstrained results: {results_file_unconstrained}")
    
    # Save frontier data
    frontier_file_unconstrained = os.path.join(RESULTS_PORTFOLIO_SHORT_SELLING_DIR, "efficient_frontier.csv")
    frontier_df_unconstrained.to_csv(frontier_file_unconstrained, index=False)
    logger.info(f" Saved unconstrained frontier data: {frontier_file_unconstrained}")
    
    # Plot - Unconstrained (Short-Selling Allowed)
    plot_path_unconstrained = os.path.join(RESULTS_FIGURES_DIR, "portfolio", "efficient_frontier_with_shortselling.png")
    os.makedirs(os.path.dirname(plot_path_unconstrained), exist_ok=True)
    plot_efficient_frontier(
        frontier_df_unconstrained,
        (min_var_return_unconstrained, min_var_vol_unconstrained),
        (tangency_return_unconstrained, tangency_vol_unconstrained),
        (equal_weighted_return, equal_weighted_vol),
        expected_returns,
        cov_matrix,
        avg_rf_rate,
        plot_path_unconstrained,
        title_suffix=" (Short-Selling Allowed)"
    )
    
    logger.info("")
    logger.info("="*70)
    logger.info("EFFICIENT FRONTIER GENERATION COMPLETE")
    logger.info("="*70)
    logger.info(f" No Short-Selling frontier: {plot_path_constrained}")
    logger.info(f" Short-Selling Allowed frontier: {plot_path_unconstrained}")
    
    # Print summary
    print("\n" + "="*70)
    print("EFFICIENT FRONTIER RESULTS")
    print("="*70)
    print("\n NO SHORT-SELLING (Long-Only, Constrained):")
    print(f"   Minimum-Variance: Return={min_var_return_constrained:.4f}%, Vol={min_var_vol_constrained:.4f}%")
    print(f"   Tangency: Return={tangency_return_constrained:.4f}%, Vol={tangency_vol_constrained:.4f}%")
    print(f"   Sharpe Ratio: {sharpe_constrained:.4f}")
    print(f"   Plot: {plot_path_constrained}")
    print("\n SHORT-SELLING ALLOWED (Unconstrained):")
    print(f"   Minimum-Variance: Return={min_var_return_unconstrained:.4f}%, Vol={min_var_vol_unconstrained:.4f}%")
    print(f"   Tangency: Return={tangency_return_unconstrained:.4f}%, Vol={tangency_vol_unconstrained:.4f}%")
    print(f"   Sharpe Ratio: {sharpe_unconstrained:.4f}")
    print(f"   Gross Exposure (Tangency): {np.sum(np.abs(tangency_weights_unconstrained)):.2f}")
    print(f"   Plot: {plot_path_unconstrained}")
    print("="*70)


if __name__ == "__main__":
    main()

