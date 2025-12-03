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

from analysis.config import (
    DATA_PROCESSED_DIR,
    RESULTS_DATA_DIR,
    RESULTS_PLOTS_DIR,
    RESULTS_REPORTS_DIR,
    ANALYSIS_SETTINGS
)

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def calculate_expected_returns_and_covariance(panel_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calculate expected returns (mean historical returns) and covariance matrix.
    
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
    
    # Pivot to wide format: stocks as columns, dates as rows
    returns_wide = panel_df.pivot_table(
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
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Bounds: no short selling if not allowed
    if allow_short:
        bounds = [(-1, 1) for _ in range(n)]
    else:
        bounds = [(0, 1) for _ in range(n)]
    
    # Initial guess: equal weights
    x0 = np.ones(n) / n
    
    # Optimize
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        logger.warning("Optimization did not converge, trying alternative method...")
        result = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=constraints)
    
    weights = result.x
    port_return = portfolio_return(weights, expected_returns)
    port_vol = np.sqrt(portfolio_variance(weights, cov_matrix))
    
    logger.info(f"  Minimum-variance portfolio:")
    logger.info(f"    Expected return: {port_return:.4f}%")
    logger.info(f"    Volatility: {port_vol:.4f}%")
    logger.info(f"    Sharpe ratio: {port_return / port_vol:.4f}")
    
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
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Bounds
    if allow_short:
        bounds = [(-1, 1) for _ in range(n)]
    else:
        bounds = [(0, 1) for _ in range(n)]
    
    # Initial guess: equal weights
    x0 = np.ones(n) / n
    
    # Optimize
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        logger.warning("Optimization did not converge, trying alternative method...")
        result = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=constraints)
    
    weights = result.x
    port_return = portfolio_return(weights, expected_returns)
    port_vol = np.sqrt(portfolio_variance(weights, cov_matrix))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
    
    logger.info(f"  Tangency portfolio:")
    logger.info(f"    Expected return: {port_return:.4f}%")
    logger.info(f"    Volatility: {port_vol:.4f}%")
    logger.info(f"    Sharpe ratio: {sharpe:.4f}")
    
    return weights, port_return, port_vol


def calculate_efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
    allow_short: bool = False
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
    
    # Find min and max returns
    min_return = expected_returns.min()
    max_return = expected_returns.max()
    
    # Get minimum-variance portfolio
    _, min_var_return, min_var_vol = find_minimum_variance_portfolio(
        expected_returns, cov_matrix, allow_short
    )
    
    # Target returns between min_var_return and max_return
    target_returns = np.linspace(min_var_return, max_return, n_points)
    
    n = len(expected_returns)
    frontier_returns = []
    frontier_vols = []
    
    for target_return in target_returns:
        # Objective: minimize variance
        def objective(weights):
            return portfolio_variance(weights, cov_matrix)
        
        # Constraints: weights sum to 1, target return
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: portfolio_return(w, expected_returns) - target_return}
        ]
        
        # Bounds
        if allow_short:
            bounds = [(-1, 1) for _ in range(n)]
        else:
            bounds = [(0, 1) for _ in range(n)]
        
        # Initial guess
        x0 = np.ones(n) / n
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            port_return = portfolio_return(weights, expected_returns)
            port_vol = np.sqrt(portfolio_variance(weights, cov_matrix))
            frontier_returns.append(port_return)
            frontier_vols.append(port_vol)
        else:
            # If optimization fails, skip this point
            continue
    
    frontier_df = pd.DataFrame({
        'return': frontier_returns,
        'volatility': frontier_vols
    })
    
    logger.info(f"  Efficient frontier: {len(frontier_df)} points")
    
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
    frontier_df: pd.DataFrame,
    min_var_port: Tuple[float, float],
    tangency_port: Tuple[float, float],
    individual_stocks: pd.Series,
    risk_free_rate: float,
    output_path: str
) -> None:
    """
    Plot efficient frontier with minimum-variance and tangency portfolios.
    
    Parameters
    ----------
    frontier_df : pd.DataFrame
        Efficient frontier data
    min_var_port : tuple
        (return, volatility) for minimum-variance portfolio
    tangency_port : tuple
        (return, volatility) for tangency portfolio
    individual_stocks : pd.Series
        Individual stock (return, volatility) pairs
    risk_free_rate : float
        Risk-free rate
    output_path : str
        Path to save plot
    """
    logger.info("Creating efficient frontier plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot efficient frontier
    ax.plot(frontier_df['volatility'], frontier_df['return'], 
            'b-', linewidth=2, label='Efficient Frontier')
    
    # Plot minimum-variance portfolio
    ax.plot(min_var_port[1], min_var_port[0], 
            'ro', markersize=12, label='Minimum-Variance Portfolio')
    
    # Plot tangency portfolio
    ax.plot(tangency_port[1], tangency_port[0], 
            'go', markersize=12, label='Optimal Risky Portfolio (Tangency)')
    
    # Plot risk-free rate
    ax.axhline(y=risk_free_rate, color='k', linestyle='--', linewidth=1, label=f'Risk-Free Rate ({risk_free_rate:.2f}%)')
    
    # Plot capital market line (from risk-free rate through tangency portfolio)
    if tangency_port[1] > 0:
        cml_slope = (tangency_port[0] - risk_free_rate) / tangency_port[1]
        x_cml = np.linspace(0, frontier_df['volatility'].max() * 1.1, 100)
        y_cml = risk_free_rate + cml_slope * x_cml
        ax.plot(x_cml, y_cml, 'g--', linewidth=1.5, alpha=0.7, label='Capital Market Line')
    
    # Plot individual stocks (sample)
    if len(individual_stocks) > 0:
        # Sample stocks for clarity (if too many)
        if len(individual_stocks) > 50:
            sample_stocks = individual_stocks.sample(50)
        else:
            sample_stocks = individual_stocks
        
        # Calculate volatilities for sample stocks
        # This would require covariance matrix - for now, skip individual stocks
        # ax.scatter(..., alpha=0.3, s=20, label='Individual Stocks', color='gray')
        pass
    
    ax.set_xlabel('Volatility (Standard Deviation, %)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Mean-Variance Efficient Frontier', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved: {output_path}")


def run_portfolio_optimization() -> Dict:
    """
    Run complete portfolio optimization analysis.
    
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
    
    # Find minimum-variance portfolio (short selling allowed)
    min_var_weights, min_var_return, min_var_vol = find_minimum_variance_portfolio(
        expected_returns, cov_matrix, allow_short=True
    )
    
    # Find tangency portfolio
    tangency_weights, tangency_return, tangency_vol = find_tangency_portfolio(
        expected_returns, cov_matrix, avg_rf_rate, allow_short=False
    )
    
    # Calculate efficient frontier
    frontier_df = calculate_efficient_frontier(
        expected_returns, cov_matrix, n_points=50, allow_short=False
    )
    
    # Calculate diversification benefits
    div_benefits = calculate_diversification_benefits(expected_returns, cov_matrix)
    
    # Create plot
    plot_path = os.path.join(RESULTS_PLOTS_DIR, "efficient_frontier.png")
    plot_efficient_frontier(
        frontier_df,
        (min_var_return, min_var_vol),
        (tangency_return, tangency_vol),
        expected_returns,  # Individual stocks (simplified)
        avg_rf_rate,
        plot_path
    )
    
    # Save results
    results_df = pd.DataFrame({
        'portfolio': ['Minimum-Variance', 'Optimal Risky (Tangency)', 'Equal-Weighted'],
        'expected_return': [
            min_var_return,
            tangency_return,
            div_benefits['portfolio_return']
        ],
        'volatility': [
            min_var_vol,
            tangency_vol,
            div_benefits['portfolio_volatility']
        ],
        'sharpe_ratio': [
            min_var_return / min_var_vol if min_var_vol > 0 else 0,
            (tangency_return - avg_rf_rate) / tangency_vol if tangency_vol > 0 else 0,
            div_benefits['portfolio_return'] / div_benefits['portfolio_volatility'] if div_benefits['portfolio_volatility'] > 0 else 0
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
        'min_var_portfolio': {
            'weights': min_var_weights,
            'return': min_var_return,
            'volatility': min_var_vol
        },
        'tangency_portfolio': {
            'weights': tangency_weights,
            'return': tangency_return,
            'volatility': tangency_vol
        },
        'efficient_frontier': frontier_df,
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

