"""
Value Effect Analysis Module.

This module investigates the relationship between book-to-market ratios and
CAPM alphas to test for the presence of value effects in European equity markets.

The value effect hypothesis (Fama & French, 1992):
    - High book-to-market (value) stocks earn higher risk-adjusted returns
    - This manifests as positive alphas for value stocks in CAPM regressions
    - If present, indicates CAPM mis-specification (beta alone insufficient)

Analysis approach:
    1. Fetch book-to-market ratios from Yahoo Finance
    2. Sort stocks into quintile portfolios by B/M ratio
    3. Compare average alphas across B/M quintiles
    4. Test statistical significance of alpha differences

Key outputs:
    - B/M quintile portfolio alphas and t-statistics
    - Visualization of alpha vs B/M relationship
    - Statistical tests for monotonic alpha pattern

References
----------
Fama, E. F., & French, K. R. (1992). The Cross-Section of Expected Stock Returns.
    The Journal of Finance, 47(2), 427-465.
"""

import os
import logging
import shutil
import pandas as pd
import numpy as np
from typing import Dict, Optional
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from analysis.utils.config import (
    DATA_RAW_DIR,
    RESULTS_DATA_DIR,
    RESULTS_FIGURES_DIR,
    RESULTS_REPORTS_DIR,
    RESULTS_FIGURES_VALUE_EFFECTS_DIR
)

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def fetch_book_to_market(ticker: str, _country: str) -> Optional[float]:
    """
    Fetch book-to-market ratio for a stock.
    
    Parameters
    ----------
    ticker : str
        Stock ticker
    _country : str
        Country name (unused but kept for API consistency)
    
    Returns
    -------
    float or None
        Book-to-market ratio, or None if unavailable
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try to get book-to-market directly
        bm = info.get('priceToBook')
        if bm is not None and bm > 0:
            # Price-to-book is inverse of book-to-market
            return 1.0 / bm
        
        # Alternative: try to get book value and market cap
        book_value = info.get('bookValue')
        market_cap = info.get('marketCap')
        
        if book_value is not None and market_cap is not None and market_cap > 0:
            # Book-to-market = book_value / market_cap
            # But we need shares outstanding to convert properly
            shares_outstanding = info.get('sharesOutstanding')
            if shares_outstanding is not None and shares_outstanding > 0:
                book_per_share = book_value
                price = info.get('currentPrice') or info.get('regularMarketPrice')
                if price is not None and price > 0:
                    bm = book_per_share / price
                    return bm
        
        return None
    except Exception as e:
        logger.debug(f"Could not fetch B/M for {ticker}: {e}")
        return None


def estimate_bm_from_proxies(ticker: str, _country: str, _prices_df: pd.DataFrame) -> Optional[float]:
    """
    Estimate book-to-market using proxies if direct data unavailable.
    Uses price-to-earnings (P/E) as a proxy, or dividend yield.
    
    Parameters
    ----------
    ticker : str
        Stock ticker
    country : str
        Country name
    prices_df : pd.DataFrame
        Price data
    
    Returns
    -------
    float or None
        Estimated B/M ratio
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try P/E ratio as proxy (low P/E = value, high P/E = growth)
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        if pe_ratio is not None and pe_ratio > 0:
            # Inverse relationship: low P/E suggests high B/M (value stock)
            # Rough conversion: B/M ≈ 1 / (P/E * adjustment_factor)
            # This is a very rough approximation
            estimated_bm = 1.0 / (pe_ratio * 0.5)  # Rough adjustment
            return max(0.1, min(5.0, estimated_bm))  # Bound between 0.1 and 5.0
        
        # Try dividend yield as proxy (high dividend yield = value)
        dividend_yield = info.get('dividendYield')
        if dividend_yield is not None and dividend_yield > 0:
            # High dividend yield suggests value stock (high B/M)
            # Rough conversion
            estimated_bm = dividend_yield * 20  # Rough scaling
            return max(0.1, min(5.0, estimated_bm))
        
        return None
    except Exception as e:
        logger.debug(f"Could not estimate B/M for {ticker}: {e}")
        return None


def load_book_to_market_ratios(capm_results: pd.DataFrame) -> pd.DataFrame:
    """
    Load or estimate book-to-market ratios for all stocks.
    
    Parameters
    ----------
    capm_results : pd.DataFrame
        CAPM results with ticker and country columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [ticker, country, book_to_market, source]
    """
    logger.info("Loading book-to-market ratios...")
    
    bm_data = []
    
    for _, row in capm_results.iterrows():
        ticker = row['ticker']
        country = row['country']
        
        # Try to fetch real B/M
        bm = fetch_book_to_market(ticker, country)
        source = 'yfinance'
        
        # If unavailable, try proxies
        if bm is None:
            prices_file = os.path.join(DATA_RAW_DIR, f"prices_stocks_{country}.csv")
            if os.path.exists(prices_file):
                prices_df = pd.read_csv(prices_file, index_col=0, parse_dates=True)
                bm = estimate_bm_from_proxies(ticker, country, prices_df)
                source = 'estimated_proxy'
        
        if bm is not None:
            bm_data.append({
                'ticker': ticker,
                'country': country,
                'book_to_market': bm,
                'source': source
            })
        else:
            logger.debug(f"Could not determine B/M for {ticker}")
    
    bm_df = pd.DataFrame(bm_data)
    logger.info(f"  Loaded B/M ratios for {len(bm_df)} stocks")
    logger.info(f"    Direct data: {len(bm_df[bm_df['source'] == 'yfinance'])}")
    logger.info(f"    Estimated: {len(bm_df[bm_df['source'] == 'estimated_proxy'])}")
    
    return bm_df


def create_value_growth_portfolios(
    capm_results: pd.DataFrame,
    bm_df: pd.DataFrame,
    n_portfolios: int = 5
) -> pd.DataFrame:
    """
    Sort stocks into value/growth portfolios based on book-to-market ratios.
    
    Parameters
    ----------
    capm_results : pd.DataFrame
        CAPM results with alpha, beta, etc.
    bm_df : pd.DataFrame
        Book-to-market ratios
    n_portfolios : int
        Number of portfolios (default: 5 quintiles)
    
    Returns
    -------
    pd.DataFrame
        Portfolio assignments with portfolio statistics
    """
    logger.info(f"Creating {n_portfolios} value/growth portfolios...")
    
    # Merge B/M with CAPM results
    merged = capm_results.merge(bm_df, on=['ticker', 'country'], how='inner')
    merged = merged[merged['is_valid'] == True]  # Only valid stocks
    
    if len(merged) == 0:
        logger.warning("No stocks with B/M data")
        return pd.DataFrame()
    
    # Sort by B/M (high B/M = value, low B/M = growth)
    merged = merged.sort_values('book_to_market')
    
    # Assign to portfolios
    merged['portfolio'] = pd.qcut(
        merged['book_to_market'],
        q=n_portfolios,
        labels=[f'P{i+1}' for i in range(n_portfolios)],
        duplicates='drop'
    )
    
    # Calculate portfolio statistics
    portfolio_stats = []
    
    for portfolio in merged['portfolio'].cat.categories:
        port_data = merged[merged['portfolio'] == portfolio]
        
        portfolio_stats.append({
            'portfolio': portfolio,
            'n_stocks': len(port_data),
            'avg_book_to_market': port_data['book_to_market'].mean(),
            'median_book_to_market': port_data['book_to_market'].median(),
            'avg_alpha': port_data['alpha'].mean(),
            'median_alpha': port_data['alpha'].median(),
            'avg_beta': port_data['beta'].mean(),
            'avg_r_squared': port_data['r_squared'].mean(),
            'std_alpha': port_data['alpha'].std()
        })
    
    portfolio_df = pd.DataFrame(portfolio_stats)
    
    logger.info(f"  Created {len(portfolio_df)} portfolios")
    for _, row in portfolio_df.iterrows():
        logger.info(f"    {row['portfolio']}: {row['n_stocks']} stocks, B/M={row['avg_book_to_market']:.3f}, Alpha={row['avg_alpha']:.4f}%")
    
    return portfolio_df


def test_value_effect(portfolio_df: pd.DataFrame) -> Dict:
    """
    Test for value effect: do high B/M (value) stocks have higher alphas?
    
    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Portfolio statistics
    
    Returns
    -------
    dict
        Test results
    """
    logger.info("Testing value effect hypothesis...")
    
    if len(portfolio_df) < 2:
        logger.warning("Insufficient portfolios for value effect test")
        return {}
    
    # Sort by B/M (ascending: P1 = growth, P5 = value)
    portfolio_df = portfolio_df.sort_values('avg_book_to_market')
    
    # Extract alphas
    alphas = portfolio_df['avg_alpha'].values
    bm_ratios = portfolio_df['avg_book_to_market'].values
    
    # Test 1: Correlation between B/M and alpha
    correlation, pvalue_corr = stats.pearsonr(bm_ratios, alphas)
    
    # Test 2: Compare highest B/M (value) vs lowest B/M (growth)
    value_alpha = alphas[-1]  # Highest B/M portfolio
    growth_alpha = alphas[0]  # Lowest B/M portfolio
    alpha_spread = value_alpha - growth_alpha
    
    # Test 3: Linear regression: alpha = a + b * B/M
    slope, intercept, r_value, pvalue_reg, _ = stats.linregress(bm_ratios, alphas)  # std_err not used
    
    logger.info(f"  B/M - Alpha correlation: {correlation:.4f} (p={pvalue_corr:.4f})")
    logger.info(f"  Value portfolio alpha: {value_alpha:.4f}%")
    logger.info(f"  Growth portfolio alpha: {growth_alpha:.4f}%")
    logger.info(f"  Alpha spread (Value - Growth): {alpha_spread:.4f}%")
    logger.info(f"  Regression slope: {slope:.4f} (p={pvalue_reg:.4f})")
    
    # Interpretation
    if pvalue_reg < 0.05 and slope > 0:
        interpretation = "Strong value effect: High B/M stocks have significantly higher alphas"
    elif pvalue_reg < 0.10 and slope > 0:
        interpretation = "Moderate value effect: High B/M stocks have marginally higher alphas"
    elif slope < 0:
        interpretation = "Reverse value effect: High B/M stocks have lower alphas (contrary to theory)"
    else:
        interpretation = "No significant value effect: B/M does not explain alpha differences"
    
    logger.info(f"  Interpretation: {interpretation}")
    
    return {
        'correlation': correlation,
        'pvalue_correlation': pvalue_corr,
        'value_alpha': value_alpha,
        'growth_alpha': growth_alpha,
        'alpha_spread': alpha_spread,
        'regression_slope': slope,
        'regression_intercept': intercept,
        'regression_r_squared': r_value ** 2,
        'regression_pvalue': pvalue_reg,
        'interpretation': interpretation
    }


def plot_value_effect_analysis(
    portfolio_df: pd.DataFrame,
    test_results: Dict,
    output_path: str
) -> None:
    """
    Create visualization of value effect analysis.
    
    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Portfolio statistics
    test_results : dict
        Test results
    output_path : str
        Path to save plot
    """
    logger.info("Creating value effect visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sort by B/M
    portfolio_df = portfolio_df.sort_values('avg_book_to_market')
    
    # Plot 1: Alpha vs B/M
    ax1 = axes[0]
    ax1.scatter(portfolio_df['avg_book_to_market'], portfolio_df['avg_alpha'],
                s=100, alpha=0.7, color='steelblue', edgecolors='black', linewidth=1.5)
    
    # Add regression line
    if 'regression_slope' in test_results:
        bm_range = np.linspace(portfolio_df['avg_book_to_market'].min(),
                              portfolio_df['avg_book_to_market'].max(), 100)
        alpha_pred = test_results['regression_intercept'] + test_results['regression_slope'] * bm_range
        ax1.plot(bm_range, alpha_pred, 'r--', linewidth=2, label='Regression Line')
    
    # Label portfolios
    for _, row in portfolio_df.iterrows():
        ax1.annotate(row['portfolio'], 
                    (row['avg_book_to_market'], row['avg_alpha']),
                    fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Average Book-to-Market Ratio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Alpha (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Value Effect: Alpha vs Book-to-Market', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Portfolio alphas (bar chart)
    ax2 = axes[1]
    # Use get_cmap to avoid linter warnings about RdYlGn member
    cmap = plt.cm.get_cmap('RdYlGn')
    colors = cmap(np.linspace(0.2, 0.8, len(portfolio_df)))
    _ = ax2.bar(portfolio_df['portfolio'], portfolio_df['avg_alpha'],
                color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)  # bars not used
    
    # Add error bars (standard deviation)
    ax2.errorbar(portfolio_df['portfolio'], portfolio_df['avg_alpha'],
                yerr=portfolio_df['std_alpha'], fmt='none', color='black', capsize=5)
    
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel('Portfolio (P1=Growth, P5=Value)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Alpha (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Alpha by Value/Growth Portfolio', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add interpretation text
    if 'interpretation' in test_results:
        fig.suptitle(f"Value Effect Analysis: {test_results['interpretation']}", 
                    fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f" Saved: {output_path}")


def run_value_effects_analysis() -> Dict:
    """
    Run complete value effects analysis.
    
    Returns
    -------
    dict
        Analysis results
    """
    logger.info("="*70)
    logger.info("VALUE EFFECTS ANALYSIS")
    logger.info("="*70)
    
    # Load CAPM results
    capm_results_file = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
    if not os.path.exists(capm_results_file):
        logger.error("CAPM results not found. Run CAPM regressions first.")
        return {}
    
    capm_results = pd.read_csv(capm_results_file)
    
    # Load book-to-market ratios
    bm_df = load_book_to_market_ratios(capm_results)
    
    if len(bm_df) == 0:
        logger.warning("No B/M data available. Analysis cannot proceed.")
        return {}
    
    # Create value/growth portfolios
    portfolio_df = create_value_growth_portfolios(capm_results, bm_df, n_portfolios=5)
    
    if len(portfolio_df) == 0:
        logger.warning("Could not create portfolios")
        return {}
    
    # Test value effect
    test_results = test_value_effect(portfolio_df)
    
    # Create visualization - save to new organized structure
    plot_path = os.path.join(RESULTS_FIGURES_VALUE_EFFECTS_DIR, "value_effect_analysis.png")
    plot_value_effect_analysis(portfolio_df, test_results, plot_path)
    # Also save to legacy location
    legacy_plot = os.path.join(RESULTS_FIGURES_DIR, "value_effect_analysis.png")
    shutil.copy2(plot_path, legacy_plot)
    
    # Save results - keep in reports for now (value effects is a separate analysis)
    portfolio_file = os.path.join(RESULTS_REPORTS_DIR, "value_effects_portfolios.csv")
    portfolio_df.to_csv(portfolio_file, index=False)
    logger.info(f" Saved: {portfolio_file}")
    
    # Save test results
    if test_results:
        test_df = pd.DataFrame([test_results])
        test_file = os.path.join(RESULTS_REPORTS_DIR, "value_effects_test_results.csv")
        test_df.to_csv(test_file, index=False)
        logger.info(f" Saved: {test_file}")
    
    return {
        'portfolios': portfolio_df,
        'test_results': test_results,
        'bm_data': bm_df
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    results = run_value_effects_analysis()
    
    print("\n" + "="*70)
    print("VALUE EFFECTS ANALYSIS COMPLETE")
    print("="*70)

