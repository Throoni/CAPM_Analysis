"""
capm_regression.py

Stage 4: CAPM Regression Estimation

Run CAPM regressions for all stocks to estimate beta, alpha, R², and statistical significance.
Generate country-level summaries and cross-country comparisons.

Regression Specification:
- Dependent variable: E_i,t = R_i,t - R_f (stock excess return)
- Independent variable: E_m,t = R_m,t - R_f (market excess return)
- Model: E_i,t = α_i + β_i * E_m,t + ε_i,t
- Observations: 59 months per stock

Note: Currently uses placeholder risk-free rates. Will work seamlessly with actual
country-specific rates when provided (just update panel and re-run).
"""

import os
import logging
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.utils.config import (
    DATA_PROCESSED_DIR,
    RESULTS_DATA_DIR,
    RESULTS_REPORTS_DIR,
    RESULTS_FIGURES_DIR,
    COUNTRIES,
)

logger = logging.getLogger(__name__)

# Configure matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# -------------------------------------------------------------------
# SINGLE STOCK REGRESSION
# -------------------------------------------------------------------

def run_capm_regression(
    stock_data: pd.DataFrame,
    stock_excess_col: str = 'stock_excess_return',
    market_excess_col: str = 'market_excess_return'
) -> Dict:
    """
    Run CAPM regression for a single stock.
    
    Parameters
    ----------
    stock_data : pd.DataFrame
        DataFrame with columns [stock_excess_return, market_excess_return]
        Should have exactly 59 observations (monthly returns)
    stock_excess_col : str
        Column name for stock excess returns
    market_excess_col : str
        Column name for market excess returns
    
    Returns
    -------
    dict
        Dictionary with regression results:
        {
            'beta': float,
            'alpha': float,
            'beta_tstat': float,
            'alpha_tstat': float,
            'r_squared': float,
            'beta_se': float,
            'alpha_se': float,
            'pvalue_beta': float,
            'pvalue_alpha': float,
            'n_obs': int
        }
    """
    # Drop rows with missing values
    clean_data = stock_data[[stock_excess_col, market_excess_col]].dropna()
    
    # Pre-regression data quality checks
    if len(clean_data) > 0:
        # Check for extreme returns that might indicate data errors
        extreme_stock_returns = clean_data[stock_excess_col].abs() > 100  # >100% monthly return
        extreme_market_returns = clean_data[market_excess_col].abs() > 50  # >50% monthly return
        
        if extreme_stock_returns.any():
            n_extreme = extreme_stock_returns.sum()
            logger.warning(f"Found {n_extreme} extreme stock returns (>100%) - possible data error")
            # Optionally filter out extreme returns
            # clean_data = clean_data[~extreme_stock_returns]
        
        if extreme_market_returns.any():
            n_extreme = extreme_market_returns.sum()
            logger.warning(f"Found {n_extreme} extreme market returns (>50%) - verify market data")
    
    if len(clean_data) < 10:  # Need minimum observations for regression
        logger.warning(f"Insufficient data for regression: {len(clean_data)} observations")
        return {
            'beta': np.nan,
            'alpha': np.nan,
            'beta_tstat': np.nan,
            'alpha_tstat': np.nan,
            'r_squared': np.nan,
            'beta_se': np.nan,
            'alpha_se': np.nan,
            'pvalue_beta': np.nan,
            'pvalue_alpha': np.nan,
            'n_obs': len(clean_data)
        }
    
    # Prepare data for regression
    y = clean_data[stock_excess_col].values
    X = clean_data[market_excess_col].values
    
    # Add constant term for intercept (alpha)
    X_with_const = sm.add_constant(X)
    
    # Run OLS regression
    try:
        model = sm.OLS(y, X_with_const)
        results = model.fit()
        
        # Extract coefficients
        alpha = results.params[0]  # Intercept
        beta = results.params[1]   # Slope
        
        # Extract standard errors
        alpha_se = results.bse[0]
        beta_se = results.bse[1]
        
        # Extract t-statistics
        alpha_tstat = results.tvalues[0]
        beta_tstat = results.tvalues[1]
        
        # Extract p-values
        alpha_pvalue = results.pvalues[0]
        beta_pvalue = results.pvalues[1]
        
        # Extract R-squared
        r_squared = results.rsquared
        
        return {
            'beta': beta,
            'alpha': alpha,
            'beta_tstat': beta_tstat,
            'alpha_tstat': alpha_tstat,
            'r_squared': r_squared,
            'beta_se': beta_se,
            'alpha_se': alpha_se,
            'pvalue_beta': beta_pvalue,
            'pvalue_alpha': alpha_pvalue,
            'n_obs': len(clean_data)
        }
        
    except Exception as e:
        logger.error(f"Regression failed: {e}")
        return {
            'beta': np.nan,
            'alpha': np.nan,
            'beta_tstat': np.nan,
            'alpha_tstat': np.nan,
            'r_squared': np.nan,
            'beta_se': np.nan,
            'alpha_se': np.nan,
            'pvalue_beta': np.nan,
            'pvalue_alpha': np.nan,
            'n_obs': len(clean_data)
        }


# -------------------------------------------------------------------
# BATCH REGRESSION PROCESSING
# -------------------------------------------------------------------

def run_all_capm_regressions(
    panel_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Run CAPM regressions for all stocks in the returns panel.
    
    Parameters
    ----------
    panel_path : str, optional
        Path to returns panel CSV. If None, uses default:
        data/processed/returns_panel.csv
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per stock containing all regression results.
        Columns: country, ticker, beta, alpha, beta_tstat, alpha_tstat,
        r_squared, beta_se, alpha_se, pvalue_beta, pvalue_alpha, n_observations
    """
    if panel_path is None:
        panel_path = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
    
    logger.info("="*70)
    logger.info("RUNNING CAPM REGRESSIONS FOR ALL STOCKS")
    logger.info("="*70)
    logger.info(f"Loading returns panel from: {panel_path}")
    
    # Load returns panel
    panel = pd.read_csv(panel_path, parse_dates=['date'])
    logger.info(f"✅ Loaded panel: {len(panel)} rows")
    logger.info(f"   Countries: {panel['country'].nunique()}")
    logger.info(f"   Stocks: {panel['ticker'].nunique()}")
    logger.info(f"   Date range: {panel['date'].min()} to {panel['date'].max()}")
    
    # Group by country and ticker
    results_list = []
    
    for (country, ticker), group in panel.groupby(['country', 'ticker']):
        logger.debug(f"Running regression for {country} - {ticker} ({len(group)} observations)")
        
        # Run regression
        regression_result = run_capm_regression(group)
        
        # Add stock identifiers
        regression_result['country'] = country
        regression_result['ticker'] = ticker
        
        results_list.append(regression_result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Rename n_obs to n_observations for consistency
    results_df.rename(columns={'n_obs': 'n_observations'}, inplace=True)
    
    # Reorder columns
    column_order = [
        'country', 'ticker',
        'beta', 'alpha',
        'beta_tstat', 'alpha_tstat',
        'r_squared',
        'beta_se', 'alpha_se',
        'pvalue_beta', 'pvalue_alpha',
        'n_observations'
    ]
    results_df = results_df[column_order]
    
    logger.info(f"\n✅ Completed regressions for {len(results_df)} stocks")
    logger.info(f"   Average R²: {results_df['r_squared'].mean():.4f}")
    logger.info(f"   Average beta: {results_df['beta'].mean():.4f}")
    
    return results_df


# -------------------------------------------------------------------
# VALIDATION
# -------------------------------------------------------------------

def validate_regression_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate regression results and flag quality issues.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with regression results
    
    Returns
    -------
    pd.DataFrame
        Results DataFrame with added 'is_valid' column
    """
    logger.info("\n" + "="*70)
    logger.info("VALIDATING REGRESSION RESULTS")
    logger.info("="*70)
    
    # Initialize validation flags
    results_df = results_df.copy()
    results_df['is_valid'] = True
    
    # Check 1: Insufficient observations
    insufficient_obs = results_df['n_observations'] < 59
    if insufficient_obs.any():
        logger.warning(f"⚠️  {insufficient_obs.sum()} stocks have < 59 observations")
        results_df.loc[insufficient_obs, 'is_valid'] = False
    
    # Check 2: Very low R² (low market dependence or data errors)
    # Lower threshold to catch potential data errors
    very_low_r2 = results_df['r_squared'] < 0.01
    low_r2 = results_df['r_squared'] < 0.05
    if very_low_r2.any():
        logger.warning(f"⚠️  {very_low_r2.sum()} stocks have R² < 0.01 (likely data errors)")
        results_df.loc[very_low_r2, 'is_valid'] = False
    if low_r2.any():
        logger.warning(f"⚠️  {low_r2.sum()} stocks have R² < 0.05 (very low market dependence)")
        # Don't automatically invalidate, but flag for review
    
    # Check 3: Extreme beta values (possible errors)
    # Lower threshold to catch more issues earlier
    extreme_beta = (results_df['beta'].abs() > 3.0) | (results_df['beta'] < -1.0)
    if extreme_beta.any():
        logger.warning(f"⚠️  {extreme_beta.sum()} stocks have extreme beta values (|beta| > 3.0 or < -1.0)")
        results_df.loc[extreme_beta, 'is_valid'] = False
    
    # Check 4: Statistically insignificant beta (p-value > 0.05)
    insignificant_beta = results_df['pvalue_beta'] > 0.05
    if insignificant_beta.any():
        logger.info(f"ℹ️  {insignificant_beta.sum()} stocks have statistically insignificant beta (p > 0.05)")
    
    # Check 5: Missing/invalid results
    missing_results = results_df[['beta', 'alpha', 'r_squared']].isna().any(axis=1)
    if missing_results.any():
        logger.warning(f"⚠️  {missing_results.sum()} stocks have missing regression results")
        results_df.loc[missing_results, 'is_valid'] = False
    
    # Summary
    valid_count = results_df['is_valid'].sum()
    total_count = len(results_df)
    logger.info(f"\n✅ Validation complete:")
    logger.info(f"   Valid stocks: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    logger.info(f"   Invalid stocks: {total_count - valid_count}")
    
    return results_df


# -------------------------------------------------------------------
# COUNTRY-LEVEL SUMMARIES
# -------------------------------------------------------------------

def create_country_summaries(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create country-level summary statistics.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with regression results
    
    Returns
    -------
    pd.DataFrame
        Country-level summary statistics
    """
    logger.info("\n" + "="*70)
    logger.info("CREATING COUNTRY-LEVEL SUMMARIES")
    logger.info("="*70)
    
    summaries = []
    
    for country in results_df['country'].unique():
        country_data = results_df[results_df['country'] == country]
        
        # Basic counts
        n_stocks = len(country_data)
        
        # Beta statistics
        mean_beta = country_data['beta'].mean()
        median_beta = country_data['beta'].median()
        std_beta = country_data['beta'].std()
        
        # Beta distribution percentages
        pct_beta_gt_1 = (country_data['beta'] > 1).sum() / n_stocks * 100
        pct_beta_lt_1 = (country_data['beta'] < 1).sum() / n_stocks * 100
        pct_beta_lt_0 = (country_data['beta'] < 0).sum() / n_stocks * 100
        
        # Alpha statistics
        mean_alpha = country_data['alpha'].mean()
        median_alpha = country_data['alpha'].median()
        
        # R² statistics
        mean_r2 = country_data['r_squared'].mean()
        median_r2 = country_data['r_squared'].median()
        
        summaries.append({
            'country': country,
            'n_stocks': n_stocks,
            'mean_beta': mean_beta,
            'median_beta': median_beta,
            'std_beta': std_beta,
            'pct_beta_gt_1': pct_beta_gt_1,
            'pct_beta_lt_1': pct_beta_lt_1,
            'pct_beta_lt_0': pct_beta_lt_0,
            'mean_alpha': mean_alpha,
            'median_alpha': median_alpha,
            'mean_r_squared': mean_r2,
            'median_r_squared': median_r2
        })
    
    summary_df = pd.DataFrame(summaries)
    
    logger.info(f"✅ Created summaries for {len(summary_df)} countries")
    
    return summary_df


def get_top_bottom_stocks(results_df: pd.DataFrame, n: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Get top and bottom stocks by beta and alpha.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with regression results
    n : int
        Number of top/bottom stocks to return
    
    Returns
    -------
    dict
        Dictionary with keys: 'top_beta', 'bottom_beta', 'top_alpha', 'bottom_alpha'
    """
    # Filter valid results
    valid_results = results_df[results_df['is_valid'] == True].copy()
    
    # Top/bottom by beta
    top_beta = valid_results.nlargest(n, 'beta')[['country', 'ticker', 'beta', 'r_squared', 'pvalue_beta']]
    bottom_beta = valid_results.nsmallest(n, 'beta')[['country', 'ticker', 'beta', 'r_squared', 'pvalue_beta']]
    
    # Top/bottom by alpha
    top_alpha = valid_results.nlargest(n, 'alpha')[['country', 'ticker', 'alpha', 'r_squared', 'pvalue_alpha']]
    bottom_alpha = valid_results.nsmallest(n, 'alpha')[['country', 'ticker', 'alpha', 'r_squared', 'pvalue_alpha']]
    
    return {
        'top_beta': top_beta,
        'bottom_beta': bottom_beta,
        'top_alpha': top_alpha,
        'bottom_alpha': bottom_alpha
    }


# -------------------------------------------------------------------
# REPORT GENERATION
# -------------------------------------------------------------------

def generate_capm_reports(
    results_df: pd.DataFrame,
    country_summaries: pd.DataFrame,
    extremes: Dict[str, pd.DataFrame]
) -> None:
    """
    Save all CAPM reports to CSV files.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Full regression results
    country_summaries : pd.DataFrame
        Country-level summaries
    extremes : dict
        Dictionary with top/bottom stocks
    """
    logger.info("\n" + "="*70)
    logger.info("GENERATING CAPM REPORTS")
    logger.info("="*70)
    
    # Save full results
    results_path = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"✅ Saved full results: {results_path}")
    
    # Save country summaries
    summary_path = os.path.join(RESULTS_REPORTS_DIR, "capm_by_country.csv")
    country_summaries.to_csv(summary_path, index=False)
    logger.info(f"✅ Saved country summaries: {summary_path}")
    
    # Save extremes
    extremes_path = os.path.join(RESULTS_REPORTS_DIR, "capm_extremes.csv")
    
    # Combine extremes into single DataFrame
    extremes_list = []
    for category, df in extremes.items():
        df_copy = df.copy()
        df_copy['category'] = category
        extremes_list.append(df_copy)
    
    extremes_combined = pd.concat(extremes_list, ignore_index=True)
    extremes_combined.to_csv(extremes_path, index=False)
    logger.info(f"✅ Saved extremes: {extremes_path}")
    
    logger.info("\n✅ All reports generated successfully")


# -------------------------------------------------------------------
# VISUALIZATIONS
# -------------------------------------------------------------------

def create_capm_visualizations(results_df: pd.DataFrame) -> None:
    """
    Create all CAPM visualizations and save to results/plots/.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with regression results
    """
    logger.info("\n" + "="*70)
    logger.info("CREATING CAPM VISUALIZATIONS")
    logger.info("="*70)
    
    # Filter valid results
    valid_results = results_df[results_df['is_valid'] == True].copy()
    
    # 1. Beta distribution by country (histogram)
    logger.info("Creating beta distribution histograms...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    countries = sorted(valid_results['country'].unique())
    for idx, country in enumerate(countries):
        if idx < len(axes):
            country_data = valid_results[valid_results['country'] == country]
            axes[idx].hist(country_data['beta'], bins=20, edgecolor='black', alpha=0.7)
            axes[idx].axvline(country_data['beta'].mean(), color='red', linestyle='--', label=f"Mean: {country_data['beta'].mean():.2f}")
            axes[idx].set_title(f"{country}\n(n={len(country_data)})")
            axes[idx].set_xlabel('Beta')
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(countries), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    beta_hist_path = os.path.join(RESULTS_FIGURES_DIR, "beta_distribution_by_country.png")
    plt.savefig(beta_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {beta_hist_path}")
    
    # 2. Alpha distribution by country (histogram)
    logger.info("Creating alpha distribution histograms...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, country in enumerate(countries):
        if idx < len(axes):
            country_data = valid_results[valid_results['country'] == country]
            axes[idx].hist(country_data['alpha'], bins=20, edgecolor='black', alpha=0.7)
            axes[idx].axvline(country_data['alpha'].mean(), color='red', linestyle='--', label=f"Mean: {country_data['alpha'].mean():.2f}")
            axes[idx].set_title(f"{country}\n(n={len(country_data)})")
            axes[idx].set_xlabel('Alpha (%)')
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    for idx in range(len(countries), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    alpha_hist_path = os.path.join(RESULTS_FIGURES_DIR, "alpha_distribution_by_country.png")
    plt.savefig(alpha_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {alpha_hist_path}")
    
    # 3. Beta vs R² scatterplot (colored by country)
    logger.info("Creating beta vs R² scatterplot...")
    plt.figure(figsize=(14, 8))
    for country in countries:
        country_data = valid_results[valid_results['country'] == country]
        plt.scatter(country_data['beta'], country_data['r_squared'], 
                   label=country, alpha=0.6, s=50)
    plt.xlabel('Beta', fontsize=12)
    plt.ylabel('R²', fontsize=12)
    plt.title('Beta vs R² by Country', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = os.path.join(RESULTS_FIGURES_DIR, "beta_vs_r2_scatter.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {scatter_path}")
    
    # 4. Cross-country beta comparison (box plot)
    logger.info("Creating cross-country beta comparison...")
    plt.figure(figsize=(14, 8))
    valid_results.boxplot(column='beta', by='country', ax=plt.gca())
    plt.title('Beta Distribution by Country', fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remove default title
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Beta', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    boxplot_path = os.path.join(RESULTS_FIGURES_DIR, "beta_boxplot_by_country.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {boxplot_path}")
    
    # 5. Top 10 highest/lowest beta stocks
    logger.info("Creating top/bottom beta stocks chart...")
    extremes = get_top_bottom_stocks(valid_results, n=10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top 10 beta
    top_beta = extremes['top_beta']
    ax1.barh(range(len(top_beta)), top_beta['beta'], color='green', alpha=0.7)
    ax1.set_yticks(range(len(top_beta)))
    ax1.set_yticklabels([f"{row['ticker']} ({row['country']})" for _, row in top_beta.iterrows()])
    ax1.set_xlabel('Beta', fontsize=12)
    ax1.set_title('Top 10 Highest Beta Stocks', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Bottom 10 beta
    bottom_beta = extremes['bottom_beta']
    ax2.barh(range(len(bottom_beta)), bottom_beta['beta'], color='red', alpha=0.7)
    ax2.set_yticks(range(len(bottom_beta)))
    ax2.set_yticklabels([f"{row['ticker']} ({row['country']})" for _, row in bottom_beta.iterrows()])
    ax2.set_xlabel('Beta', fontsize=12)
    ax2.set_title('Bottom 10 Lowest Beta Stocks', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    top_bottom_beta_path = os.path.join(RESULTS_FIGURES_DIR, "top_bottom_beta_stocks.png")
    plt.savefig(top_bottom_beta_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {top_bottom_beta_path}")
    
    # 6. Average beta by country (bar chart)
    logger.info("Creating average beta by country...")
    country_avg_beta = valid_results.groupby('country')['beta'].mean().sort_values()
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(country_avg_beta)), country_avg_beta.values, color='steelblue', alpha=0.7)
    plt.xticks(range(len(country_avg_beta)), country_avg_beta.index, rotation=45, ha='right')
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Average Beta', fontsize=12)
    plt.title('Average Beta by Country', fontsize=14, fontweight='bold')
    plt.axhline(y=1.0, color='red', linestyle='--', label='Beta = 1.0')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    avg_beta_path = os.path.join(RESULTS_FIGURES_DIR, "average_beta_by_country.png")
    plt.savefig(avg_beta_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {avg_beta_path}")
    
    logger.info("\n✅ All visualizations created successfully")


# -------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------

def run_full_capm_analysis(panel_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete CAPM analysis pipeline.
    
    Parameters
    ----------
    panel_path : str, optional
        Path to returns panel CSV
    
    Returns
    -------
    tuple
        (results_df, country_summaries_df)
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 1. Run all regressions
    results_df = run_all_capm_regressions(panel_path)
    
    # 2. Validate results
    results_df = validate_regression_results(results_df)
    
    # 3. Create country summaries
    country_summaries = create_country_summaries(results_df)
    
    # 4. Get extremes
    extremes = get_top_bottom_stocks(results_df, n=10)
    
    # 5. Generate reports
    generate_capm_reports(results_df, country_summaries, extremes)
    
    # 6. Create visualizations
    create_capm_visualizations(results_df)
    
    # 7. Print summary
    print("\n" + "="*70)
    print("CAPM ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n📊 Summary Statistics:")
    print(f"   Total stocks analyzed: {len(results_df)}")
    print(f"   Valid stocks: {results_df['is_valid'].sum()}")
    print(f"   Average R²: {results_df['r_squared'].mean():.4f}")
    print(f"   Average beta: {results_df['beta'].mean():.4f}")
    print(f"   Average alpha: {results_df['alpha'].mean():.4f}%")
    
    print(f"\n📈 Country-Level Averages:")
    for _, row in country_summaries.iterrows():
        print(f"   {row['country']:15s}: Beta={row['mean_beta']:6.3f}, Alpha={row['mean_alpha']:6.3f}%, R²={row['mean_r_squared']:.3f}")
    
    print(f"\n📁 Output Files:")
    print(f"   Results: {os.path.join(RESULTS_DATA_DIR, 'capm_results.csv')}")
    print(f"   Summaries: {os.path.join(RESULTS_REPORTS_DIR, 'capm_by_country.csv')}")
    print(f"   Extremes: {os.path.join(RESULTS_REPORTS_DIR, 'capm_extremes.csv')}")
    print(f"   Plots: {RESULTS_FIGURES_DIR}/")
    print("="*70)
    
    return results_df, country_summaries


if __name__ == "__main__":
    run_full_capm_analysis()

