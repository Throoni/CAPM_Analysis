"""
Fama-MacBeth Cross-Sectional Regression Module.

This module implements the Fama-MacBeth (1973) two-pass regression methodology
to test whether beta explains cross-sectional variation in expected returns.

The methodology consists of two passes:

    Pass 1 (Time-Series): For each stock i, estimate beta from:
        R_i,t - R_f = alpha_i + beta_i * (R_m,t - R_f) + epsilon_i,t

    Pass 2 (Cross-Sectional): For each month t, regress returns on betas:
        R_i,t = gamma_0,t + gamma_1,t * beta_i + u_i,t

The key test statistic is the time-series average of gamma_1,t coefficients:
    - If gamma_1 > 0 and significant: Higher beta stocks earn higher returns (CAPM holds)
    - If gamma_0 is significantly different from R_f: Abnormal returns exist

The standard errors are computed using the Fama-MacBeth approach:
    SE(gamma) = std(gamma_t) / sqrt(T)

This accounts for cross-sectional correlation in returns by using the time-series
variation of the monthly coefficients.

References
----------
Fama, E. F., & MacBeth, J. D. (1973). Risk, Return, and Equilibrium: 
    Empirical Tests. Journal of Political Economy, 81(3), 607-636.
"""

import os
import logging
import shutil
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from analysis.utils.config import (
    DATA_PROCESSED_DIR,
    RESULTS_DATA_DIR,
    RESULTS_REPORTS_DIR,
    RESULTS_FIGURES_DIR,
    RESULTS_CAPM_CROSSSECTIONAL_DIR,
    RESULTS_CAPM_CROSSSECTIONAL_FIGURES_DIR,
)

logger = logging.getLogger(__name__)

# Configure matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# -------------------------------------------------------------------
# STEP 1: PREPARE CROSS-SECTIONAL DATA
# -------------------------------------------------------------------

def prepare_cross_sectional_data(
    panel_df: pd.DataFrame,
    beta_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare cross-sectional data: merge betas with average realized returns.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Returns panel with columns: [date, country, ticker, stock_return, ...]
    beta_df : pd.DataFrame
        CAPM results with columns: [country, ticker, beta, ...]
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per stock: [ticker, country, beta, avg_return]
    """
    logger.info("="*70)
    logger.info("STEP 1: PREPARING CROSS-SECTIONAL DATA")
    logger.info("="*70)
    
    # Compute average realized return for each stock
    avg_returns = panel_df.groupby(['country', 'ticker'])['stock_return'].mean().reset_index()
    avg_returns.rename(columns={'stock_return': 'avg_return'}, inplace=True)
    
    logger.info(f"Computed average returns for {len(avg_returns)} stocks")
    
    # Merge with betas
    # Only use valid stocks (is_valid == True) - cache boolean mask
    valid_mask = beta_df['is_valid'] == True
    valid_betas = beta_df[valid_mask][['country', 'ticker', 'beta']].copy()
    
    cross_sectional = avg_returns.merge(
        valid_betas,
        on=['country', 'ticker'],
        how='inner'
    )
    
    logger.info(f"Merged with betas: {len(cross_sectional)} stocks")
    logger.info(f"  Beta range: {cross_sectional['beta'].min():.3f} to {cross_sectional['beta'].max():.3f}")
    logger.info(f"  Avg return range: {cross_sectional['avg_return'].min():.2f}% to {cross_sectional['avg_return'].max():.2f}%")
    
    return cross_sectional


# -------------------------------------------------------------------
# STEP 2: RUN MONTHLY CROSS-SECTIONAL REGRESSIONS
# -------------------------------------------------------------------

def run_monthly_cross_sectional_regressions(
    panel_df: pd.DataFrame,
    beta_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Run monthly cross-sectional regressions: R_{i,t} = γ_{0,t} + γ_{1,t} * β_i + u_{i,t}
    
    NOTE: Returns used in this regression are excess returns calculated using German
    3-month Bund (EUR) as the risk-free rate for all countries. This ensures
    consistency with the time-series CAPM regressions.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Returns panel with monthly returns (excess returns using German Bund)
    beta_df : pd.DataFrame
        CAPM results with betas (estimated using German Bund risk-free rate)
    
    Returns
    -------
    pd.DataFrame
        Monthly regression results with columns:
        [date, gamma_0, gamma_1, gamma_0_se, gamma_1_se, 
         gamma_0_tstat, gamma_1_tstat, n_stocks]
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 2: RUNNING MONTHLY CROSS-SECTIONAL REGRESSIONS")
    logger.info("="*70)
    
    # Get valid betas - cache boolean mask
    valid_mask = beta_df['is_valid'] == True
    valid_betas = beta_df[valid_mask][['country', 'ticker', 'beta']].copy()
    
    # Create beta lookup dictionary (vectorized - more efficient than iterrows)
    beta_dict = dict(zip(
        zip(valid_betas['country'], valid_betas['ticker']),
        valid_betas['beta']
    ))
    
    # Get unique months - cache date conversion (only convert if not already datetime)
    if not pd.api.types.is_datetime64_any_dtype(panel_df['date']):
        panel_df['date'] = pd.to_datetime(panel_df['date'])
    months = sorted(panel_df['date'].unique())
    
    logger.info(f"Running regressions for {len(months)} months")
    
    monthly_results = []
    
    for month in months:
        # Get returns for this month (no need to copy if we're not modifying)
        month_mask = panel_df['date'] == month
        month_data = panel_df[month_mask]
        
        # Merge with betas (vectorized lookup instead of apply)
        # Create MultiIndex for efficient lookup
        month_data_indexed = month_data.set_index(['country', 'ticker'])
        month_data['beta'] = month_data_indexed.index.map(beta_dict).values
        month_data = month_data.reset_index(drop=True)
        
        # Drop stocks without beta or with NaN returns
        month_data = month_data.dropna(subset=['stock_return', 'beta'])
        
        if len(month_data) < 50:
            logger.warning(f"Month {month.strftime('%Y-%m')}: Only {len(month_data)} stocks, skipping")
            continue
        
        # Prepare regression data
        y = month_data['stock_return'].values  # R_{i,t}
        X = month_data['beta'].values  # β_i
        
        # Add constant for intercept
        X_with_const = sm.add_constant(X)
        
        # Run OLS regression
        try:
            model = sm.OLS(y, X_with_const)
            results = model.fit()
            
            # Extract coefficients
            gamma_0 = results.params[0]  # Intercept
            gamma_1 = results.params[1]  # Slope (market price of risk)
            
            # Extract standard errors
            gamma_0_se = results.bse[0]
            gamma_1_se = results.bse[1]
            
            # Extract t-statistics
            gamma_0_tstat = results.tvalues[0]
            gamma_1_tstat = results.tvalues[1]
            
            # Extract p-values
            gamma_0_pvalue = results.pvalues[0]
            gamma_1_pvalue = results.pvalues[1]
            
            # R-squared
            r_squared = results.rsquared
            
            monthly_results.append({
                'date': month,
                'gamma_0': gamma_0,
                'gamma_1': gamma_1,
                'gamma_0_se': gamma_0_se,
                'gamma_1_se': gamma_1_se,
                'gamma_0_tstat': gamma_0_tstat,
                'gamma_1_tstat': gamma_1_tstat,
                'gamma_0_pvalue': gamma_0_pvalue,
                'gamma_1_pvalue': gamma_1_pvalue,
                'r_squared': r_squared,
                'n_stocks': len(month_data)
            })
            
        except Exception as e:
            logger.error(f"Error in regression for {month.strftime('%Y-%m')}: {e}")
            continue
    
    monthly_coefs_df = pd.DataFrame(monthly_results)
    
    logger.info(f"\n Completed {len(monthly_coefs_df)} monthly regressions")
    logger.info(f"   Average stocks per month: {monthly_coefs_df['n_stocks'].mean():.1f}")
    logger.info(f"   Average gamma_1: {monthly_coefs_df['gamma_1'].mean():.4f}")
    logger.info(f"   Average gamma_0: {monthly_coefs_df['gamma_0'].mean():.4f}")
    
    return monthly_coefs_df


# -------------------------------------------------------------------
# STEP 3: AVERAGE COEFFICIENTS
# -------------------------------------------------------------------

def compute_fama_macbeth_averages(monthly_coefs_df: pd.DataFrame) -> Dict:
    """
    Compute average coefficients across months.
    
    Parameters
    ----------
    monthly_coefs_df : pd.DataFrame
        Monthly regression results
    
    Returns
    -------
    dict
        Dictionary with average coefficients and standard deviations
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 3: COMPUTING AVERAGE COEFFICIENTS")
    logger.info("="*70)
    
    avg_gamma_1 = monthly_coefs_df['gamma_1'].mean()
    avg_gamma_0 = monthly_coefs_df['gamma_0'].mean()
    
    std_gamma_1 = monthly_coefs_df['gamma_1'].std()
    std_gamma_0 = monthly_coefs_df['gamma_0'].std()
    
    logger.info(f"Average gamma_1 (market price of risk): {avg_gamma_1:.4f}")
    logger.info(f"Average gamma_0 (intercept): {avg_gamma_0:.4f}")
    logger.info(f"Std dev gamma_1: {std_gamma_1:.4f}")
    logger.info(f"Std dev gamma_0: {std_gamma_0:.4f}")
    
    return {
        'avg_gamma_1': avg_gamma_1,
        'avg_gamma_0': avg_gamma_0,
        'std_gamma_1': std_gamma_1,
        'std_gamma_0': std_gamma_0,
        'n_months': len(monthly_coefs_df)
    }


# -------------------------------------------------------------------
# STEP 4: COMPUTE FAMA-MACBETH T-STATISTICS
# -------------------------------------------------------------------

def compute_fama_macbeth_tstats(monthly_coefs_df: pd.DataFrame) -> Dict:
    """
    Compute Fama-MacBeth t-statistics.
    
    Fama-MacBeth standard errors:
    - SE(gamma_avg) = std(gamma) / sqrt(T) where T = number of months
    - t(gamma_avg) = gamma_avg / SE(gamma_avg)
    
    Parameters
    ----------
    monthly_coefs_df : pd.DataFrame
        Monthly regression results
    
    Returns
    -------
    dict
        Dictionary with averages, standard errors, t-statistics, and p-values
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 4: COMPUTING FAMA-MACBETH T-STATISTICS")
    logger.info("="*70)
    
    T = len(monthly_coefs_df)
    
    # Average coefficients
    avg_gamma_1 = monthly_coefs_df['gamma_1'].mean()
    avg_gamma_0 = monthly_coefs_df['gamma_0'].mean()
    
    # Standard deviations across months
    std_gamma_1 = monthly_coefs_df['gamma_1'].std()
    std_gamma_0 = monthly_coefs_df['gamma_0'].std()
    
    # Fama-MacBeth standard errors
    se_gamma_1 = std_gamma_1 / np.sqrt(T)
    se_gamma_0 = std_gamma_0 / np.sqrt(T)
    
    # t-statistics
    tstat_gamma_1 = avg_gamma_1 / se_gamma_1 if se_gamma_1 > 0 else np.nan
    tstat_gamma_0 = avg_gamma_0 / se_gamma_0 if se_gamma_0 > 0 else np.nan
    
    # p-values (two-tailed test, degrees of freedom = T - 1)
    df = T - 1
    pvalue_gamma_1 = 2 * (1 - stats.t.cdf(abs(tstat_gamma_1), df)) if not np.isnan(tstat_gamma_1) else np.nan
    pvalue_gamma_0 = 2 * (1 - stats.t.cdf(abs(tstat_gamma_0), df)) if not np.isnan(tstat_gamma_0) else np.nan
    
    results = {
        'avg_gamma_1': avg_gamma_1,
        'avg_gamma_0': avg_gamma_0,
        'std_gamma_1': std_gamma_1,
        'std_gamma_0': std_gamma_0,
        'se_gamma_1': se_gamma_1,
        'se_gamma_0': se_gamma_0,
        'tstat_gamma_1': tstat_gamma_1,
        'tstat_gamma_0': tstat_gamma_0,
        'pvalue_gamma_1': pvalue_gamma_1,
        'pvalue_gamma_0': pvalue_gamma_0,
        'n_months': T
    }
    
    logger.info(f"Average gamma_1: {avg_gamma_1:.4f}")
    logger.info(f"  SE: {se_gamma_1:.4f}")
    logger.info(f"  t-stat: {tstat_gamma_1:.3f}")
    logger.info(f"  p-value: {pvalue_gamma_1:.4f}")
    logger.info(f"  {' Significant' if abs(tstat_gamma_1) > 1.96 and pvalue_gamma_1 < 0.05 else ' Not significant'} (5% level)")
    
    logger.info(f"\nAverage gamma_0: {avg_gamma_0:.4f}")
    logger.info(f"  SE: {se_gamma_0:.4f}")
    logger.info(f"  t-stat: {tstat_gamma_0:.3f}")
    logger.info(f"  p-value: {pvalue_gamma_0:.4f}")
    logger.info(f"  {' Significant' if abs(tstat_gamma_0) > 1.96 and pvalue_gamma_0 < 0.05 else ' Not significant'} (5% level)")
    
    return results


# -------------------------------------------------------------------
# STEP 5: VISUALIZATIONS
# -------------------------------------------------------------------

def create_fama_macbeth_visualizations(
    monthly_coefs_df: pd.DataFrame,
    tstats: Dict,
    beta_avg_returns_df: pd.DataFrame
) -> None:
    """
    Create all Fama-MacBeth visualizations.
    
    Parameters
    ----------
    monthly_coefs_df : pd.DataFrame
        Monthly regression results
    tstats : dict
        Fama-MacBeth test statistics
    beta_avg_returns_df : pd.DataFrame
        Beta vs average return data
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 5: CREATING VISUALIZATIONS")
    logger.info("="*70)
    
    avg_gamma_1 = tstats['avg_gamma_1']
    avg_gamma_0 = tstats['avg_gamma_0']
    
    # 1. Time-series of gamma_1
    logger.info("Creating time-series plot of gamma_1...")
    plt.figure(figsize=(14, 6))
    plt.plot(monthly_coefs_df['date'], monthly_coefs_df['gamma_1'], 
             linewidth=1.5, alpha=0.7, label='Monthly γ₁')
    plt.axhline(y=avg_gamma_1, color='red', linestyle='--', linewidth=2, 
                label=f'Average γ̄₁ = {avg_gamma_1:.4f}')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('gamma_1 (Market Price of Risk)', fontsize=12)
    plt.title('Monthly Market Price of Risk (gamma_1)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Save to new organized structure
    gamma1_ts_path = os.path.join(RESULTS_CAPM_CROSSSECTIONAL_FIGURES_DIR, "gamma1_timeseries.png")
    plt.savefig(gamma1_ts_path, dpi=300, bbox_inches='tight')
    # Also save to legacy location
    legacy_gamma1_ts = os.path.join(RESULTS_FIGURES_DIR, "gamma1_timeseries.png")
    shutil.copy2(gamma1_ts_path, legacy_gamma1_ts)
    plt.close()
    logger.info(" Saved: gamma1_timeseries.png")
    
    # 2. Histogram of gamma_1
    logger.info("Creating histogram of gamma_1...")
    plt.figure(figsize=(10, 6))
    plt.hist(monthly_coefs_df['gamma_1'], bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=avg_gamma_1, color='red', linestyle='--', linewidth=2, 
                label=f'Average gamma_1 = {avg_gamma_1:.4f}')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.xlabel('gamma_1 (Market Price of Risk)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Monthly Market Price of Risk', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    # Save to new organized structure
    gamma1_hist_path = os.path.join(RESULTS_CAPM_CROSSSECTIONAL_FIGURES_DIR, "gamma1_histogram.png")
    plt.savefig(gamma1_hist_path, dpi=300, bbox_inches='tight')
    # Also save to legacy location
    legacy_gamma1_hist = os.path.join(RESULTS_FIGURES_DIR, "gamma1_histogram.png")
    shutil.copy2(gamma1_hist_path, legacy_gamma1_hist)
    plt.close()
    logger.info(" Saved: gamma1_histogram.png")
    
    # 3. Beta vs Average Return Scatter
    logger.info("Creating beta vs return scatter plot...")
    plt.figure(figsize=(12, 8))
    
    # Color by country - pre-compute country data to avoid repeated filtering
    countries = beta_avg_returns_df['country'].unique()
    country_data_dict = {
        country: beta_avg_returns_df[beta_avg_returns_df['country'] == country]
        for country in countries
    }
    for country in countries:
        country_data = country_data_dict[country]
        plt.scatter(country_data['beta'], country_data['avg_return'], 
                   label=country, alpha=0.6, s=50)
    
    # Add fitted line: R_avg = gamma_0 + gamma_1 * beta
    beta_range = np.linspace(beta_avg_returns_df['beta'].min(), 
                            beta_avg_returns_df['beta'].max(), 100)
    fitted_line = avg_gamma_0 + avg_gamma_1 * beta_range
    plt.plot(beta_range, fitted_line, 'r--', linewidth=2, 
            label=f'Fitted: R̄ = {avg_gamma_0:.3f} + {avg_gamma_1:.3f} × β')
    
    plt.xlabel('Beta (β)', fontsize=12)
    plt.ylabel('Average Realized Return (%)', fontsize=12)
    plt.title('Beta vs Average Realized Return', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Save to new organized structure
    beta_return_path = os.path.join(RESULTS_CAPM_CROSSSECTIONAL_FIGURES_DIR, "beta_vs_return_scatter.png")
    plt.savefig(beta_return_path, dpi=300, bbox_inches='tight')
    # Also save to legacy location
    legacy_beta_return = os.path.join(RESULTS_FIGURES_DIR, "beta_vs_return_scatter.png")
    shutil.copy2(beta_return_path, legacy_beta_return)
    plt.close()
    logger.info(" Saved: beta_vs_return_scatter.png")
    
    # 4. Cross-country summary (optional)
    logger.info("Creating cross-country summary...")
    # Note: country_summary computed but not used - kept for potential future use
    _ = beta_avg_returns_df.groupby('country').agg({
        'beta': ['mean', 'std'],
        'avg_return': ['mean', 'std']
    }).round(3)
    
    # Create a simple summary plot
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average beta by country
    country_avg_beta = beta_avg_returns_df.groupby('country')['beta'].mean().sort_values()
    ax1.barh(range(len(country_avg_beta)), country_avg_beta.values, color='steelblue', alpha=0.7)
    ax1.set_yticks(range(len(country_avg_beta)))
    ax1.set_yticklabels(country_avg_beta.index)
    ax1.set_xlabel('Average Beta', fontsize=12)
    ax1.set_title('Average Beta by Country', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Average return by country
    country_avg_return = beta_avg_returns_df.groupby('country')['avg_return'].mean().sort_values()
    ax2.barh(range(len(country_avg_return)), country_avg_return.values, color='green', alpha=0.7)
    ax2.set_yticks(range(len(country_avg_return)))
    ax2.set_yticklabels(country_avg_return.index)
    ax2.set_xlabel('Average Return (%)', fontsize=12)
    ax2.set_title('Average Return by Country', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    # Save to new organized structure
    fm_country_path = os.path.join(RESULTS_CAPM_CROSSSECTIONAL_FIGURES_DIR, "fama_macbeth_by_country.png")
    plt.savefig(fm_country_path, dpi=300, bbox_inches='tight')
    # Also save to legacy location
    legacy_fm_country = os.path.join(RESULTS_FIGURES_DIR, "fama_macbeth_by_country.png")
    shutil.copy2(fm_country_path, legacy_fm_country)
    plt.close()
    logger.info(" Saved: fama_macbeth_by_country.png")
    
    logger.info("\n All visualizations created successfully")


# -------------------------------------------------------------------
# REPORT GENERATION
# -------------------------------------------------------------------

def generate_fama_macbeth_report(
    monthly_coefs_df: pd.DataFrame,
    tstats: Dict,
    beta_avg_returns_df: pd.DataFrame
) -> None:
    """
    Generate and save Fama-MacBeth reports.
    
    Parameters
    ----------
    monthly_coefs_df : pd.DataFrame
        Monthly regression results
    tstats : dict
        Fama-MacBeth test statistics
    beta_avg_returns_df : pd.DataFrame
        Beta vs average return data
    """
    logger.info("\n" + "="*70)
    logger.info("GENERATING FAMA-MACBETH REPORTS")
    logger.info("="*70)
    
    # 1. Monthly coefficients - save to new organized structure
    monthly_path = os.path.join(RESULTS_CAPM_CROSSSECTIONAL_DIR, "monthly_coefficients.csv")
    monthly_coefs_df.to_csv(monthly_path, index=False)
    # Also save to legacy location
    legacy_monthly = os.path.join(RESULTS_REPORTS_DIR, "fama_macbeth_monthly_coefficients.csv")
    monthly_coefs_df.to_csv(legacy_monthly, index=False)
    logger.info(f" Saved: {monthly_path}")
    
    # 2. Summary statistics - save to new organized structure
    summary_df = pd.DataFrame([tstats])
    summary_path = os.path.join(RESULTS_CAPM_CROSSSECTIONAL_DIR, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    # Also save to legacy location
    legacy_summary = os.path.join(RESULTS_REPORTS_DIR, "fama_macbeth_summary.csv")
    summary_df.to_csv(legacy_summary, index=False)
    logger.info(f" Saved: {summary_path}")
    
    # 3. Beta vs average returns - save to new organized structure
    beta_returns_path = os.path.join(RESULTS_CAPM_CROSSSECTIONAL_DIR, "beta_returns.csv")
    beta_avg_returns_df.to_csv(beta_returns_path, index=False)
    # Also save to legacy location
    legacy_beta_returns = os.path.join(RESULTS_REPORTS_DIR, "fama_macbeth_beta_returns.csv")
    beta_avg_returns_df.to_csv(legacy_beta_returns, index=False)
    logger.info(f" Saved: {beta_returns_path}")
    
    logger.info("\n All reports generated successfully")


# -------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------

def run_fama_macbeth_test() -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """
    Run complete Fama-MacBeth test pipeline.
    
    Returns
    -------
    tuple
        (monthly_coefs_df, tstats, beta_avg_returns_df)
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    logger.info("Loading data...")
    panel_path = os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv")
    beta_path = os.path.join(RESULTS_DATA_DIR, "capm_results.csv")
    
    panel_df = pd.read_csv(panel_path, parse_dates=['date'])
    beta_df = pd.read_csv(beta_path)
    
    logger.info(f"Loaded panel: {len(panel_df)} rows")
    logger.info(f"Loaded betas: {len(beta_df)} stocks")
    
    # Step 1: Prepare cross-sectional data
    beta_avg_returns_df = prepare_cross_sectional_data(panel_df, beta_df)
    
    # Step 2: Run monthly regressions
    monthly_coefs_df = run_monthly_cross_sectional_regressions(panel_df, beta_df)
    
    # Step 3: Compute averages (stored in tstats dict below)
    _ = compute_fama_macbeth_averages(monthly_coefs_df)
    
    # Step 4: Compute t-statistics
    tstats = compute_fama_macbeth_tstats(monthly_coefs_df)
    
    # Step 5: Generate visualizations
    create_fama_macbeth_visualizations(monthly_coefs_df, tstats, beta_avg_returns_df)
    
    # Generate reports
    generate_fama_macbeth_report(monthly_coefs_df, tstats, beta_avg_returns_df)
    
    # Print summary
    print("\n" + "="*70)
    print("FAMA-MACBETH TEST COMPLETE")
    print("="*70)
    print(f"\n Results Summary:")
    print(f"   Number of months: {tstats['n_months']}")
    print(f"   Average stocks per month: {monthly_coefs_df['n_stocks'].mean():.1f}")
    
    print(f"\n Market Price of Risk (gamma_1):")
    print(f"   Average: {tstats['avg_gamma_1']:.4f}")
    print(f"   Standard Error: {tstats['se_gamma_1']:.4f}")
    print(f"   t-statistic: {tstats['tstat_gamma_1']:.3f}")
    print(f"   p-value: {tstats['pvalue_gamma_1']:.4f}")
    
    if abs(tstats['tstat_gamma_1']) > 1.96 and tstats['pvalue_gamma_1'] < 0.05:
        print(f"    SIGNIFICANT at 5% level → CAPM VALIDATED")
        print(f"      Higher beta → Higher return (as predicted by CAPM)")
    else:
        print(f"    NOT SIGNIFICANT at 5% level → CAPM REJECTED")
        print(f"      Beta does not explain cross-sectional variation in returns")
    
    print(f"\n Intercept (gamma_0):")
    print(f"   Average: {tstats['avg_gamma_0']:.4f}")
    print(f"   Standard Error: {tstats['se_gamma_0']:.4f}")
    print(f"   t-statistic: {tstats['tstat_gamma_0']:.3f}")
    print(f"   p-value: {tstats['pvalue_gamma_0']:.4f}")
    
    if abs(tstats['tstat_gamma_0']) < 1.96 or tstats['pvalue_gamma_0'] >= 0.05:
        print(f"    NOT SIGNIFICANT → No abnormal return (CAPM well-specified)")
    else:
        if tstats['avg_gamma_0'] > 0:
            print(f"     SIGNIFICANT POSITIVE → CAPM under-predicts returns")
        else:
            print(f"     SIGNIFICANT NEGATIVE → CAPM over-predicts returns")
    
    print(f"\n Output Files:")
    print(f"   Monthly coefficients: {os.path.join(RESULTS_REPORTS_DIR, 'fama_macbeth_monthly_coefficients.csv')}")
    print(f"   Summary: {os.path.join(RESULTS_REPORTS_DIR, 'fama_macbeth_summary.csv')}")
    print(f"   Beta vs Returns: {os.path.join(RESULTS_REPORTS_DIR, 'fama_macbeth_beta_returns.csv')}")
    print(f"   Plots: {RESULTS_FIGURES_DIR}/")
    print("="*70)
    
    return monthly_coefs_df, tstats, beta_avg_returns_df


if __name__ == "__main__":
    run_fama_macbeth_test()

