"""
generate_thesis_tables.py

Stage 7.1: Generate publication-ready tables for thesis chapter.

Creates all 6 tables in both CSV and LaTeX formats with proper formatting.
"""

import os
import logging
from typing import Dict
import pandas as pd
import numpy as np

from analysis.utils.config import (
    DATA_PROCESSED_DIR,
    RESULTS_DATA_DIR,
    RESULTS_REPORTS_DIR,
    COUNTRIES,
    MSCI_INDEX_TICKERS,
)

logger = logging.getLogger(__name__)

# Create tables directory if it doesn't exist
RESULTS_TABLES_DIR = os.path.join(os.path.dirname(RESULTS_REPORTS_DIR), "tables")
os.makedirs(RESULTS_TABLES_DIR, exist_ok=True)


def format_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """
    Convert DataFrame to LaTeX table format.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert
    caption : str
        Table caption
    label : str
        LaTeX label for referencing
    
    Returns
    -------
    str
        LaTeX table code
    """
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{" + caption + "}\n"
    latex += "\\label{" + label + "}\n"
    latex += "\\begin{tabular}{" + "l" + "c" * (len(df.columns) - 1) + "}\n"
    latex += "\\toprule\n"
    
    # Header
    headers = [col.replace('_', ' ').title() for col in df.columns]
    latex += " & ".join(headers) + " \\\\\n"
    latex += "\\midrule\n"
    
    # Data rows
    for idx, row in df.iterrows():
        values = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, (int, np.integer)):
                values.append(str(val))
            elif isinstance(val, (float, np.floating)):
                if abs(val) < 0.001:
                    values.append("0.000")
                else:
                    values.append(f"{val:.3f}")
            else:
                values.append(str(val))
        latex += " & ".join(values) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


def generate_table1_capm_timeseries() -> pd.DataFrame:
    """
    Generate Table 1: CAPM Time-Series Summary by Country.
    
    Returns
    -------
    pd.DataFrame
        Formatted table
    """
    logger.info("Generating Table 1: CAPM Time-Series Summary...")
    
    # Load data
    country_summary = pd.read_csv(os.path.join(RESULTS_REPORTS_DIR, "capm_by_country.csv"))
    capm_results = pd.read_csv(os.path.join(RESULTS_DATA_DIR, "capm_results.csv"))
    
    # Filter to only valid stocks for consistency
    valid_results = capm_results[capm_results['is_valid'] == True].copy()
    
    # Calculate % significant betas per country (only for valid stocks)
    significant_betas = valid_results.groupby('country').apply(
        lambda x: (x['pvalue_beta'] < 0.05).sum() / len(x) * 100,
        include_groups=False
    ).reset_index()
    significant_betas.columns = ['country', 'pct_significant_betas']
    
    # Merge
    table1 = country_summary.merge(significant_betas, on='country', how='left')
    
    # Select and rename columns
    table1 = table1[[
        'country', 'n_stocks', 'mean_beta', 'median_beta', 'std_beta',
        'mean_r_squared', 'median_r_squared', 'pct_significant_betas'
    ]].copy()
    
    # Round values
    table1['mean_beta'] = table1['mean_beta'].round(3)
    table1['median_beta'] = table1['median_beta'].round(3)
    table1['std_beta'] = table1['std_beta'].round(3)
    table1['mean_r_squared'] = table1['mean_r_squared'].round(3)
    table1['median_r_squared'] = table1['median_r_squared'].round(3)
    table1['pct_significant_betas'] = table1['pct_significant_betas'].round(1)
    
    # Rename columns
    table1.columns = [
        'Country', 'N Stocks', 'Mean Beta', 'Median Beta', 'Std Beta',
        'Mean R²', 'Median R²', '% Significant Betas'
    ]
    
    # Save
    table1.to_csv(os.path.join(RESULTS_TABLES_DIR, "table1_capm_timeseries_summary.csv"), index=False)
    
    # Generate LaTeX
    latex = format_latex_table(
        table1,
        "CAPM Time-Series Summary by Country",
        "tab:capm_timeseries"
    )
    with open(os.path.join(RESULTS_TABLES_DIR, "table1_capm_timeseries_summary.tex"), 'w') as f:
        f.write(latex)
    
    logger.info(" Table 1 generated")
    return table1


def generate_table2_fama_macbeth() -> pd.DataFrame:
    """
    Generate Table 2: Fama-MacBeth CAPM Test Results.
    
    Returns
    -------
    pd.DataFrame
        Formatted table
    """
    logger.info("Generating Table 2: Fama-MacBeth CAPM Test...")
    
    # Load data
    fm_summary = pd.read_csv(os.path.join(RESULTS_REPORTS_DIR, "fama_macbeth_summary.csv"))
    
    # Create table (check available columns)
    available_cols = fm_summary.columns.tolist()
    
    table2_data = {
        'gamma_0_mean': [fm_summary['avg_gamma_0'].iloc[0]],
        'gamma_0_tstat': [fm_summary['tstat_gamma_0'].iloc[0]],
        'gamma_0_pvalue': [fm_summary['pvalue_gamma_0'].iloc[0]],
        'gamma_1_mean': [fm_summary['avg_gamma_1'].iloc[0]],
        'gamma_1_tstat': [fm_summary['tstat_gamma_1'].iloc[0]],
        'gamma_1_pvalue': [fm_summary['pvalue_gamma_1'].iloc[0]],
        'n_months': [int(fm_summary['n_months'].iloc[0])]
    }
    
    # Add avg_r_squared if available (from monthly coefficients)
    if 'avg_r_squared' in available_cols:
        table2_data['avg_r_squared'] = [fm_summary['avg_r_squared'].iloc[0]]
    else:
        # Calculate from monthly coefficients if needed
        monthly_fm = pd.read_csv(os.path.join(RESULTS_REPORTS_DIR, "fama_macbeth_monthly_coefficients.csv"))
        table2_data['avg_r_squared'] = [monthly_fm['r_squared'].mean()]
    
    table2 = pd.DataFrame(table2_data)
    
    # Round values
    table2['gamma_0_mean'] = table2['gamma_0_mean'].round(4)
    table2['gamma_0_tstat'] = table2['gamma_0_tstat'].round(3)
    table2['gamma_0_pvalue'] = table2['gamma_0_pvalue'].round(4)
    table2['gamma_1_mean'] = table2['gamma_1_mean'].round(4)
    table2['gamma_1_tstat'] = table2['gamma_1_tstat'].round(3)
    table2['gamma_1_pvalue'] = table2['gamma_1_pvalue'].round(4)
    table2['avg_r_squared'] = table2['avg_r_squared'].round(3)
    
    # Rename columns
    table2.columns = [
        'γ₀ Mean', 't(γ₀)', 'p-value', 'γ₁ Mean', 't(γ₁)', 'p-value',
        'Avg Cross-Sectional R²', 'N Months'
    ]
    
    # Save
    table2.to_csv(os.path.join(RESULTS_TABLES_DIR, "table2_fama_macbeth_results.csv"), index=False)
    
    # Generate LaTeX
    latex = format_latex_table(
        table2,
        "Fama-MacBeth Cross-Sectional CAPM Test",
        "tab:fama_macbeth"
    )
    with open(os.path.join(RESULTS_TABLES_DIR, "table2_fama_macbeth_results.tex"), 'w') as f:
        f.write(latex)
    
    logger.info(" Table 2 generated")
    return table2


def generate_table3_subperiod() -> pd.DataFrame:
    """
    Generate Table 3: Subperiod Fama-MacBeth Results.
    
    Returns
    -------
    pd.DataFrame
        Formatted table
    """
    logger.info("Generating Table 3: Subperiod Fama-MacBeth Results...")
    
    # Load data
    subperiod = pd.read_csv(os.path.join(RESULTS_REPORTS_DIR, "fm_subperiod_comparison.csv"))
    
    # Create table
    table3 = subperiod[[
        'period', 'avg_gamma_0', 'tstat_gamma_0', 'avg_gamma_1', 'tstat_gamma_1',
        'n_months', 'avg_r_squared'
    ]].copy()
    
    # Round values
    table3['avg_gamma_0'] = table3['avg_gamma_0'].round(4)
    table3['tstat_gamma_0'] = table3['tstat_gamma_0'].round(3)
    table3['avg_gamma_1'] = table3['avg_gamma_1'].round(4)
    table3['tstat_gamma_1'] = table3['tstat_gamma_1'].round(3)
    table3['avg_r_squared'] = table3['avg_r_squared'].round(3)
    table3['n_months'] = table3['n_months'].astype(int)
    
    # Rename columns
    table3.columns = [
        'Period', 'γ₀ Mean', 't(γ₀)', 'γ₁ Mean', 't(γ₁)',
        'N Months', 'Avg R²'
    ]
    
    # Save
    table3.to_csv(os.path.join(RESULTS_TABLES_DIR, "table3_subperiod_results.csv"), index=False)
    
    # Generate LaTeX
    latex = format_latex_table(
        table3,
        "Fama-MacBeth Results by Subperiod",
        "tab:subperiod"
    )
    with open(os.path.join(RESULTS_TABLES_DIR, "table3_subperiod_results.tex"), 'w') as f:
        f.write(latex)
    
    logger.info(" Table 3 generated")
    return table3


def generate_table4_country_level() -> pd.DataFrame:
    """
    Generate Table 4: Country-Level Fama-MacBeth Results.
    
    Returns
    -------
    pd.DataFrame
        Formatted table
    """
    logger.info("Generating Table 4: Country-Level Fama-MacBeth Results...")
    
    # Load data
    country_fm = pd.read_csv(os.path.join(RESULTS_REPORTS_DIR, "fm_by_country.csv"))
    
    # Create table
    table4 = country_fm[[
        'country', 'avg_gamma_0', 'tstat_gamma_0', 'pvalue_gamma_0', 
        'avg_gamma_1', 'tstat_gamma_1', 'pvalue_gamma_1',
        'n_months', 'avg_r_squared'
    ]].copy()
    
    # Round values
    table4['avg_gamma_0'] = table4['avg_gamma_0'].round(4)
    table4['tstat_gamma_0'] = table4['tstat_gamma_0'].round(3)
    # Format p-values: use scientific notation for very small values (< 0.0001)
    table4['pvalue_gamma_0'] = table4['pvalue_gamma_0'].apply(
        lambda x: f"{x:.4e}" if x < 0.0001 and x > 0 else f"{x:.4f}"
    )
    table4['avg_gamma_1'] = table4['avg_gamma_1'].round(4)
    table4['tstat_gamma_1'] = table4['tstat_gamma_1'].round(3)
    table4['pvalue_gamma_1'] = table4['pvalue_gamma_1'].apply(
        lambda x: f"{x:.4e}" if x < 0.0001 and x > 0 else f"{x:.4f}"
    )
    table4['avg_r_squared'] = table4['avg_r_squared'].round(3)
    table4['n_months'] = table4['n_months'].astype(int)
    
    # Rename columns
    table4.columns = [
        'Country', 'γ₀ Mean', 't(γ₀)', 'p-value(γ₀)', 'γ₁ Mean', 't(γ₁)', 'p-value(γ₁)',
        'N Months', 'Avg R²'
    ]
    
    # Save
    table4.to_csv(os.path.join(RESULTS_TABLES_DIR, "table4_country_level_results.csv"), index=False)
    
    # Generate LaTeX
    latex = format_latex_table(
        table4,
        "Fama-MacBeth Results by Country",
        "tab:country_level"
    )
    with open(os.path.join(RESULTS_TABLES_DIR, "table4_country_level_results.tex"), 'w') as f:
        f.write(latex)
    
    logger.info(" Table 4 generated")
    return table4


def generate_table5_beta_portfolios() -> pd.DataFrame:
    """
    Generate Table 5: Beta-Sorted Portfolio Returns.
    
    Returns
    -------
    pd.DataFrame
        Formatted table
    """
    logger.info("Generating Table 5: Beta-Sorted Portfolio Returns...")
    
    # Load data
    portfolios = pd.read_csv(os.path.join(RESULTS_REPORTS_DIR, "beta_sorted_portfolios.csv"))
    
    # Create beta range column
    portfolios['beta_range'] = portfolios.apply(
        lambda row: f"{row['beta_min']:.3f} - {row['beta_max']:.3f}",
        axis=1
    )
    
    # Create table
    table5 = portfolios[[
        'portfolio', 'beta_range', 'portfolio_beta', 'avg_return', 'std_return', 'n_stocks'
    ]].copy()
    
    # Round values
    table5['portfolio_beta'] = table5['portfolio_beta'].round(3)
    table5['avg_return'] = table5['avg_return'].round(2)
    table5['std_return'] = table5['std_return'].round(2)
    table5['n_stocks'] = table5['n_stocks'].astype(int)
    
    # Rename columns
    table5.columns = [
        'Portfolio', 'Beta Range', 'Portfolio Beta', 'Average Return (%)',
        'Std Return (%)', 'N Stocks'
    ]
    
    # Save
    table5.to_csv(os.path.join(RESULTS_TABLES_DIR, "table5_beta_sorted_portfolios.csv"), index=False)
    
    # Generate LaTeX
    latex = format_latex_table(
        table5,
        "Beta-Sorted Portfolio Returns",
        "tab:beta_portfolios"
    )
    with open(os.path.join(RESULTS_TABLES_DIR, "table5_beta_sorted_portfolios.tex"), 'w') as f:
        f.write(latex)
    
    logger.info(" Table 5 generated")
    return table5


def generate_table6_descriptive() -> pd.DataFrame:
    """
    Generate Table 6: Descriptive Statistics.
    
    Returns
    -------
    pd.DataFrame
        Formatted table
    """
    logger.info("Generating Table 6: Descriptive Statistics...")
    
    # Load data to get stock counts
    capm_results = pd.read_csv(os.path.join(RESULTS_DATA_DIR, "capm_results.csv"))
    panel = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "returns_panel.csv"), parse_dates=['date'])
    
    # Get stock counts and date ranges by country
    # Filter to only valid stocks for consistency
    valid_results = capm_results[capm_results['is_valid'] == True].copy()
    
    stock_counts = valid_results.groupby('country')['ticker'].nunique()
    date_ranges = panel.groupby('country')['date'].agg(['min', 'max'])
    
    # Create table
    table6_data = []
    for country in sorted(COUNTRIES.keys()):
        n_stocks = stock_counts.get(country, 0)
        date_min = date_ranges.loc[country, 'min'] if country in date_ranges.index else None
        date_max = date_ranges.loc[country, 'max'] if country in date_ranges.index else None
        
        if date_min and date_max:
            date_range = f"{date_min.strftime('%Y-%m')} to {date_max.strftime('%Y-%m')}"
        else:
            date_range = "2021-01 to 2025-11"
        
        # Market proxy: All countries now use MSCI Europe (IEUR)
        # Risk-free rate: All countries use German 3-month Bund (EUR)
        rf_source = "German 3-month Bund (EUR)"
        
        table6_data.append({
            'Country': country,
            'N Stocks': n_stocks,
            'Date Range': date_range,
            'Market Proxy': "MSCI Europe (IEUR, EUR)",
            'Risk-Free Rate': rf_source
        })
    
    table6 = pd.DataFrame(table6_data)
    
    # Save
    table6.to_csv(os.path.join(RESULTS_TABLES_DIR, "table6_descriptive_statistics.csv"), index=False)
    
    # Generate LaTeX (need special formatting for multi-line cells)
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Descriptive Statistics}\n"
    latex += "\\label{tab:descriptive}\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\toprule\n"
    latex += "Country & N Stocks & Date Range & Market Proxy & Risk-Free Rate \\\\\n"
    latex += "\\midrule\n"
    
    for _, row in table6.iterrows():
        latex += f"{row['Country']} & {row['N Stocks']} & {row['Date Range']} & "
        latex += f"{row['Market Proxy']} & {row['Risk-Free Rate']} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    with open(os.path.join(RESULTS_TABLES_DIR, "table6_descriptive_statistics.tex"), 'w') as f:
        f.write(latex)
    
    logger.info(" Table 6 generated")
    return table6


def generate_all_tables() -> Dict[str, pd.DataFrame]:
    """
    Generate all tables for thesis chapter.
    
    Returns
    -------
    dict
        Dictionary of all tables
    """
    logger.info("="*70)
    logger.info("GENERATING ALL THESIS TABLES")
    logger.info("="*70)
    
    tables = {}
    
    tables['table1'] = generate_table1_capm_timeseries()
    tables['table2'] = generate_table2_fama_macbeth()
    tables['table3'] = generate_table3_subperiod()
    tables['table4'] = generate_table4_country_level()
    tables['table5'] = generate_table5_beta_portfolios()
    tables['table6'] = generate_table6_descriptive()
    
    logger.info("\n All 6 tables generated successfully!")
    logger.info(f"   Tables saved to: {RESULTS_TABLES_DIR}")
    
    return tables


if __name__ == "__main__":
    import sys
    from analysis.utils.config import DATA_PROCESSED_DIR
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    generate_all_tables()

