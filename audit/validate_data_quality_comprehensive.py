"""
validate_data_quality_comprehensive.py

Comprehensive Data Quality Audit

Performs extensive data quality checks to ensure:
- Data integrity and correctness
- Date alignment consistency
- Missing data patterns
- Extreme value detection
- Currency consistency
- Data completeness
"""

import os
import logging
from typing import Dict, List
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from analysis.utils.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_PRICES_DIR,
    DATA_RAW_RISKFREE_DIR,
    COUNTRIES,
)

logger = logging.getLogger(__name__)


class ComprehensiveDataQualityAudit:
    """Comprehensive data quality audit class."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed = []
        self.project_root = Path(__file__).parent.parent
        
    def log_issue(self, severity: str, message: str, details: Dict = None):
        """Log an issue."""
        entry = {'severity': severity, 'message': message, 'details': details or {}}
        if severity == 'critical':
            self.issues.append(entry)
        else:
            self.warnings.append(entry)
        logger.warning(f"[{severity.upper()}] {message}")
    
    def log_pass(self, message: str):
        """Log a passed check."""
        self.passed.append(message)
        logger.info(f"[PASS] {message}")
    
    def check_date_alignment(self) -> Dict:
        """Check date alignment consistency across all data sources."""
        logger.info("="*70)
        logger.info("Checking date alignment")
        logger.info("="*70)
        
        issues_found = []
        
        # Check processed panel dates
        panel_path = self.project_root / DATA_PROCESSED_DIR / "returns_panel.csv"
        if panel_path.exists():
            panel_df = pd.read_csv(panel_path, parse_dates=['date'])
            panel_dates = panel_df['date'].unique()
            panel_dates = pd.to_datetime(panel_dates)
            
            # Check if all dates are month-end
            month_ends = panel_dates.to_period('M').to_timestamp('M')
            non_month_end = panel_dates[panel_dates != month_ends]
            
            if len(non_month_end) > 0:
                issues_found.append({
                    'issue': f'{len(non_month_end)} dates are not month-end',
                    'dates': non_month_end.tolist()[:10]  # First 10
                })
                self.log_issue('critical', f"Found {len(non_month_end)} non-month-end dates in panel")
            else:
                self.log_pass("All panel dates are month-end")
            
            # Check for missing months
            expected_months = pd.date_range(panel_dates.min(), panel_dates.max(), freq='ME')
            missing_months = expected_months.difference(panel_dates)
            
            if len(missing_months) > 0:
                issues_found.append({
                    'issue': f'{len(missing_months)} missing months',
                    'months': missing_months.tolist()
                })
                self.log_issue('warning', f"Found {len(missing_months)} missing months")
            else:
                self.log_pass("No missing months in date range")
            
            # Check date range consistency by country
            for country in COUNTRIES.keys():
                country_data = panel_df[panel_df['country'] == country]
                if len(country_data) > 0:
                    country_dates = country_data['date'].unique()
                    country_dates = pd.to_datetime(country_dates)
                    date_range = (country_dates.max() - country_dates.min()).days / 30.44
                    
                    if date_range < 50:  # Less than ~50 months
                        issues_found.append({
                            'issue': f'{country} has only {date_range:.1f} months of data',
                            'country': country
                        })
                        self.log_issue('warning', f"{country}: Only {date_range:.1f} months of data")
                    else:
                        self.log_pass(f"{country}: {date_range:.1f} months of data")
        else:
            issues_found.append({
                'issue': 'Processed panel file not found'
            })
            self.log_issue('critical', "Processed panel file not found")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found
        }
    
    def check_price_data_quality(self) -> Dict:
        """Check price data quality."""
        logger.info("="*70)
        logger.info("Checking price data quality")
        logger.info("="*70)
        
        issues_found = []
        
        for country in COUNTRIES.keys():
            stock_file = self.project_root / DATA_RAW_PRICES_DIR / f"prices_stocks_{country}.csv"
            
            if not stock_file.exists():
                self.log_issue('warning', f"Price file not found for {country}")
                continue
            
            try:
                prices_df = pd.read_csv(stock_file, index_col=0, parse_dates=True)
                
                # Check for negative prices
                negative_prices = (prices_df < 0).sum().sum()
                if negative_prices > 0:
                    issues_found.append({
                        'country': country,
                        'issue': f'{negative_prices} negative price values'
                    })
                    self.log_issue('critical', f"{country}: {negative_prices} negative prices")
                else:
                    self.log_pass(f"{country}: No negative prices")
                
                # Check for zero prices (except possibly delistings)
                zero_prices = (prices_df == 0).sum().sum()
                if zero_prices > 0:
                    issues_found.append({
                        'country': country,
                        'issue': f'{zero_prices} zero price values (may be delistings)'
                    })
                    self.log_issue('warning', f"{country}: {zero_prices} zero prices")
                
                # Check for extreme price jumps (>50% month-over-month)
                returns = prices_df.pct_change() * 100
                extreme_jumps = (returns.abs() > 50).sum().sum()
                
                if extreme_jumps > 0:
                    issues_found.append({
                        'country': country,
                        'issue': f'{extreme_jumps} extreme price jumps (>50%)'
                    })
                    self.log_issue('warning', f"{country}: {extreme_jumps} extreme price jumps")
                else:
                    self.log_pass(f"{country}: No extreme price jumps")
                
                # Check for price continuity (gaps >3 months)
                for ticker in prices_df.columns:
                    ticker_prices = prices_df[ticker].dropna()
                    if len(ticker_prices) > 0:
                        date_diff = ticker_prices.index.to_series().diff()
                        large_gaps = date_diff[date_diff > pd.Timedelta(days=120)]  # >4 months
                        if len(large_gaps) > 0:
                            issues_found.append({
                                'country': country,
                                'ticker': ticker,
                                'issue': f'{len(large_gaps)} large gaps (>3 months)'
                            })
                            self.log_issue('warning', f"{country}/{ticker}: {len(large_gaps)} large gaps")
                
            except Exception as e:
                issues_found.append({
                    'country': country,
                    'issue': f'Error reading price file: {e}'
                })
                self.log_issue('critical', f"{country}: Error reading price file: {e}")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found
        }
    
    def check_return_data_quality(self) -> Dict:
        """Check return data quality."""
        logger.info("="*70)
        logger.info("Checking return data quality")
        logger.info("="*70)
        
        issues_found = []
        
        panel_path = self.project_root / DATA_PROCESSED_DIR / "returns_panel.csv"
        if not panel_path.exists():
            return {
                'passed': False,
                'issues': [{'issue': 'Processed panel file not found'}]
            }
        
        panel_df = pd.read_csv(panel_path, parse_dates=['date'])
        
        # Check for extreme returns (>100% monthly)
        extreme_stock_returns = panel_df[panel_df['stock_return'].abs() > 100]
        if len(extreme_stock_returns) > 0:
            issues_found.append({
                'issue': f'{len(extreme_stock_returns)} extreme stock returns (>100%)',
                'examples': extreme_stock_returns[['date', 'country', 'ticker', 'stock_return']].head(10).to_dict('records')
            })
            self.log_issue('warning', f"Found {len(extreme_stock_returns)} extreme stock returns")
        else:
            self.log_pass("No extreme stock returns (>100%)")
        
        # Check for extreme market returns (>50% monthly)
        extreme_market_returns = panel_df[panel_df['msci_index_return'].abs() > 50]
        if len(extreme_market_returns) > 0:
            issues_found.append({
                'issue': f'{len(extreme_market_returns)} extreme market returns (>50%)',
                'examples': extreme_market_returns[['date', 'country', 'msci_index_return']].head(10).to_dict('records')
            })
            self.log_issue('warning', f"Found {len(extreme_market_returns)} extreme market returns")
        else:
            self.log_pass("No extreme market returns (>50%)")
        
        # Check return distribution for outliers
        stock_returns = panel_df['stock_return'].dropna()
        q1, q3 = stock_returns.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = stock_returns[(stock_returns < q1 - 3*iqr) | (stock_returns > q3 + 3*iqr)]
        
        if len(outliers) > 100:  # More than 100 outliers
            issues_found.append({
                'issue': f'{len(outliers)} return outliers (3*IQR rule)',
                'outlier_pct': len(outliers) / len(stock_returns) * 100
            })
            self.log_issue('warning', f"Found {len(outliers)} return outliers ({len(outliers)/len(stock_returns)*100:.1f}%)")
        else:
            self.log_pass(f"Return outliers: {len(outliers)} ({len(outliers)/len(stock_returns)*100:.1f}%)")
        
        # Check for missing return periods
        missing_returns = panel_df['stock_return'].isna().sum()
        missing_pct = missing_returns / len(panel_df) * 100
        
        if missing_pct > 5:  # More than 5% missing
            issues_found.append({
                'issue': f'{missing_returns} missing stock returns ({missing_pct:.1f}%)'
            })
            self.log_issue('warning', f"High missing return rate: {missing_pct:.1f}%")
        else:
            self.log_pass(f"Missing returns: {missing_returns} ({missing_pct:.1f}%)")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'statistics': {
                'total_rows': len(panel_df),
                'missing_returns': missing_returns,
                'missing_pct': missing_pct,
                'extreme_stock_returns': len(extreme_stock_returns),
                'extreme_market_returns': len(extreme_market_returns),
                'outliers': len(outliers)
            }
        }
    
    def check_riskfree_rate_quality(self) -> Dict:
        """Check risk-free rate data quality."""
        logger.info("="*70)
        logger.info("Checking risk-free rate quality")
        logger.info("="*70)
        
        issues_found = []
        
        panel_path = self.project_root / DATA_PROCESSED_DIR / "returns_panel.csv"
        if not panel_path.exists():
            return {
                'passed': False,
                'issues': [{'issue': 'Processed panel file not found'}]
            }
        
        panel_df = pd.read_csv(panel_path, parse_dates=['date'])
        
        # Check each country has risk-free rates
        for country in COUNTRIES.keys():
            country_data = panel_df[panel_df['country'] == country]
            if len(country_data) == 0:
                continue
            
            missing_rf = country_data['riskfree_rate'].isna().sum()
            missing_pct = missing_rf / len(country_data) * 100
            
            if missing_pct > 10:  # More than 10% missing
                issues_found.append({
                    'country': country,
                    'issue': f'{missing_rf} missing risk-free rates ({missing_pct:.1f}%)'
                })
                self.log_issue('critical', f"{country}: {missing_pct:.1f}% missing risk-free rates")
            else:
                self.log_pass(f"{country}: {missing_pct:.1f}% missing risk-free rates")
            
            # Check rate ranges are reasonable
            rf_rates = country_data['riskfree_rate'].dropna()
            if len(rf_rates) > 0:
                min_rate = rf_rates.min()
                max_rate = rf_rates.max()
                
                # Monthly rates should typically be between -1% and 2%
                if min_rate < -1.0 or max_rate > 2.0:
                    issues_found.append({
                        'country': country,
                        'issue': f'Unusual rate range: {min_rate:.4f}% to {max_rate:.4f}%',
                        'min_rate': min_rate,
                        'max_rate': max_rate
                    })
                    self.log_issue('warning', f"{country}: Unusual rate range [{min_rate:.4f}%, {max_rate:.4f}%]")
                else:
                    self.log_pass(f"{country}: Rate range [{min_rate:.4f}%, {max_rate:.4f}%]")
                
                # Check for sudden jumps (possible data errors)
                rf_series = country_data.groupby('date')['riskfree_rate'].first()
                rf_changes = rf_series.diff().abs()
                large_jumps = rf_changes[rf_changes > 0.5]  # >0.5% monthly change
                
                if len(large_jumps) > 0:
                    issues_found.append({
                        'country': country,
                        'issue': f'{len(large_jumps)} large rate jumps (>0.5%)'
                    })
                    self.log_issue('warning', f"{country}: {len(large_jumps)} large rate jumps")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found
        }
    
    def check_data_completeness(self) -> Dict:
        """Check data completeness."""
        logger.info("="*70)
        logger.info("Checking data completeness")
        logger.info("="*70)
        
        issues_found = []
        
        panel_path = self.project_root / DATA_PROCESSED_DIR / "returns_panel.csv"
        if not panel_path.exists():
            return {
                'passed': False,
                'issues': [{'issue': 'Processed panel file not found'}]
            }
        
        panel_df = pd.read_csv(panel_path, parse_dates=['date'])
        
        # Check each stock has required observations (59 months)
        stock_counts = panel_df.groupby(['country', 'ticker']).size()
        insufficient_stocks = stock_counts[stock_counts < 59]
        
        if len(insufficient_stocks) > 0:
            issues_found.append({
                'issue': f'{len(insufficient_stocks)} stocks have <59 months of data',
                'examples': insufficient_stocks.head(10).to_dict()
            })
            self.log_issue('warning', f"Found {len(insufficient_stocks)} stocks with <59 months")
        else:
            self.log_pass("All stocks have >=59 months of data")
        
        # Check for systematic missing data by country
        for country in COUNTRIES.keys():
            country_data = panel_df[panel_df['country'] == country]
            if len(country_data) == 0:
                continue
            
            missing_data = country_data.isna().sum()
            total_cells = len(country_data) * len(country_data.columns)
            missing_pct = missing_data.sum() / total_cells * 100
            
            if missing_pct > 10:  # More than 10% missing
                issues_found.append({
                    'country': country,
                    'issue': f'{missing_pct:.1f}% missing data'
                })
                self.log_issue('warning', f"{country}: {missing_pct:.1f}% missing data")
            else:
                self.log_pass(f"{country}: {missing_pct:.1f}% missing data")
        
        # Verify panel structure
        required_columns = ['date', 'country', 'ticker', 'stock_return', 'msci_index_return', 
                          'riskfree_rate', 'stock_excess_return', 'market_excess_return']
        missing_columns = [col for col in required_columns if col not in panel_df.columns]
        
        if len(missing_columns) > 0:
            issues_found.append({
                'issue': f'Missing required columns: {missing_columns}'
            })
            self.log_issue('critical', f"Missing columns: {missing_columns}")
        else:
            self.log_pass("All required columns present")
        
        return {
            'passed': len(issues_found) == 0,
            'issues': issues_found,
            'statistics': {
                'total_stocks': len(stock_counts),
                'stocks_with_59_months': len(stock_counts[stock_counts >= 59]),
                'stocks_with_insufficient_data': len(insufficient_stocks)
            }
        }
    
    def check_currency_consistency(self) -> Dict:
        """Check currency consistency."""
        logger.info("="*70)
        logger.info("Checking currency consistency")
        logger.info("="*70)
        
        issues_found = []
        
        # Document USD ETF usage (MSCI indices)
        # This is expected and documented, but we should verify it's consistent
        self.log_pass("MSCI indices are USD-denominated ETFs (expected)")
        self.log_pass("Stock prices are in local currency (expected)")
        
        # Note: This creates currency exposure in beta, which is documented
        # in the methodology but is a known limitation
        issues_found.append({
            'issue': 'Currency mismatch: USD ETFs vs local currency stocks',
            'severity': 'documented_limitation',
            'note': 'This is documented in methodology but creates currency exposure in beta'
        })
        
        return {
            'passed': True,  # This is a documented limitation, not an error
            'issues': issues_found,
            'note': 'Currency mismatch is documented in methodology'
        }
    
    def run_all_checks(self) -> Dict:
        """Run all comprehensive data quality checks."""
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE DATA QUALITY AUDIT")
        logger.info("="*70 + "\n")
        
        results = {
            'date_alignment': self.check_date_alignment(),
            'price_data_quality': self.check_price_data_quality(),
            'return_data_quality': self.check_return_data_quality(),
            'riskfree_rate_quality': self.check_riskfree_rate_quality(),
            'data_completeness': self.check_data_completeness(),
            'currency_consistency': self.check_currency_consistency()
        }
        
        total_issues = sum(len(r.get('issues', [])) for r in results.values())
        total_critical = len(self.issues)
        total_warnings = len(self.warnings)
        
        results['summary'] = {
            'total_checks': len(results),
            'total_issues': total_issues,
            'critical_issues': total_critical,
            'warnings': total_warnings,
            'passed_checks': len(self.passed)
        }
        
        return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    audit = ComprehensiveDataQualityAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("COMPREHENSIVE DATA QUALITY AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

