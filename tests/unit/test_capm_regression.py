"""
Unit tests for CAPM regression functions
"""

import pytest
import pandas as pd
import numpy as np
from analysis.core.capm_regression import run_capm_regression


class TestCAPMRegression:
    """Test CAPM regression function."""
    
    def test_basic_regression(self):
        """Test basic CAPM regression with synthetic data."""
        # Create synthetic data
        np.random.seed(42)
        n = 59
        market_excess = np.random.normal(0, 2, n)
        stock_excess = 0.5 + 0.8 * market_excess + np.random.normal(0, 1, n)
        
        data = pd.DataFrame({
            'stock_excess_return': stock_excess,
            'market_excess_return': market_excess
        })
        
        result = run_capm_regression(data)
        
        # Check that result is a dictionary
        assert isinstance(result, dict)
        
        # Check required keys
        required_keys = ['beta', 'alpha', 'r_squared', 'pvalue_beta', 'n_obs']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check that beta is reasonable (should be close to 0.8)
        assert 0.5 < result['beta'] < 1.2, f"Beta {result['beta']} out of expected range"
        
        # Check that R² is positive
        assert result['r_squared'] >= 0, "R² should be non-negative"
        
        # Check n_obs
        assert result['n_obs'] == n, f"Expected {n} observations, got {result['n_obs']}"
    
    def test_insufficient_data(self):
        """Test regression with insufficient data."""
        # Create data with only 5 observations
        data = pd.DataFrame({
            'stock_excess_return': np.random.normal(0, 1, 5),
            'market_excess_return': np.random.normal(0, 1, 5)
        })
        
        result = run_capm_regression(data)
        
        # Should return NaN values
        assert np.isnan(result['beta']), "Beta should be NaN for insufficient data"
        assert np.isnan(result['alpha']), "Alpha should be NaN for insufficient data"
        assert result['n_obs'] == 5, "Should report correct number of observations"
    
    def test_missing_values(self):
        """Test regression with missing values."""
        np.random.seed(42)
        n = 59
        market_excess = np.random.normal(0, 2, n)
        stock_excess = 0.5 + 0.8 * market_excess + np.random.normal(0, 1, n)
        
        data = pd.DataFrame({
            'stock_excess_return': stock_excess,
            'market_excess_return': market_excess
        })
        
        # Add some missing values
        data.loc[10:15, 'stock_excess_return'] = np.nan
        data.loc[20:25, 'market_excess_return'] = np.nan
        
        result = run_capm_regression(data)
        
        # Should handle missing values (drop them)
        assert result['n_obs'] < n, "Should have fewer observations due to missing values"
        assert not np.isnan(result['beta']), "Beta should be calculated despite missing values"
    
    def test_perfect_correlation(self):
        """Test regression with perfect correlation."""
        np.random.seed(42)
        n = 59
        market_excess = np.random.normal(0, 2, n)
        stock_excess = 1.0 * market_excess  # Perfect correlation, beta = 1.0
        
        data = pd.DataFrame({
            'stock_excess_return': stock_excess,
            'market_excess_return': market_excess
        })
        
        result = run_capm_regression(data)
        
        # Beta should be close to 1.0
        assert abs(result['beta'] - 1.0) < 0.1, f"Beta should be ~1.0, got {result['beta']}"
        # R² should be very high (close to 1.0)
        assert result['r_squared'] > 0.99, f"R² should be very high, got {result['r_squared']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

