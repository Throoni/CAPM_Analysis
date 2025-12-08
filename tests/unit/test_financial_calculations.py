"""
Unit tests for financial calculations.

Tests all critical financial calculations to ensure correctness:
- Risk-free rate conversion (compounding vs simple)
- Return calculation formula
- Excess return calculation
- Beta calculation verification
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from analysis.data.riskfree_helper import (
    convert_annual_to_monthly_rate,
    convert_annual_pct_to_monthly_pct
)
from analysis.core.returns_processing import (
    prices_to_returns,
    create_excess_returns
)
from analysis.core.capm_regression import run_capm_regression


class TestRiskFreeRateConversion:
    """Test risk-free rate conversion formulas."""
    
    def test_compounding_formula(self):
        """Test that compounding formula is used (not simple division)."""
        # 3% annual should give ~0.247% monthly (compounding)
        # Simple division would give 0.25% monthly
        annual_3pct = 3.0
        monthly_compounding = convert_annual_pct_to_monthly_pct(annual_3pct)
        
        # Compounding: (1.03^(1/12) - 1) * 100 ≈ 0.2466%
        expected_compounding = 0.2466
        expected_simple = 0.25
        
        # Should be closer to compounding than simple
        assert abs(monthly_compounding - expected_compounding) < abs(monthly_compounding - expected_simple), \
            f"Conversion should use compounding, got {monthly_compounding:.4f}%"
        
        # Should be approximately 0.247%
        assert abs(monthly_compounding - expected_compounding) < 0.01, \
            f"Expected ~{expected_compounding}%, got {monthly_compounding:.4f}%"
    
    def test_low_rate_conversion(self):
        """Test conversion for low rates."""
        # 0.5% annual should give ~0.0416% monthly
        annual_0_5pct = 0.5
        monthly = convert_annual_pct_to_monthly_pct(annual_0_5pct)
        expected = 0.0416
        
        assert abs(monthly - expected) < 0.001, \
            f"Expected ~{expected}%, got {monthly:.4f}%"
    
    def test_decimal_conversion(self):
        """Test decimal form conversion."""
        # 5% annual in decimal = 0.05
        annual_decimal = 0.05
        monthly_decimal = convert_annual_to_monthly_rate(annual_decimal)
        
        # Should be approximately (1.05^(1/12) - 1) ≈ 0.00407
        expected = 0.00407
        assert abs(monthly_decimal - expected) < 0.0001, \
            f"Expected ~{expected}, got {monthly_decimal:.6f}"


class TestReturnCalculations:
    """Test return calculation formulas."""
    
    def test_simple_returns(self):
        """Test that simple returns (not log returns) are used."""
        # Prices: [100, 110, 105]
        # Simple returns: [NaN, 10%, -4.545%]
        # Log returns would be: [NaN, 9.53%, -4.65%]
        test_prices = pd.DataFrame({
            'TEST': [100, 110, 105]
        }, index=pd.date_range('2020-01-31', periods=3, freq='ME'))
        
        returns = prices_to_returns(test_prices)
        
        # After dropping NaN, should have 2 returns
        assert len(returns) == 2, "Should have 2 returns from 3 prices"
        
        # First return should be 10% (simple, not log)
        first_return = returns['TEST'].iloc[0]
        assert abs(first_return - 10.0) < 0.01, \
            f"Expected 10% (simple return), got {first_return:.4f}%"
        
        # Second return should be -4.545% (simple)
        second_return = returns['TEST'].iloc[1]
        expected_second = -4.545454545
        assert abs(second_return - expected_second) < 0.1, \
            f"Expected ~{expected_second}%, got {second_return:.4f}%"
    
    def test_return_percentage_form(self):
        """Test that returns are in percentage form (not decimal)."""
        test_prices = pd.DataFrame({
            'TEST': [100, 110]
        }, index=pd.date_range('2020-01-31', periods=2, freq='ME'))
        
        returns = prices_to_returns(test_prices)
        first_return = returns['TEST'].iloc[0]
        
        # Should be in percentage form (10, not 0.10)
        assert abs(first_return) >= 1.0, \
            f"Returns should be in percentage form, got {first_return:.4f}%"


class TestExcessReturnCalculations:
    """Test excess return calculations."""
    
    def test_excess_return_formula(self):
        """Test excess return calculation formula."""
        # Stock return: 1.0%, Market return: 0.5%, Risk-free: 0.1%
        # Stock excess: 1.0 - 0.1 = 0.9%
        # Market excess: 0.5 - 0.1 = 0.4%
        stock_ret = pd.Series([1.0, 2.0], index=pd.date_range('2021-01-31', periods=2, freq='ME'))
        market_ret = pd.Series([0.5, 1.0], index=pd.date_range('2021-01-31', periods=2, freq='ME'))
        rf = pd.Series([0.1, 0.2], index=pd.date_range('2021-01-31', periods=2, freq='ME'))
        
        result = create_excess_returns(stock_ret, market_ret, rf)
        
        # Check stock excess returns
        stock_excess = result['stock_excess_return'].values
        assert abs(stock_excess[0] - 0.9) < 0.001, \
            f"Expected 0.9%, got {stock_excess[0]:.4f}%"
        assert abs(stock_excess[1] - 1.8) < 0.001, \
            f"Expected 1.8%, got {stock_excess[1]:.4f}%"
        
        # Check market excess returns
        market_excess = result['market_excess_return'].values
        assert abs(market_excess[0] - 0.4) < 0.001, \
            f"Expected 0.4%, got {market_excess[0]:.4f}%"
        assert abs(market_excess[1] - 0.8) < 0.001, \
            f"Expected 0.8%, got {market_excess[1]:.4f}%"
    
    def test_excess_return_sign(self):
        """Test that excess returns can be negative."""
        # Negative stock return with positive risk-free rate
        stock_ret = pd.Series([-1.0], index=pd.date_range('2021-01-31', periods=1, freq='ME'))
        market_ret = pd.Series([-0.5], index=pd.date_range('2021-01-31', periods=1, freq='ME'))
        rf = pd.Series([0.1], index=pd.date_range('2021-01-31', periods=1, freq='ME'))
        
        result = create_excess_returns(stock_ret, market_ret, rf)
        
        # Stock excess should be -1.0 - 0.1 = -1.1%
        stock_excess = result['stock_excess_return'].values[0]
        assert abs(stock_excess - (-1.1)) < 0.001, \
            f"Expected -1.1%, got {stock_excess:.4f}%"


class TestBetaCalculation:
    """Test beta calculation in CAPM regression."""
    
    def test_perfect_correlation(self):
        """Test beta calculation with perfect correlation."""
        # If stock excess return = 2 * market excess return, beta should be 2.0
        dates = pd.date_range('2021-01-31', periods=10, freq='ME')
        market_excess = pd.Series([1.0, 2.0, -1.0, 0.5, 1.5, -0.5, 1.0, 2.0, -1.0, 0.5], index=dates)
        stock_excess = market_excess * 2.0  # Perfect correlation, beta = 2.0
        
        stock_data = pd.DataFrame({
            'stock_excess_return': stock_excess,
            'market_excess_return': market_excess
        })
        
        result = run_capm_regression(stock_data)
        
        # Beta should be approximately 2.0
        assert abs(result['beta'] - 2.0) < 0.01, \
            f"Expected beta ≈ 2.0, got {result['beta']:.4f}"
        
        # R-squared should be 1.0 (perfect fit)
        assert abs(result['r_squared'] - 1.0) < 0.01, \
            f"Expected R² ≈ 1.0, got {result['r_squared']:.4f}"
    
    def test_zero_correlation(self):
        """Test beta calculation with zero correlation."""
        # Random stock returns, zero correlation with market
        dates = pd.date_range('2021-01-31', periods=20, freq='ME')
        np.random.seed(42)
        market_excess = pd.Series(np.random.randn(20), index=dates)
        stock_excess = pd.Series(np.random.randn(20), index=dates)  # Independent
        
        stock_data = pd.DataFrame({
            'stock_excess_return': stock_excess,
            'market_excess_return': market_excess
        })
        
        result = run_capm_regression(stock_data)
        
        # Beta should be close to zero (low correlation)
        assert abs(result['beta']) < 1.0, \
            f"Expected |beta| < 1.0 for zero correlation, got {result['beta']:.4f}"
        
        # R-squared should be low
        assert result['r_squared'] < 0.5, \
            f"Expected R² < 0.5 for zero correlation, got {result['r_squared']:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

