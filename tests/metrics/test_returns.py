"""Tests for returns calculation utilities."""

import numpy as np
import pandas as pd
import pytest

from src.metrics.returns import (
    calculate_annualized_return,
    calculate_cagr,
    calculate_cumulative_returns,
    calculate_returns,
    calculate_total_return,
    calculate_volatility,
)


class TestCalculateReturns:
    """Test returns calculation from prices."""

    def test_simple_returns(self):
        """Test simple returns calculation."""
        prices = pd.Series([100, 110, 105, 115])
        returns = calculate_returns(prices, method="simple")

        expected = pd.Series([np.nan, 0.10, -0.0454545, 0.0952381])
        assert np.allclose(returns.dropna(), expected.dropna(), rtol=0.01)

    def test_log_returns(self):
        """Test log returns calculation."""
        prices = pd.Series([100, 110, 105, 115])
        returns = calculate_returns(prices, method="log")

        # Log returns: ln(110/100), ln(105/110), ln(115/105)
        expected = pd.Series([np.nan, np.log(1.1), np.log(105 / 110), np.log(115 / 105)])
        assert np.allclose(returns.dropna(), expected.dropna(), rtol=0.01)

    def test_invalid_method(self):
        """Invalid method should raise ValueError."""
        prices = pd.Series([100, 110, 105])

        with pytest.raises(ValueError, match="Invalid method"):
            calculate_returns(prices, method="invalid")

    def test_log_returns_negative_prices(self):
        """Log returns with negative prices should raise ValueError."""
        prices = pd.Series([100, -110, 105])

        with pytest.raises(ValueError, match="strictly positive"):
            calculate_returns(prices, method="log")

    def test_empty_prices(self):
        """Empty prices should return empty series."""
        prices = pd.Series([], dtype=float)
        returns = calculate_returns(prices)

        assert len(returns) == 0

    def test_single_price(self):
        """Single price should return single NaN."""
        prices = pd.Series([100.0])
        returns = calculate_returns(prices)

        assert len(returns) == 1
        assert np.isnan(returns.iloc[0])


class TestCalculateCumulativeReturns:
    """Test cumulative returns calculation."""

    def test_basic_cumulative(self):
        """Test basic cumulative returns."""
        returns = pd.Series([0.1, 0.05, -0.02, 0.03])
        cumulative = calculate_cumulative_returns(returns, starting_value=100)

        # 100 * 1.1 * 1.05 * 0.98 * 1.03
        expected_final = 100 * 1.1 * 1.05 * 0.98 * 1.03
        assert np.isclose(cumulative.iloc[-1], expected_final, rtol=0.01)

    def test_starting_value_one(self):
        """Test with starting value of 1."""
        returns = pd.Series([0.1, 0.05, -0.02])
        cumulative = calculate_cumulative_returns(returns, starting_value=1.0)

        expected_final = 1.1 * 1.05 * 0.98
        assert np.isclose(cumulative.iloc[-1], expected_final, rtol=0.01)

    def test_all_positive_returns(self):
        """All positive returns should monotonically increase."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.01])
        cumulative = calculate_cumulative_returns(returns)

        assert all(cumulative.diff().dropna() > 0)

    def test_all_negative_returns(self):
        """All negative returns should monotonically decrease."""
        returns = pd.Series([-0.01, -0.02, -0.015, -0.01])
        cumulative = calculate_cumulative_returns(returns)

        assert all(cumulative.diff().dropna() < 0)

    def test_empty_returns(self):
        """Empty returns should return empty series."""
        returns = pd.Series([], dtype=float)
        cumulative = calculate_cumulative_returns(returns)

        assert len(cumulative) == 0


class TestCalculateTotalReturn:
    """Test total return calculation."""

    def test_basic_total_return(self):
        """Test basic total return calculation."""
        equity = pd.Series([100, 110, 105, 115, 120])
        total_return = calculate_total_return(equity, as_percentage=True)

        expected = ((120 - 100) / 100) * 100
        assert np.isclose(total_return, expected)

    def test_total_return_decimal(self):
        """Test total return as decimal."""
        equity = pd.Series([100, 110, 105, 115, 120])
        total_return = calculate_total_return(equity, as_percentage=False)

        expected = (120 - 100) / 100
        assert np.isclose(total_return, expected)

    def test_negative_total_return(self):
        """Test negative total return."""
        equity = pd.Series([100, 110, 95, 90])
        total_return = calculate_total_return(equity, as_percentage=True)

        expected = ((90 - 100) / 100) * 100
        assert np.isclose(total_return, expected)
        assert total_return < 0

    def test_zero_starting_value(self):
        """Zero starting value should return 0."""
        equity = pd.Series([0, 10, 20])
        total_return = calculate_total_return(equity)

        assert total_return == 0.0

    def test_single_value(self):
        """Single value should return 0."""
        equity = pd.Series([100.0])
        total_return = calculate_total_return(equity)

        assert total_return == 0.0

    def test_empty_equity(self):
        """Empty equity should return 0."""
        equity = pd.Series([], dtype=float)
        total_return = calculate_total_return(equity)

        assert total_return == 0.0


class TestCalculateAnnualizedReturn:
    """Test annualized return calculation."""

    def test_basic_annualized_daily(self):
        """Test basic annualized return with daily data."""
        # 10% return over 252 days
        returns = pd.Series([0.1 / 252] * 252)
        annualized = calculate_annualized_return(returns, periods_per_year=252)

        # Should be approximately 10%
        assert np.isclose(annualized, 10.0, rtol=0.5)

    def test_annualized_15m_bars(self):
        """Test annualized return with 15-minute bars."""
        # 4320 bars per year
        returns = pd.Series([0.0001] * 4320)
        annualized = calculate_annualized_return(returns, periods_per_year=4320)

        assert isinstance(annualized, float)
        assert not np.isnan(annualized)

    def test_negative_annualized_return(self):
        """Test negative annualized return."""
        returns = pd.Series([-0.001] * 252)
        annualized = calculate_annualized_return(returns, periods_per_year=252)

        assert annualized < 0

    def test_empty_returns(self):
        """Empty returns should return 0."""
        returns = pd.Series([], dtype=float)
        annualized = calculate_annualized_return(returns)

        assert annualized == 0.0

    def test_zero_periods_per_year(self):
        """Zero periods per year should return 0."""
        returns = pd.Series([0.01, 0.02])
        annualized = calculate_annualized_return(returns, periods_per_year=0)

        assert annualized == 0.0

    def test_complete_loss(self):
        """Complete loss should return -100%."""
        returns = pd.Series([-0.5, -0.5, -0.5])
        annualized = calculate_annualized_return(returns, periods_per_year=252)

        assert annualized == -100.0


class TestCalculateVolatility:
    """Test volatility calculation."""

    def test_basic_volatility_daily(self):
        """Test basic volatility with daily data."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        volatility = calculate_volatility(returns, annualized=True, periods_per_year=252)

        # Should be around 20% annualized (0.02 * sqrt(252))
        assert 25 < volatility < 35

    def test_volatility_15m_bars(self):
        """Test volatility with 15-minute bars."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0001, 0.005, 4320))

        volatility = calculate_volatility(returns, annualized=True, periods_per_year=4320)

        assert isinstance(volatility, float)
        assert volatility > 0

    def test_non_annualized_volatility(self):
        """Test non-annualized volatility."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.02])
        volatility = calculate_volatility(returns, annualized=False)

        expected = returns.std() * 100
        assert np.isclose(volatility, expected, rtol=0.01)

    def test_zero_volatility(self):
        """Zero volatility returns should give 0."""
        returns = pd.Series([0.01] * 100)
        volatility = calculate_volatility(returns)

        # Numerical precision: should be very close to 0
        assert np.isclose(volatility, 0.0, atol=1e-10)

    def test_empty_returns(self):
        """Empty returns should return 0."""
        returns = pd.Series([], dtype=float)
        volatility = calculate_volatility(returns)

        assert volatility == 0.0


class TestCalculateCAGR:
    """Test CAGR calculation."""

    def test_basic_cagr_daily(self):
        """Test basic CAGR with daily data."""
        # 20% return over 1 year (252 days)
        equity = pd.Series([100] + [100 * 1.2] * 252)
        cagr = calculate_cagr(equity, periods_per_year=252)

        # Should be approximately 20%
        assert np.isclose(cagr, 20.0, rtol=0.01)

    def test_cagr_15m_bars(self):
        """Test CAGR with 15-minute bars."""
        # 2 years of data
        equity = pd.Series([100] + [120] * (4320 * 2))
        cagr = calculate_cagr(equity, periods_per_year=4320)

        # 20% over 2 years = ~9.5% CAGR
        assert 9 < cagr < 11

    def test_negative_cagr(self):
        """Test negative CAGR."""
        equity = pd.Series([100] + [80] * 252)
        cagr = calculate_cagr(equity, periods_per_year=252)

        assert cagr < 0

    def test_flat_equity(self):
        """Flat equity should give 0 CAGR."""
        equity = pd.Series([100.0] * 252)
        cagr = calculate_cagr(equity, periods_per_year=252)

        assert np.isclose(cagr, 0.0, atol=0.01)

    def test_single_value(self):
        """Single value should return 0."""
        equity = pd.Series([100.0])
        cagr = calculate_cagr(equity)

        assert cagr == 0.0

    def test_empty_equity(self):
        """Empty equity should return 0."""
        equity = pd.Series([], dtype=float)
        cagr = calculate_cagr(equity)

        assert cagr == 0.0

    def test_zero_starting_value(self):
        """Zero starting value should return 0."""
        equity = pd.Series([0, 10, 20])
        cagr = calculate_cagr(equity)

        assert cagr == 0.0

    def test_complete_loss(self):
        """Complete loss should return -100%."""
        equity = pd.Series([100, 50, 10, 0])
        cagr = calculate_cagr(equity, periods_per_year=252)

        assert cagr == -100.0


class TestReturnsEdgeCases:
    """Test edge cases for returns utilities."""

    def test_nan_handling_in_prices(self):
        """Test NaN handling in price series."""
        prices = pd.Series([100, 110, np.nan, 120, 115])
        returns = calculate_returns(prices)

        # Should drop NaN and calculate on valid prices
        assert not all(np.isnan(returns))

    def test_very_small_prices(self):
        """Test with very small prices (numerical stability)."""
        prices = pd.Series([1e-10, 1.1e-10, 1.05e-10, 1.15e-10])
        returns = calculate_returns(prices)

        assert not any(np.isnan(returns.dropna()))

    def test_very_large_prices(self):
        """Test with very large prices."""
        prices = pd.Series([1e10, 1.1e10, 1.05e10, 1.15e10])
        returns = calculate_returns(prices)

        assert not any(np.isnan(returns.dropna()))

    def test_extreme_returns(self):
        """Test with extreme returns."""
        returns = pd.Series([0.5, -0.4, 0.6, -0.3])
        cumulative = calculate_cumulative_returns(returns)

        assert all(cumulative > 0)  # Should stay positive
        assert not any(np.isnan(cumulative))
