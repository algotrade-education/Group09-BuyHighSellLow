"""Tests for Sharpe Ratio metric."""

import numpy as np
import pandas as pd

from src.metrics.sharpe_ratio import SharpeRatio, calculate_sharpe_ratio


class TestSharpeRatio:
    """Test Sharpe Ratio calculations."""

    def test_basic_calculation_daily(self, daily_returns):
        """Test basic Sharpe calculation with daily data."""
        metric = SharpeRatio(annualization_factor=252, risk_free_rate=0.02)
        sharpe = metric.calculate(daily_returns, annualized=True)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)

    def test_basic_calculation_15m(self, intraday_15m_returns):
        """Test Sharpe with 15-minute intraday data."""
        # 18 bars/day * 240 days = 4320 bars/year
        metric = SharpeRatio(annualization_factor=4320, risk_free_rate=0.02)
        sharpe = metric.calculate(intraday_15m_returns, annualized=True)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_positive_returns_positive_sharpe(self, positive_returns):
        """Positive returns should give positive Sharpe."""
        metric = SharpeRatio(risk_free_rate=0.0)
        sharpe = metric.calculate(positive_returns)

        assert sharpe > 0

    def test_negative_returns_negative_sharpe(self, negative_returns):
        """Negative returns should give negative Sharpe."""
        metric = SharpeRatio(risk_free_rate=0.0)
        sharpe = metric.calculate(negative_returns)

        assert sharpe < 0

    def test_zero_volatility_returns_zero(self, zero_returns):
        """Zero volatility should return 0."""
        metric = SharpeRatio()
        sharpe = metric.calculate(zero_returns)

        assert sharpe == 0.0

    def test_empty_returns_returns_zero(self):
        """Empty returns should return 0."""
        metric = SharpeRatio()
        sharpe = metric.calculate(pd.Series([], dtype=float))

        assert sharpe == 0.0

    def test_annualized_vs_non_annualized(self, daily_returns):
        """Annualized Sharpe should be sqrt(252) times non-annualized."""
        metric = SharpeRatio(annualization_factor=252)

        sharpe_annual = metric.calculate(daily_returns, annualized=True)
        sharpe_non_annual = metric.calculate(daily_returns, annualized=False)

        expected_ratio = np.sqrt(252)
        actual_ratio = sharpe_annual / sharpe_non_annual if sharpe_non_annual != 0 else 0

        assert np.isclose(actual_ratio, expected_ratio, rtol=0.01)

    def test_risk_free_rate_impact(self, positive_returns):
        """Higher risk-free rate should lower Sharpe."""
        sharpe_rf0 = SharpeRatio(risk_free_rate=0.0).calculate(positive_returns)
        sharpe_rf5 = SharpeRatio(risk_free_rate=0.05).calculate(positive_returns)

        assert sharpe_rf0 > sharpe_rf5

    def test_different_input_types(self):
        """Test with Series, array, and list inputs."""
        returns_list = [0.01, 0.02, -0.01, 0.015]
        returns_array = np.array(returns_list)
        returns_series = pd.Series(returns_list)

        metric = SharpeRatio()

        sharpe_list = metric.calculate(returns_list)
        sharpe_array = metric.calculate(returns_array)
        sharpe_series = metric.calculate(returns_series)

        assert np.isclose(sharpe_list, sharpe_array)
        assert np.isclose(sharpe_list, sharpe_series)

    def test_convenience_function(self, daily_returns):
        """Test convenience function matches class method."""
        metric = SharpeRatio(annualization_factor=252, risk_free_rate=0.02)
        sharpe_class = metric.calculate(daily_returns)

        sharpe_func = calculate_sharpe_ratio(
            daily_returns, risk_free_rate=0.02, annualization_factor=252
        )

        assert np.isclose(sharpe_class, sharpe_func)

    def test_known_values(self):
        """Test against known Sharpe values."""
        # Returns with known mean and std
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        metric = SharpeRatio(annualization_factor=252, risk_free_rate=0.0)

        sharpe = metric.calculate(returns, annualized=False)

        # Manual calculation
        expected = returns.mean() / returns.std()
        assert np.isclose(sharpe, expected, rtol=0.01)

    def test_high_volatility_lowers_sharpe(self):
        """Higher volatility should lower Sharpe for same mean."""
        np.random.seed(42)
        returns_low_vol = pd.Series(np.random.normal(0.001, 0.01, 252))
        returns_high_vol = pd.Series(np.random.normal(0.001, 0.03, 252))

        metric = SharpeRatio()

        sharpe_low = metric.calculate(returns_low_vol)
        sharpe_high = metric.calculate(returns_high_vol)

        # Same mean, higher vol = lower Sharpe
        assert sharpe_low > sharpe_high

    def test_nan_handling(self):
        """Test handling of NaN values."""
        returns = pd.Series([0.01, np.nan, 0.02, 0.015, np.nan, -0.01])
        metric = SharpeRatio()

        sharpe = metric.calculate(returns)

        assert not np.isnan(sharpe)

    def test_intraday_5m_bars(self):
        """Test with 5-minute bars (more granular)."""
        # 5-min bars: ~54 bars/day * 240 days = 12960 bars/year
        np.random.seed(42)
        returns_5m = pd.Series(np.random.normal(0.00005, 0.003, 12960))

        metric = SharpeRatio(annualization_factor=12960, risk_free_rate=0.02)
        sharpe = metric.calculate(returns_5m)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_hourly_bars(self):
        """Test with hourly bars."""
        # Hourly: ~6 bars/day * 240 days = 1440 bars/year
        np.random.seed(42)
        returns_1h = pd.Series(np.random.normal(0.0005, 0.01, 1440))

        metric = SharpeRatio(annualization_factor=1440, risk_free_rate=0.02)
        sharpe = metric.calculate(returns_1h)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)


class TestSharpeRatioEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_return(self):
        """Single return should return 0 (no std)."""
        metric = SharpeRatio()
        sharpe = metric.calculate(pd.Series([0.01]))

        # Single value has no std, should return 0
        assert sharpe == 0.0 or np.isnan(sharpe)

    def test_two_returns(self):
        """Two returns should calculate properly."""
        metric = SharpeRatio()
        sharpe = metric.calculate(pd.Series([0.01, 0.02]))

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_all_nan_returns(self):
        """All NaN returns should return 0."""
        metric = SharpeRatio()
        sharpe = metric.calculate(pd.Series([np.nan, np.nan, np.nan]))

        assert sharpe == 0.0

    def test_extreme_positive_returns(self):
        """Test with extreme positive returns."""
        returns = pd.Series([0.5, 0.6, 0.7, 0.8])
        metric = SharpeRatio()

        sharpe = metric.calculate(returns)

        assert sharpe > 0
        assert not np.isinf(sharpe)

    def test_extreme_negative_returns(self):
        """Test with extreme negative returns."""
        returns = pd.Series([-0.5, -0.6, -0.7, -0.8])
        metric = SharpeRatio()

        sharpe = metric.calculate(returns)

        assert sharpe < 0
        assert not np.isinf(sharpe)

    def test_very_small_returns(self):
        """Test with very small returns (numerical stability)."""
        returns = pd.Series([1e-10, 2e-10, -1e-10, 1.5e-10] * 100)
        metric = SharpeRatio()

        sharpe = metric.calculate(returns)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
