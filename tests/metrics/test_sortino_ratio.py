"""Tests for Sortino Ratio metric."""

import numpy as np
import pandas as pd

from src.metrics.sortino_ratio import SortinoRatio, calculate_sortino_ratio


class TestSortinoRatio:
    """Test Sortino Ratio calculations."""

    def test_basic_calculation_daily(self, daily_returns):
        """Test basic Sortino calculation with daily data."""
        metric = SortinoRatio(annualization_factor=252, minimum_acceptable_return=0.0)
        sortino = metric.calculate(daily_returns, annualized=True)

        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

    def test_basic_calculation_15m(self, intraday_15m_returns):
        """Test Sortino with 15-minute intraday data."""
        metric = SortinoRatio(annualization_factor=4320, minimum_acceptable_return=0.0)
        sortino = metric.calculate(intraday_15m_returns, annualized=True)

        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

    def test_positive_returns_no_downside_returns_zero(self, positive_returns):
        """All-positive returns have no downside and should return 0."""
        metric = SortinoRatio(minimum_acceptable_return=0.0)
        sortino = metric.calculate(positive_returns)

        assert sortino == 0.0

    def test_negative_returns_negative_sortino(self, negative_returns):
        """Negative returns should give negative Sortino."""
        metric = SortinoRatio(minimum_acceptable_return=0.0)
        sortino = metric.calculate(negative_returns)

        assert sortino < 0

    def test_sortino_higher_than_sharpe_with_positive_skew(self):
        """Sortino should be higher than Sharpe for positively skewed returns."""
        # More small losses, few large gains
        returns = pd.Series([-0.01] * 10 + [0.05, 0.06, 0.07])

        from src.metrics.sharpe_ratio import SharpeRatio

        sharpe = SharpeRatio().calculate(returns)
        sortino = SortinoRatio().calculate(returns)

        # Sortino penalizes only downside, so should be higher
        assert sortino > sharpe

    def test_mar_impact(self, mixed_returns):
        """Higher MAR should lower Sortino."""
        sortino_mar0 = SortinoRatio(minimum_acceptable_return=0.0).calculate(mixed_returns)
        sortino_mar5 = SortinoRatio(minimum_acceptable_return=0.05).calculate(mixed_returns)

        assert sortino_mar0 > sortino_mar5

    def test_annualized_vs_non_annualized(self, daily_returns):
        """Annualized Sortino should be sqrt(252) times non-annualized."""
        metric = SortinoRatio(annualization_factor=252)

        sortino_annual = metric.calculate(daily_returns, annualized=True)
        sortino_non_annual = metric.calculate(daily_returns, annualized=False)

        if not np.isinf(sortino_annual) and sortino_non_annual != 0:
            expected_ratio = np.sqrt(252)
            actual_ratio = sortino_annual / sortino_non_annual

            assert np.isclose(actual_ratio, expected_ratio, rtol=0.01)

    def test_downside_deviation_calculation(self, mixed_returns):
        """Test downside deviation calculation separately."""
        metric = SortinoRatio(minimum_acceptable_return=0.0, annualization_factor=252)

        dd = metric.calculate_downside_deviation(mixed_returns, annualized=False)

        assert dd > 0
        assert not np.isnan(dd)

    def test_downside_deviation_only_negative(self):
        """Downside deviation should only consider negative returns."""
        returns = pd.Series([0.02, 0.03, -0.01, -0.02, 0.01])
        metric = SortinoRatio(minimum_acceptable_return=0.0)

        dd = metric.calculate_downside_deviation(returns, annualized=False)

        # Manual calculation: only [-0.01, -0.02]
        downside = np.array([-0.01, -0.02])
        expected_dd = np.sqrt((downside**2).mean())

        assert np.isclose(dd, expected_dd, rtol=0.01)

    def test_different_input_types(self):
        """Test with Series, array, and list inputs."""
        returns_list = [0.01, 0.02, -0.01, 0.015, -0.02]
        returns_array = np.array(returns_list)
        returns_series = pd.Series(returns_list)

        metric = SortinoRatio()

        sortino_list = metric.calculate(returns_list)
        sortino_array = metric.calculate(returns_array)
        sortino_series = metric.calculate(returns_series)

        assert np.isclose(sortino_list, sortino_array)
        assert np.isclose(sortino_list, sortino_series)

    def test_convenience_function(self, daily_returns):
        """Test convenience function matches class method."""
        metric = SortinoRatio(annualization_factor=252, minimum_acceptable_return=0.02)
        sortino_class = metric.calculate(daily_returns)

        sortino_func = calculate_sortino_ratio(
            daily_returns, minimum_acceptable_return=0.02, annualization_factor=252
        )

        assert np.isclose(sortino_class, sortino_func)

    def test_empty_returns_returns_zero(self):
        """Empty returns should return 0."""
        metric = SortinoRatio()
        sortino = metric.calculate(pd.Series([], dtype=float))

        assert sortino == 0.0

    def test_nan_handling(self):
        """Test handling of NaN values."""
        returns = pd.Series([0.01, np.nan, -0.02, 0.015, np.nan, -0.01])
        metric = SortinoRatio()

        sortino = metric.calculate(returns)

        assert not np.isnan(sortino)

    def test_intraday_5m_bars(self):
        """Test with 5-minute bars."""
        np.random.seed(42)
        returns_5m = pd.Series(np.random.normal(0.0001, 0.003, 12960))

        metric = SortinoRatio(annualization_factor=12960, minimum_acceptable_return=0.02)
        sortino = metric.calculate(returns_5m)

        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

    def test_hourly_bars(self):
        """Test with hourly bars."""
        np.random.seed(42)
        returns_1h = pd.Series(np.random.normal(0.0005, 0.01, 1440))

        metric = SortinoRatio(annualization_factor=1440, minimum_acceptable_return=0.02)
        sortino = metric.calculate(returns_1h)

        assert isinstance(sortino, float)
        assert not np.isnan(sortino)


class TestSortinoRatioEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_return(self):
        """Single return should handle gracefully."""
        metric = SortinoRatio()
        sortino = metric.calculate(pd.Series([0.01]))

        assert sortino == 0.0

    def test_single_negative_return(self):
        """Single negative return should calculate."""
        metric = SortinoRatio()
        sortino = metric.calculate(pd.Series([-0.01]))

        assert isinstance(sortino, float)
        assert sortino < 0

    def test_all_nan_returns(self):
        """All NaN returns should return 0."""
        metric = SortinoRatio()
        sortino = metric.calculate(pd.Series([np.nan, np.nan, np.nan]))

        assert sortino == 0.0

    def test_extreme_positive_returns(self):
        """Test with extreme positive returns."""
        returns = pd.Series([0.5, 0.6, 0.7, 0.8])
        metric = SortinoRatio()

        sortino = metric.calculate(returns)

        assert sortino == 0.0

    def test_extreme_negative_returns(self):
        """Test with extreme negative returns."""
        returns = pd.Series([-0.5, -0.6, -0.7, -0.8])
        metric = SortinoRatio()

        sortino = metric.calculate(returns)

        assert sortino < 0
        assert not np.isinf(sortino)

    def test_zero_downside_deviation(self):
        """Test when downside deviation is zero."""
        # All returns above MAR
        returns = pd.Series([0.01, 0.02, 0.03, 0.04])
        metric = SortinoRatio(minimum_acceptable_return=0.0)

        sortino = metric.calculate(returns)

        assert sortino == 0.0

    def test_mar_above_all_returns(self):
        """Test when MAR is above all returns."""
        returns = pd.Series([0.001, 0.002, 0.0015, 0.001])
        # MAR = 10% annualized = 0.10/252 daily ≈ 0.0004
        metric = SortinoRatio(minimum_acceptable_return=0.10, annualization_factor=252)

        sortino = metric.calculate(returns, annualized=False)

        # All returns above MAR but close, should be positive or small
        assert isinstance(sortino, float)

    def test_very_small_returns(self):
        """Test with very small returns (numerical stability)."""
        returns = pd.Series([1e-10, -2e-10, 1e-10, -1.5e-10] * 100)
        metric = SortinoRatio()

        sortino = metric.calculate(returns)

        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

    def test_downside_deviation_no_downside(self):
        """Downside deviation should be 0 when no downside."""
        returns = pd.Series([0.01, 0.02, 0.03])
        metric = SortinoRatio(minimum_acceptable_return=0.0)

        dd = metric.calculate_downside_deviation(returns)

        assert dd == 0.0
