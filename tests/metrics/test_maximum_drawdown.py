"""Tests for Maximum Drawdown metric."""

import numpy as np
import pandas as pd

from src.metrics.maximum_drawdown import (
    DrawdownInfo,
    MaximumDrawdown,
    calculate_max_drawdown,
)


class TestMaximumDrawdown:
    """Test Maximum Drawdown calculations."""

    def test_basic_calculation(self, equity_with_drawdown):
        """Test basic MDD calculation."""
        metric = MaximumDrawdown()
        mdd = metric.calculate(equity_with_drawdown)

        assert isinstance(mdd, float)
        assert mdd < 0  # Drawdown is negative
        assert not np.isnan(mdd)

    def test_known_drawdown(self):
        """Test with known drawdown values."""
        # Peak at 120, trough at 85
        # MDD = (85 - 120) / 120 = -29.17%
        equity = pd.Series([100, 110, 120, 115, 100, 90, 85, 95, 105])
        metric = MaximumDrawdown()

        mdd = metric.calculate(equity, as_percentage=True)

        expected_mdd = ((85 - 120) / 120) * 100
        assert np.isclose(mdd, expected_mdd, rtol=0.01)

    def test_no_drawdown(self, equity_monotonic_up):
        """Monotonically increasing equity should have 0 drawdown."""
        metric = MaximumDrawdown()
        mdd = metric.calculate(equity_monotonic_up)

        assert mdd == 0.0

    def test_percentage_vs_decimal(self, equity_with_drawdown):
        """Test percentage vs decimal output."""
        metric = MaximumDrawdown()

        mdd_pct = metric.calculate(equity_with_drawdown, as_percentage=True)
        mdd_dec = metric.calculate(equity_with_drawdown, as_percentage=False)

        assert np.isclose(mdd_pct, mdd_dec * 100)

    def test_calculate_with_info(self):
        """Test detailed drawdown info calculation."""
        equity = pd.Series([100, 110, 120, 115, 100, 90, 85, 95, 105, 115, 125])
        metric = MaximumDrawdown()

        info = metric.calculate_with_info(equity)

        assert isinstance(info, DrawdownInfo)
        assert info.peak_idx == 2  # Index of 120
        assert info.trough_idx == 6  # Index of 85
        assert info.peak_value == 120
        assert info.trough_value == 85
        assert info.duration == 4  # 6 - 2
        assert info.recovery_idx is not None

    def test_unrecovered_drawdown(self):
        """Test drawdown that hasn't recovered."""
        equity = pd.Series([100, 110, 120, 115, 100, 90, 85, 95])
        metric = MaximumDrawdown()

        info = metric.calculate_with_info(equity)

        assert info.recovery_idx is None
        assert info.recovery_duration is None

    def test_multiple_drawdowns(self):
        """Test finding multiple drawdowns."""
        equity = pd.Series([100, 110, 100, 110, 100, 90, 100, 110, 95, 110])
        metric = MaximumDrawdown()

        drawdowns = metric.calculate_all_drawdowns(equity, threshold=-5.0)

        # Should find drawdowns > 5%
        assert len(drawdowns) > 0
        assert all(isinstance(dd, DrawdownInfo) for dd in drawdowns)
        assert all(dd.max_drawdown <= -5.0 for dd in drawdowns)

    def test_threshold_filtering(self):
        """Test threshold filtering in calculate_all_drawdowns."""
        equity = pd.Series([100, 110, 108, 110, 105, 90, 100])
        metric = MaximumDrawdown()

        # Strict threshold
        drawdowns_strict = metric.calculate_all_drawdowns(equity, threshold=-15.0)
        # Loose threshold
        drawdowns_loose = metric.calculate_all_drawdowns(equity, threshold=-1.0)

        assert len(drawdowns_loose) >= len(drawdowns_strict)

    def test_empty_equity(self):
        """Empty equity should return 0."""
        metric = MaximumDrawdown()
        mdd = metric.calculate(pd.Series([], dtype=float))

        assert mdd == 0.0

    def test_single_value(self):
        """Single value should return 0."""
        metric = MaximumDrawdown()
        mdd = metric.calculate(pd.Series([100.0]))

        assert mdd == 0.0

    def test_two_values_up(self):
        """Two values going up should have 0 drawdown."""
        metric = MaximumDrawdown()
        mdd = metric.calculate(pd.Series([100.0, 110.0]))

        assert mdd == 0.0

    def test_two_values_down(self):
        """Two values going down should have drawdown."""
        metric = MaximumDrawdown()
        mdd = metric.calculate(pd.Series([100.0, 90.0]))

        expected = ((90 - 100) / 100) * 100
        assert np.isclose(mdd, expected)

    def test_different_input_types(self):
        """Test with Series, array, and list inputs."""
        equity_list = [100, 110, 105, 115, 100]
        equity_array = np.array(equity_list)
        equity_series = pd.Series(equity_list)

        metric = MaximumDrawdown()

        mdd_list = metric.calculate(equity_list)
        mdd_array = metric.calculate(equity_array)
        mdd_series = metric.calculate(equity_series)

        assert np.isclose(mdd_list, mdd_array)
        assert np.isclose(mdd_list, mdd_series)

    def test_convenience_function(self, equity_with_drawdown):
        """Test convenience function matches class method."""
        metric = MaximumDrawdown()
        mdd_class = metric.calculate(equity_with_drawdown)

        mdd_func = calculate_max_drawdown(equity_with_drawdown)

        assert np.isclose(mdd_class, mdd_func)

    def test_nan_handling(self):
        """Test handling of NaN values."""
        equity = pd.Series([100, 110, np.nan, 120, 100, np.nan, 90])
        metric = MaximumDrawdown()

        mdd = metric.calculate(equity)

        assert not np.isnan(mdd)

    def test_intraday_equity_curve(self):
        """Test with intraday equity curve (many data points)."""
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.005, 4320)  # 15-min bars
        equity = pd.Series((1 + returns).cumprod() * 100000)

        metric = MaximumDrawdown()
        mdd = metric.calculate(equity)

        assert isinstance(mdd, float)
        assert mdd <= 0
        assert not np.isnan(mdd)


class TestMaximumDrawdownEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extreme_drawdown(self):
        """Test with extreme drawdown (99%)."""
        equity = pd.Series([100, 110, 120, 10, 15])
        metric = MaximumDrawdown()

        mdd = metric.calculate(equity, as_percentage=True)

        expected = ((10 - 120) / 120) * 100
        assert np.isclose(mdd, expected, rtol=0.01)

    def test_complete_loss(self):
        """Test with complete loss (100% drawdown)."""
        equity = pd.Series([100, 110, 120, 0])
        metric = MaximumDrawdown()

        mdd = metric.calculate(equity, as_percentage=True)

        assert np.isclose(mdd, -100.0, rtol=0.01)

    def test_recovery_to_new_high(self):
        """Test recovery to new high."""
        equity = pd.Series([100, 120, 100, 110, 130])
        metric = MaximumDrawdown()

        info = metric.calculate_with_info(equity)

        assert info.recovery_idx is not None
        assert equity.iloc[info.recovery_idx] >= info.peak_value

    def test_multiple_peaks_same_height(self):
        """Test with multiple peaks at same level."""
        equity = pd.Series([100, 100, 90, 100, 100, 85, 100])
        metric = MaximumDrawdown()

        mdd = metric.calculate(equity, as_percentage=True)

        expected = ((85 - 100) / 100) * 100
        assert np.isclose(mdd, expected)

    def test_all_nan_equity(self):
        """All NaN equity should return 0."""
        metric = MaximumDrawdown()
        mdd = metric.calculate(pd.Series([np.nan, np.nan, np.nan]))

        assert mdd == 0.0

    def test_very_small_drawdown(self):
        """Test with very small drawdown (numerical precision)."""
        equity = pd.Series([100.0, 100.001, 99.999, 100.0])
        metric = MaximumDrawdown()

        mdd = metric.calculate(equity, as_percentage=True)

        assert mdd < 0
        assert abs(mdd) < 0.01  # Less than 0.01%

    def test_drawdown_at_end(self):
        """Test when drawdown is at the end (unrecovered)."""
        equity = pd.Series([100, 110, 120, 130, 120, 110, 100, 90])
        metric = MaximumDrawdown()

        info = metric.calculate_with_info(equity)

        assert info.recovery_idx is None
        assert info.trough_idx == len(equity) - 1

    def test_immediate_drawdown(self):
        """Test when drawdown starts immediately."""
        equity = pd.Series([100, 90, 80, 70, 80, 90, 100])
        metric = MaximumDrawdown()

        info = metric.calculate_with_info(equity)

        assert info.peak_idx == 0
        assert info.recovery_idx is not None

    def test_v_shaped_recovery(self):
        """Test V-shaped drawdown and recovery."""
        equity = pd.Series([100, 90, 80, 70, 80, 90, 100])
        metric = MaximumDrawdown()

        info = metric.calculate_with_info(equity)

        assert info.trough_idx == 3  # Index of 70
        assert info.recovery_idx == 6  # Index of 100
        assert info.recovery_duration == 3

    def test_zero_equity_values(self):
        """Test with zero equity values."""
        equity = pd.Series([100, 50, 0, 0, 10])
        metric = MaximumDrawdown()

        mdd = metric.calculate(equity)

        assert mdd <= -99.0  # At least 99% drawdown
