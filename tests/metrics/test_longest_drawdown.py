"""Tests for Longest Drawdown metric."""

import numpy as np
import pandas as pd

from src.metrics.longest_drawdown import LongestDrawdown, calculate_longest_drawdown


class TestLongestDrawdown:
    """Test Longest Drawdown Duration calculations."""

    def test_basic_calculation(self, equity_with_drawdown):
        """Test basic longest drawdown calculation."""
        metric = LongestDrawdown()
        longest = metric.calculate(equity_with_drawdown)

        assert isinstance(longest, int)
        assert longest >= 0

    def test_known_duration(self):
        """Test with known drawdown duration."""
        # Underwater from index 2 to 8 (7 periods)
        equity = pd.Series([100, 110, 120, 115, 110, 105, 100, 95, 90, 100, 110, 120])
        metric = LongestDrawdown()

        longest = metric.calculate(equity)

        # From peak at 120 (index 2) to recovery at 120 (index 11) = 8 underwater periods
        assert longest == 8

    def test_no_drawdown(self, equity_monotonic_up):
        """Monotonically increasing equity should have 0 longest drawdown."""
        metric = LongestDrawdown()
        longest = metric.calculate(equity_monotonic_up)

        assert longest == 0

    def test_multiple_drawdowns(self):
        """Test with multiple drawdowns of different lengths."""
        # First DD: 3 periods, Second DD: 5 periods
        equity = pd.Series([100, 110, 105, 100, 110, 120, 115, 110, 105, 100, 95, 120])
        metric = LongestDrawdown()

        longest = metric.calculate(equity)

        # Should return the longest one
        assert longest >= 5

    def test_calculate_all_periods(self):
        """Test getting all underwater periods."""
        equity = pd.Series([100, 110, 105, 110, 105, 100, 110])
        metric = LongestDrawdown()

        periods = metric.calculate_all_periods(equity)

        assert isinstance(periods, list)
        assert len(periods) > 0
        assert all(isinstance(p, int) for p in periods)

    def test_calculate_average_underwater(self):
        """Test average underwater period calculation."""
        equity = pd.Series([100, 110, 105, 110, 105, 100, 110, 105, 110])
        metric = LongestDrawdown()

        avg = metric.calculate_average_underwater(equity)

        assert isinstance(avg, float)
        assert avg >= 0

    def test_calculate_time_underwater(self):
        """Test percentage of time underwater."""
        # 5 out of 10 periods underwater
        equity = pd.Series([100, 110, 105, 100, 95, 90, 100, 110, 105, 110])
        metric = LongestDrawdown()

        pct_underwater = metric.calculate_time_underwater(equity)

        assert isinstance(pct_underwater, float)
        assert 0 <= pct_underwater <= 100

    def test_always_underwater(self):
        """Test when always underwater (never recovers)."""
        equity = pd.Series([100, 95, 90, 85, 80, 75])
        metric = LongestDrawdown()

        longest = metric.calculate(equity)
        pct_underwater = metric.calculate_time_underwater(equity)

        assert longest == len(equity) - 1  # All periods after peak
        assert pct_underwater > 80  # Most time underwater

    def test_empty_equity(self):
        """Empty equity should return 0."""
        metric = LongestDrawdown()
        longest = metric.calculate(pd.Series([], dtype=float))

        assert longest == 0

    def test_single_value(self):
        """Single value should return 0."""
        metric = LongestDrawdown()
        longest = metric.calculate(pd.Series([100.0]))

        assert longest == 0

    def test_two_values(self):
        """Two values should handle properly."""
        metric = LongestDrawdown()

        # Going up - no drawdown
        longest_up = metric.calculate(pd.Series([100.0, 110.0]))
        assert longest_up == 0

        # Going down - 1 period underwater
        longest_down = metric.calculate(pd.Series([100.0, 90.0]))
        assert longest_down == 1

    def test_different_input_types(self):
        """Test with Series, array, and list inputs."""
        equity_list = [100, 110, 105, 115, 110, 120]
        equity_array = np.array(equity_list)
        equity_series = pd.Series(equity_list)

        metric = LongestDrawdown()

        longest_list = metric.calculate(equity_list)
        longest_array = metric.calculate(equity_array)
        longest_series = metric.calculate(equity_series)

        assert longest_list == longest_array
        assert longest_list == longest_series

    def test_convenience_function(self, equity_with_drawdown):
        """Test convenience function matches class method."""
        metric = LongestDrawdown()
        longest_class = metric.calculate(equity_with_drawdown)

        longest_func = calculate_longest_drawdown(equity_with_drawdown)

        assert longest_class == longest_func

    def test_nan_handling(self):
        """Test handling of NaN values."""
        equity = pd.Series([100, 110, np.nan, 105, 100, np.nan, 110])
        metric = LongestDrawdown()

        longest = metric.calculate(equity)

        assert isinstance(longest, int)
        assert longest >= 0

    def test_intraday_equity_curve(self):
        """Test with intraday equity curve (many data points)."""
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.005, 4320)  # 15-min bars
        equity = pd.Series((1 + returns).cumprod() * 100000)

        metric = LongestDrawdown()
        longest = metric.calculate(equity)

        assert isinstance(longest, int)
        assert longest >= 0


class TestLongestDrawdownEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_immediate_recovery(self):
        """Test when recovery is immediate (1 period)."""
        equity = pd.Series([100, 110, 109, 110, 109, 110])
        metric = LongestDrawdown()

        periods = metric.calculate_all_periods(equity)

        assert all(p == 1 for p in periods)

    def test_long_drawdown_period(self):
        """Test with very long drawdown period."""
        # 99 periods underwater (from index 1 to 99)
        equity = pd.Series([100] + list(range(99, 0, -1)) + [100])
        metric = LongestDrawdown()

        longest = metric.calculate(equity)

        assert longest == 99

    def test_alternating_up_down(self):
        """Test with alternating up/down (many short drawdowns)."""
        equity = pd.Series([100, 110, 105, 115, 110, 120, 115, 125])
        metric = LongestDrawdown()

        periods = metric.calculate_all_periods(equity)

        # Each drawdown is 1 period
        assert all(p == 1 for p in periods)

    def test_flat_equity(self):
        """Test with flat equity (no change)."""
        equity = pd.Series([100.0] * 10)
        metric = LongestDrawdown()

        longest = metric.calculate(equity)

        assert longest == 0

    def test_all_nan_equity(self):
        """All NaN equity should return 0."""
        metric = LongestDrawdown()
        longest = metric.calculate(pd.Series([np.nan, np.nan, np.nan]))

        assert longest == 0

    def test_time_underwater_no_drawdown(self, equity_monotonic_up):
        """No drawdown should give 0% time underwater."""
        metric = LongestDrawdown()
        pct = metric.calculate_time_underwater(equity_monotonic_up)

        assert pct == 0.0

    def test_time_underwater_always_down(self):
        """Always underwater should give ~100% time underwater."""
        equity = pd.Series([100, 95, 90, 85, 80])
        metric = LongestDrawdown()

        pct = metric.calculate_time_underwater(equity)

        # 4 out of 5 periods underwater = 80%
        assert pct == 80.0

    def test_average_underwater_single_drawdown(self):
        """Single drawdown should have average equal to longest."""
        equity = pd.Series([100, 110, 105, 100, 95, 110])
        metric = LongestDrawdown()

        avg = metric.calculate_average_underwater(equity)
        longest = metric.calculate(equity)

        # Only one drawdown period
        assert avg == longest

    def test_average_underwater_multiple_drawdowns(self):
        """Multiple drawdowns should average correctly."""
        # Two drawdowns: 2 periods and 4 periods
        equity = pd.Series([100, 110, 105, 100, 110, 105, 100, 95, 90, 110])
        metric = LongestDrawdown()

        avg = metric.calculate_average_underwater(equity)

        # Should be between 2 and 4
        assert 2 <= avg <= 4

    def test_very_small_drawdowns(self):
        """Test with very small drawdowns (numerical precision)."""
        equity = pd.Series([100.0, 100.001, 99.999, 100.0, 100.001, 99.998, 100.0])
        metric = LongestDrawdown()

        longest = metric.calculate(equity)

        assert longest > 0  # Should detect small drawdowns

    def test_drawdown_at_end_unrecovered(self):
        """Test when drawdown is at end (unrecovered)."""
        equity = pd.Series([100, 110, 120, 115, 110, 105, 100, 95])
        metric = LongestDrawdown()

        longest = metric.calculate(equity)

        # Underwater from peak at 120 to end
        assert longest >= 5

    def test_recovery_at_exact_peak(self):
        """Test recovery at exact peak level."""
        equity = pd.Series([100, 110, 105, 100, 105, 110, 105, 110])
        metric = LongestDrawdown()

        periods = metric.calculate_all_periods(equity)

        # Should have multiple drawdown periods
        assert len(periods) > 0

    def test_multiple_peaks_same_level(self):
        """Test with multiple peaks at same level."""
        equity = pd.Series([100, 100, 95, 100, 100, 90, 100])
        metric = LongestDrawdown()

        longest = metric.calculate(equity)

        assert longest > 0
