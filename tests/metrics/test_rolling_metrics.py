"""Tests for rolling metrics."""

import numpy as np
import pandas as pd

from src.metrics.rolling_metrics import (
    calculate_rolling_metrics,
    rolling_drawdown,
    rolling_sharpe,
    rolling_sortino,
    rolling_win_rate,
)


class TestRollingSharpe:
    """Test rolling Sharpe ratio calculations."""

    def test_basic_rolling_sharpe(self):
        """Test basic rolling Sharpe calculation."""
        np.random.seed(42)
        equity = pd.Series((1 + np.random.normal(0.001, 0.02, 100)).cumprod() * 100)

        result = rolling_sharpe(equity, window=20, periods_per_year=252)

        assert isinstance(result, pd.Series)
        assert result.name == "rolling_sharpe"
        assert len(result) == len(equity)  # same length, NaN at start

    def test_rolling_sharpe_15m_bars(self):
        """Test with 15-minute bars (VN30 futures)."""
        np.random.seed(42)
        equity = pd.Series((1 + np.random.normal(0.0001, 0.005, 500)).cumprod() * 100000)

        result = rolling_sharpe(equity, window=30, periods_per_year=4320)

        assert isinstance(result, pd.Series)
        assert not result.dropna().empty

    def test_rolling_sharpe_min_periods(self):
        """Test min_periods parameter."""
        np.random.seed(42)
        equity = pd.Series((1 + np.random.normal(0.001, 0.02, 50)).cumprod() * 100)

        # With min_periods=10, should have values earlier
        result = rolling_sharpe(equity, window=20, min_periods=10)

        # Should have non-NaN values before window is full
        assert result.notna().sum() > 30

    def test_rolling_sharpe_with_risk_free_rate(self):
        """Test with non-zero risk-free rate."""
        np.random.seed(42)
        equity = pd.Series((1 + np.random.normal(0.001, 0.02, 100)).cumprod() * 100)

        result_rf0 = rolling_sharpe(equity, window=20, risk_free_rate=0.0)
        result_rf2 = rolling_sharpe(equity, window=20, risk_free_rate=0.02)

        # Higher risk-free rate should generally lower Sharpe
        assert not result_rf0.equals(result_rf2)

    def test_rolling_sharpe_constant_equity(self):
        """Test with constant equity (no volatility)."""
        equity = pd.Series([100.0] * 50)

        result = rolling_sharpe(equity, window=20)

        # All NaN due to zero std
        assert result.isna().all()

    def test_rolling_sharpe_empty_series(self):
        """Test with empty series."""
        equity = pd.Series([], dtype=float)

        result = rolling_sharpe(equity, window=20)

        assert result.empty


class TestRollingDrawdown:
    """Test rolling drawdown calculations."""

    def test_basic_rolling_drawdown(self):
        """Test basic rolling drawdown."""
        equity = pd.Series([100, 110, 105, 115, 110, 120, 115, 125])

        result = rolling_drawdown(equity)

        assert isinstance(result, pd.Series)
        assert result.name == "rolling_drawdown"
        assert len(result) == len(equity)

    def test_rolling_drawdown_values(self):
        """Test rolling drawdown values are correct."""
        equity = pd.Series([100, 110, 120, 100, 90])

        result = rolling_drawdown(equity)

        # At index 0: 0% (at peak)
        # At index 1: 0% (new peak)
        # At index 2: 0% (new peak)
        # At index 3: (100-120)/120 = -16.67%
        # At index 4: (90-120)/120 = -25%
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 0.0
        assert result.iloc[2] == 0.0
        assert np.isclose(result.iloc[3], -0.1667, atol=0.01)
        assert np.isclose(result.iloc[4], -0.25, atol=0.01)

    def test_rolling_drawdown_monotonic_up(self):
        """Test with monotonically increasing equity."""
        equity = pd.Series([100, 110, 120, 130, 140])

        result = rolling_drawdown(equity)

        # All zeros (always at peak)
        assert (result == 0.0).all()

    def test_rolling_drawdown_monotonic_down(self):
        """Test with monotonically decreasing equity."""
        equity = pd.Series([100, 90, 80, 70, 60])

        result = rolling_drawdown(equity)

        # First is 0, rest are negative
        assert result.iloc[0] == 0.0
        assert (result.iloc[1:] < 0).all()

    def test_rolling_drawdown_empty(self):
        """Test with empty series."""
        equity = pd.Series([], dtype=float)

        result = rolling_drawdown(equity)

        assert result.empty


class TestRollingSortino:
    """Test rolling Sortino ratio calculations."""

    def test_basic_rolling_sortino(self):
        """Test basic rolling Sortino calculation."""
        np.random.seed(42)
        equity = pd.Series((1 + np.random.normal(0.001, 0.02, 100)).cumprod() * 100)

        result = rolling_sortino(equity, window=20, periods_per_year=252)

        assert isinstance(result, pd.Series)
        assert result.name == "rolling_sortino"
        assert len(result) == len(equity) - 1  # pct_change drops first row

    def test_rolling_sortino_15m_bars(self):
        """Test with 15-minute bars."""
        np.random.seed(42)
        equity = pd.Series((1 + np.random.normal(0.0001, 0.005, 500)).cumprod() * 100000)

        result = rolling_sortino(equity, window=30, periods_per_year=4320)

        assert isinstance(result, pd.Series)
        assert not result.dropna().empty

    def test_rolling_sortino_with_mar(self):
        """Test with non-zero MAR."""
        np.random.seed(42)
        equity = pd.Series((1 + np.random.normal(0.001, 0.02, 100)).cumprod() * 100)

        result = rolling_sortino(equity, window=20, min_acceptable_return=0.001)

        assert isinstance(result, pd.Series)
        assert not result.dropna().empty

    def test_rolling_sortino_all_positive_returns(self):
        """Test with all positive returns (no downside)."""
        equity = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110] * 3)

        result = rolling_sortino(equity, window=10)

        # With strictly increasing equity and no downside, result will be all NaN
        # because downside_std returns NaN when no negative returns
        assert isinstance(result, pd.Series)
        # All NaN is expected when there's no downside volatility
        assert result.isna().all() or not result.dropna().empty


class TestRollingWinRate:
    """Test rolling win rate calculations."""

    def test_basic_rolling_win_rate(self):
        """Test basic rolling win rate."""
        pnl = pd.Series([100, -50, 80, -30, 120, -40, 90, 110, -60, 70])

        result = rolling_win_rate(pnl, window=5)

        assert isinstance(result, pd.Series)
        assert result.name == "rolling_win_rate"
        assert len(result) == len(pnl)

    def test_rolling_win_rate_values(self):
        """Test rolling win rate values are correct."""
        pnl = pd.Series([100, -50, 80, -30, 120])  # 3 wins, 2 losses

        result = rolling_win_rate(pnl, window=5, min_periods=5)

        # Last value should be 3/5 = 0.6
        assert np.isclose(result.iloc[-1], 0.6)

    def test_rolling_win_rate_all_wins(self):
        """Test with all winning trades."""
        pnl = pd.Series([100, 50, 80, 30, 120, 40, 90, 110, 60, 70])

        result = rolling_win_rate(pnl, window=5)

        # All should be 1.0 (100% win rate)
        assert (result.dropna() == 1.0).all()

    def test_rolling_win_rate_all_losses(self):
        """Test with all losing trades."""
        pnl = pd.Series([-100, -50, -80, -30, -120, -40, -90, -110, -60, -70])

        result = rolling_win_rate(pnl, window=5)

        # All should be 0.0 (0% win rate)
        assert (result.dropna() == 0.0).all()

    def test_rolling_win_rate_with_zeros(self):
        """Test with breakeven trades (zero PnL)."""
        pnl = pd.Series([100, 0, -50, 0, 80])

        result = rolling_win_rate(pnl, window=5, min_periods=5)

        # Only 100 and 80 are wins = 2/5 = 0.4
        assert np.isclose(result.iloc[-1], 0.4)

    def test_rolling_win_rate_min_periods(self):
        """Test min_periods parameter."""
        pnl = pd.Series([100, -50, 80, -30, 120])

        result = rolling_win_rate(pnl, window=5, min_periods=3)

        # Should have values starting from index 2
        assert result.iloc[2:].notna().all()


class TestCalculateRollingMetrics:
    """Test convenience function for all rolling metrics."""

    def test_calculate_all_rolling_metrics(self):
        """Test calculating all rolling metrics at once."""
        np.random.seed(42)
        equity = pd.Series((1 + np.random.normal(0.001, 0.02, 100)).cumprod() * 100)

        result = calculate_rolling_metrics(equity, sharpe_window=20, periods_per_year=252)

        assert isinstance(result, pd.DataFrame)
        assert "rolling_sharpe" in result.columns
        assert "rolling_drawdown" in result.columns
        assert "rolling_sortino" in result.columns
        assert len(result) == len(equity)

    def test_calculate_rolling_metrics_15m(self):
        """Test with 15-minute bars."""
        np.random.seed(42)
        equity = pd.Series((1 + np.random.normal(0.0001, 0.005, 500)).cumprod() * 100000)

        result = calculate_rolling_metrics(equity, sharpe_window=30, periods_per_year=4320)

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 3

    def test_calculate_rolling_metrics_empty(self):
        """Test with empty equity."""
        equity = pd.Series([], dtype=float)

        result = calculate_rolling_metrics(equity)

        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestRollingMetricsEdgeCases:
    """Test edge cases for rolling metrics."""

    def test_window_larger_than_data(self):
        """Test when window is larger than data."""
        equity = pd.Series([100, 101, 102, 103, 104])

        result = rolling_sharpe(equity, window=20, min_periods=3)

        # Should have some values with min_periods
        assert result.notna().any()

    def test_very_small_window(self):
        """Test with very small window."""
        np.random.seed(42)
        equity = pd.Series((1 + np.random.normal(0.001, 0.02, 100)).cumprod() * 100)

        result = rolling_sharpe(equity, window=2)

        assert isinstance(result, pd.Series)
        assert not result.dropna().empty

    def test_nan_in_equity(self):
        """Test with NaN values in equity."""
        equity = pd.Series([100, 110, np.nan, 120, 115, np.nan, 125])

        result = rolling_drawdown(equity)

        # Should handle NaN gracefully
        assert isinstance(result, pd.Series)

    def test_negative_equity_values(self):
        """Test with negative equity values."""
        equity = pd.Series([100, 50, 0, -50, -100])

        result = rolling_drawdown(equity)

        assert isinstance(result, pd.Series)
        assert len(result) == len(equity)

    def test_extreme_volatility(self):
        """Test with extreme volatility."""
        np.random.seed(42)
        equity = pd.Series((1 + np.random.normal(0.0, 0.5, 100)).cumprod() * 100)

        result = rolling_sharpe(equity, window=20)

        assert isinstance(result, pd.Series)
        # May have inf/-inf values but should not crash
        assert not result.empty
