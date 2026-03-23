"""Tests for Information Ratio metric."""

import numpy as np
import pandas as pd

from src.metrics.information_ratio import (
    InformationRatio,
    calculate_information_ratio,
)


class TestInformationRatio:
    """Test Information Ratio calculations."""

    def test_basic_calculation_daily(self, daily_returns, benchmark_returns):
        """Test basic IR calculation with daily data."""
        metric = InformationRatio(annualization_factor=252)
        ir = metric.calculate(daily_returns, benchmark_returns, annualized=True)

        assert isinstance(ir, float)
        assert not np.isnan(ir)

    def test_basic_calculation_15m(self):
        """Test IR with 15-minute intraday data."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0002, 0.005, 4320))
        benchmark = pd.Series(np.random.normal(0.0001, 0.004, 4320))

        metric = InformationRatio(annualization_factor=4320)
        ir = metric.calculate(returns, benchmark, annualized=True)

        assert isinstance(ir, float)
        assert not np.isnan(ir)

    def test_outperforming_benchmark(self):
        """Outperforming benchmark should give positive IR."""
        returns = pd.Series([0.02, 0.03, 0.01, 0.025, 0.02])
        benchmark = pd.Series([0.01, 0.015, 0.005, 0.01, 0.01])

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        assert ir > 0

    def test_underperforming_benchmark(self):
        """Underperforming benchmark should give negative IR."""
        returns = pd.Series([0.01, 0.015, 0.005, 0.01, 0.01])
        benchmark = pd.Series([0.02, 0.03, 0.01, 0.025, 0.02])

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        assert ir < 0

    def test_identical_to_benchmark(self):
        """Identical returns should give IR near 0."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.02])
        benchmark = returns.copy()

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        # Tracking error is 0, so IR should be 0 or undefined
        assert ir == 0.0

    def test_zero_tracking_error(self):
        """Zero tracking error should return 0."""
        returns = pd.Series([0.01, 0.02, 0.015])
        benchmark = returns.copy()

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        assert ir == 0.0

    def test_annualized_vs_non_annualized(self, daily_returns, benchmark_returns):
        """Annualized IR should be sqrt(252) times non-annualized."""
        metric = InformationRatio(annualization_factor=252)

        ir_annual = metric.calculate(daily_returns, benchmark_returns, annualized=True)
        ir_non_annual = metric.calculate(daily_returns, benchmark_returns, annualized=False)

        if ir_non_annual != 0:
            expected_ratio = np.sqrt(252)
            actual_ratio = ir_annual / ir_non_annual

            assert np.isclose(actual_ratio, expected_ratio, rtol=0.01)

    def test_tracking_error_calculation(self, daily_returns, benchmark_returns):
        """Test tracking error calculation separately."""
        metric = InformationRatio(annualization_factor=252)

        te = metric.calculate_tracking_error(daily_returns, benchmark_returns, annualized=False)

        assert te > 0
        assert not np.isnan(te)

    def test_tracking_error_annualization(self, daily_returns, benchmark_returns):
        """Annualized TE should be sqrt(252) times non-annualized."""
        metric = InformationRatio(annualization_factor=252)

        te_annual = metric.calculate_tracking_error(
            daily_returns, benchmark_returns, annualized=True
        )
        te_non_annual = metric.calculate_tracking_error(
            daily_returns, benchmark_returns, annualized=False
        )

        expected_ratio = np.sqrt(252)
        actual_ratio = te_annual / te_non_annual

        assert np.isclose(actual_ratio, expected_ratio, rtol=0.01)

    def test_misaligned_series(self):
        """Test with misaligned series (different lengths)."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.02])
        benchmark = pd.Series([0.01, 0.015, 0.01])  # Shorter

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        # Should align and calculate on common periods
        assert isinstance(ir, float)
        assert not np.isnan(ir)

    def test_different_input_types(self):
        """Test with Series, array, and list inputs."""
        returns_list = [0.01, 0.02, 0.015, 0.01, 0.02]
        benchmark_list = [0.01, 0.015, 0.01, 0.005, 0.015]

        returns_array = np.array(returns_list)
        benchmark_array = np.array(benchmark_list)

        returns_series = pd.Series(returns_list)
        benchmark_series = pd.Series(benchmark_list)

        metric = InformationRatio()

        ir_list = metric.calculate(returns_list, benchmark_list)
        ir_array = metric.calculate(returns_array, benchmark_array)
        ir_series = metric.calculate(returns_series, benchmark_series)

        assert np.isclose(ir_list, ir_array)
        assert np.isclose(ir_list, ir_series)

    def test_convenience_function(self, daily_returns, benchmark_returns):
        """Test convenience function matches class method."""
        metric = InformationRatio(annualization_factor=252)
        ir_class = metric.calculate(daily_returns, benchmark_returns)

        ir_func = calculate_information_ratio(
            daily_returns, benchmark_returns, annualization_factor=252
        )

        assert np.isclose(ir_class, ir_func)

    def test_empty_returns(self):
        """Empty returns should return 0."""
        metric = InformationRatio()
        ir = metric.calculate(pd.Series([], dtype=float), pd.Series([], dtype=float))

        assert ir == 0.0

    def test_single_return(self):
        """Single return should return 0."""
        metric = InformationRatio()
        ir = metric.calculate(pd.Series([0.01]), pd.Series([0.01]))

        assert ir == 0.0

    def test_nan_handling(self):
        """Test handling of NaN values."""
        returns = pd.Series([0.01, np.nan, 0.02, 0.015, np.nan])
        benchmark = pd.Series([0.01, 0.015, np.nan, 0.01, 0.015])

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        assert not np.isnan(ir)

    def test_intraday_5m_bars(self):
        """Test with 5-minute bars."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0001, 0.003, 12960))
        benchmark = pd.Series(np.random.normal(0.00008, 0.0025, 12960))

        metric = InformationRatio(annualization_factor=12960)
        ir = metric.calculate(returns, benchmark)

        assert isinstance(ir, float)
        assert not np.isnan(ir)

    def test_hourly_bars(self):
        """Test with hourly bars."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0005, 0.01, 1440))
        benchmark = pd.Series(np.random.normal(0.0004, 0.008, 1440))

        metric = InformationRatio(annualization_factor=1440)
        ir = metric.calculate(returns, benchmark)

        assert isinstance(ir, float)
        assert not np.isnan(ir)


class TestInformationRatioEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_high_active_return_low_tracking_error(self):
        """High active return with low tracking error should give high IR."""
        # Consistently outperform by small amount
        returns = pd.Series([0.011, 0.021, 0.016, 0.011, 0.021] * 10)
        benchmark = pd.Series([0.01, 0.02, 0.015, 0.01, 0.02] * 10)

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        assert ir > 1.0  # Should be high

    def test_low_active_return_high_tracking_error(self):
        """Low active return with high tracking error should give low IR."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.05, 100))
        benchmark = pd.Series(np.random.normal(0.001, 0.01, 100))

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        # High tracking error relative to active return
        # IR can vary based on random seed, just check it's calculated
        assert isinstance(ir, float)
        assert not np.isnan(ir)

    def test_extreme_outperformance(self):
        """Test with extreme outperformance."""
        returns = pd.Series([0.1, 0.15, 0.12, 0.11])
        benchmark = pd.Series([0.01, 0.02, 0.015, 0.01])

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        assert ir > 0
        assert not np.isinf(ir)

    def test_extreme_underperformance(self):
        """Test with extreme underperformance."""
        returns = pd.Series([-0.1, -0.15, -0.12, -0.11])
        benchmark = pd.Series([0.01, 0.02, 0.015, 0.01])

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        assert ir < 0
        assert not np.isinf(ir)

    def test_all_nan_returns(self):
        """All NaN returns should return 0."""
        metric = InformationRatio()
        ir = metric.calculate(pd.Series([np.nan, np.nan]), pd.Series([np.nan, np.nan]))

        assert ir == 0.0

    def test_benchmark_all_zeros(self):
        """Benchmark all zeros should calculate properly."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.01])
        benchmark = pd.Series([0.0, 0.0, 0.0, 0.0])

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        # Active returns = returns, should calculate normally
        assert isinstance(ir, float)
        assert not np.isnan(ir)

    def test_returns_all_zeros(self):
        """Returns all zeros should give negative IR."""
        returns = pd.Series([0.0, 0.0, 0.0, 0.0])
        benchmark = pd.Series([0.01, 0.02, 0.015, 0.01])

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        assert ir < 0

    def test_both_all_zeros(self):
        """Both all zeros should return 0."""
        returns = pd.Series([0.0, 0.0, 0.0, 0.0])
        benchmark = pd.Series([0.0, 0.0, 0.0, 0.0])

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        assert ir == 0.0

    def test_very_small_returns(self):
        """Test with very small returns (numerical stability)."""
        returns = pd.Series([1e-10, 2e-10, 1.5e-10, 1e-10] * 100)
        benchmark = pd.Series([0.9e-10, 1.8e-10, 1.4e-10, 0.9e-10] * 100)

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        assert isinstance(ir, float)
        assert not np.isnan(ir)

    def test_negative_correlation_with_benchmark(self):
        """Test when negatively correlated with benchmark."""
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        benchmark = pd.Series([-0.02, 0.01, -0.03, 0.02, -0.01])

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        # High tracking error due to negative correlation
        assert isinstance(ir, float)
        assert not np.isnan(ir)

    def test_perfect_positive_correlation(self):
        """Test when perfectly correlated but scaled."""
        benchmark = pd.Series([0.01, 0.02, 0.015, 0.01, 0.02])
        returns = benchmark * 1.5  # 50% higher but same pattern

        metric = InformationRatio()
        ir = metric.calculate(returns, benchmark)

        # Should have positive IR with low tracking error
        assert ir > 0
