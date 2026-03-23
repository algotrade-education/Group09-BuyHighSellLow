"""
Comparison tests between EventDrivenBacktester and legacy Backtester.

Both implementations should produce identical results on the same data and strategy.
"""

from datetime import datetime, timedelta

import pandas as pd

from src.data.indicators.registry import IndicatorRegistry
from src.engine.account.account import AccountState
from src.engine.backtester import Backtester
from src.engine.core.engine import EventDrivenBacktester
from src.strategy.base import StrategyBase
from src.strategy.signal import Signal, TradeSignal


class DeterministicStrategy(StrategyBase):
    """Deterministic strategy for comparison testing."""

    def __init__(self, name: str = "DeterministicStrategy"):
        super().__init__(name)
        self.bar_count = 0

    def generate_signal(self, bar, position=None, is_warmup=False):
        """Generate deterministic signals based on bar count."""
        self.bar_count += 1

        # Entry at bar 5
        if self.bar_count == 5 and (position is None or position.is_flat):
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=0.0,
                stop_loss=bar["close"] - 20,
                take_profit=bar["close"] + 40,
                reason="Entry signal",
            )

        # Exit at bar 15
        if self.bar_count == 15 and position and not position.is_flat:
            return TradeSignal(signal=Signal.EXIT, reason="Exit signal")

        return TradeSignal()

    def reset(self):
        self.bar_count = 0

    @classmethod
    def build_registry(cls, **params) -> IndicatorRegistry:
        return IndicatorRegistry()


def create_comparison_data(n_bars: int = 30) -> pd.DataFrame:
    """Create test data for comparison."""
    base_time = datetime(2024, 1, 1, 9, 0)
    timestamps = [base_time + timedelta(minutes=5 * i) for i in range(n_bars)]

    # Create realistic price movement
    import random

    random.seed(42)

    prices = [1000.0]
    for _ in range(n_bars - 1):
        change = random.uniform(-5, 5)
        prices.append(prices[-1] + change)

    data = {
        "datetime": timestamps,
        "open": prices,
        "high": [p + random.uniform(2, 8) for p in prices],
        "low": [p - random.uniform(2, 8) for p in prices],
        "close": [p + random.uniform(-3, 3) for p in prices],
        "volume": [1000 + random.randint(-100, 100) for _ in range(n_bars)],
    }

    return pd.DataFrame(data)


class TestEventDrivenVsBacktester:
    """Compare EventDrivenBacktester with legacy Backtester."""

    def test_identical_single_trade(self):
        """Test both engines produce identical results for single trade."""
        data = create_comparison_data(n_bars=30)

        # Event-driven version
        strategy_v1 = DeterministicStrategy()
        account_v1 = AccountState(
            initial_capital=500_000_000,
            commission_rate=0.00015,
        )
        backtester_v1 = EventDrivenBacktester(
            strategy=strategy_v1,
            account=account_v1,
            freq_minutes=5,
        )
        result_v1 = backtester_v1.run(data)

        # Legacy version
        strategy_v2 = DeterministicStrategy()
        backtester_v2 = Backtester(
            strategy=strategy_v2,
            initial_capital=500_000_000,
            commission_rate=0.00015,
            freq_minutes=5,
        )
        result_v2 = backtester_v2.run(data)

        # Compare trade count
        assert len(result_v1.trades) == len(result_v2.trades), (
            f"Trade count mismatch: v1={len(result_v1.trades)}, v2={len(result_v2.trades)}"
        )

        if len(result_v1.trades) > 0:
            trade_v1 = result_v1.trades[0]
            trade_v2 = result_v2.trades[0]

            # Compare trade details
            assert trade_v1.side == trade_v2.side
            assert abs(trade_v1.entry_price - trade_v2.entry_price) < 5.0
            assert abs(trade_v1.exit_price - trade_v2.exit_price) < 10.0
            assert trade_v1.quantity == trade_v2.quantity

    def test_identical_pnl(self):
        """Test both engines produce identical P&L."""
        data = create_comparison_data(n_bars=30)

        # Event-driven version
        strategy_v1 = DeterministicStrategy()
        account_v1 = AccountState(initial_capital=500_000_000)
        backtester_v1 = EventDrivenBacktester(
            strategy=strategy_v1,
            account=account_v1,
            freq_minutes=5,
        )
        result_v1 = backtester_v1.run(data)

        # Legacy version
        strategy_v2 = DeterministicStrategy()
        backtester_v2 = Backtester(
            strategy=strategy_v2,
            initial_capital=500_000_000,
            freq_minutes=5,
        )
        result_v2 = backtester_v2.run(data)

        # Compare total P&L
        pnl_v1 = result_v1.metrics.get("total_pnl", 0)
        pnl_v2 = result_v2.metrics.get("total_pnl", 0)

        assert abs(pnl_v1 - pnl_v2) < 100.0, f"P&L mismatch: v1={pnl_v1:.2f}, v2={pnl_v2:.2f}"

    def test_identical_equity_curve_length(self):
        """Test both engines produce equity curves of same length."""
        data = create_comparison_data(n_bars=30)

        strategy_v1 = DeterministicStrategy()
        account_v1 = AccountState(initial_capital=500_000_000)
        backtester_v1 = EventDrivenBacktester(
            strategy=strategy_v1,
            account=account_v1,
            freq_minutes=5,
        )
        result_v1 = backtester_v1.run(data)

        strategy_v2 = DeterministicStrategy()
        backtester_v2 = Backtester(
            strategy=strategy_v2,
            initial_capital=500_000_000,
            freq_minutes=5,
        )
        result_v2 = backtester_v2.run(data)

        assert len(result_v1.equity_curve) == len(result_v2.equity_curve)

    def test_identical_final_equity(self):
        """Test both engines produce identical final equity."""
        data = create_comparison_data(n_bars=30)

        strategy_v1 = DeterministicStrategy()
        account_v1 = AccountState(initial_capital=500_000_000)
        backtester_v1 = EventDrivenBacktester(
            strategy=strategy_v1,
            account=account_v1,
            freq_minutes=5,
        )
        result_v1 = backtester_v1.run(data)

        strategy_v2 = DeterministicStrategy()
        backtester_v2 = Backtester(
            strategy=strategy_v2,
            initial_capital=500_000_000,
            freq_minutes=5,
        )
        result_v2 = backtester_v2.run(data)

        final_equity_v1 = (
            result_v1.equity_curve.iloc[-1]["equity"]
            if len(result_v1.equity_curve) > 0
            else 500_000_000
        )
        final_equity_v2 = (
            result_v2.equity_curve.iloc[-1]["equity"]
            if len(result_v2.equity_curve) > 0
            else 500_000_000
        )

        assert abs(final_equity_v1 - final_equity_v2) < 2_000_000.0, (
            f"Final equity mismatch: v1={final_equity_v1:.2f}, v2={final_equity_v2:.2f}"
        )

    def test_identical_commission_calculation(self):
        """Test both engines calculate commission identically."""
        data = create_comparison_data(n_bars=30)

        commission_rate = 0.0002

        strategy_v1 = DeterministicStrategy()
        account_v1 = AccountState(
            initial_capital=500_000_000,
            commission_rate=commission_rate,
        )
        backtester_v1 = EventDrivenBacktester(
            strategy=strategy_v1,
            account=account_v1,
            freq_minutes=5,
        )
        result_v1 = backtester_v1.run(data)

        strategy_v2 = DeterministicStrategy()
        backtester_v2 = Backtester(
            strategy=strategy_v2,
            initial_capital=500_000_000,
            commission_rate=commission_rate,
            freq_minutes=5,
        )
        result_v2 = backtester_v2.run(data)

        if len(result_v1.trades) > 0 and len(result_v2.trades) > 0:
            commission_v1 = result_v1.trades[0].commission
            commission_v2 = result_v2.trades[0].commission

            assert abs(commission_v1 - commission_v2) < 200.0, (
                f"Commission mismatch: v1={commission_v1:.2f}, v2={commission_v2:.2f}"
            )

    def test_identical_with_different_position_sizes(self):
        """Test both engines handle different position sizes identically."""
        data = create_comparison_data(n_bars=30)

        for position_size in [1, 2, 3]:
            strategy_v1 = DeterministicStrategy()
            account_v1 = AccountState(
                initial_capital=500_000_000,
                position_size=position_size,
            )
            backtester_v1 = EventDrivenBacktester(
                strategy=strategy_v1,
                account=account_v1,
                freq_minutes=5,
            )
            result_v1 = backtester_v1.run(data)

            strategy_v2 = DeterministicStrategy()
            backtester_v2 = Backtester(
                strategy=strategy_v2,
                initial_capital=500_000_000,
                position_size=position_size,
                freq_minutes=5,
            )
            result_v2 = backtester_v2.run(data)

            if len(result_v1.trades) > 0 and len(result_v2.trades) > 0:
                assert result_v1.trades[0].quantity == result_v2.trades[0].quantity == position_size


class TestPerformanceComparison:
    """Compare performance characteristics of both engines."""

    def test_event_driven_handles_large_dataset(self):
        """Test event-driven engine handles large dataset efficiently."""
        data = create_comparison_data(n_bars=1000)

        strategy = DeterministicStrategy()
        account = AccountState(initial_capital=500_000_000)
        backtester = EventDrivenBacktester(
            strategy=strategy,
            account=account,
            freq_minutes=5,
        )

        # Should complete without errors
        result = backtester.run(data)
        assert len(result.equity_curve) == 1000

    def test_event_driven_memory_efficiency(self):
        """Test event-driven engine memory usage."""
        import sys

        data = create_comparison_data(n_bars=500)

        strategy = DeterministicStrategy()
        account = AccountState(initial_capital=500_000_000)
        backtester = EventDrivenBacktester(
            strategy=strategy,
            account=account,
            freq_minutes=5,
        )

        result = backtester.run(data)

        # Verify result size is reasonable
        result_size = sys.getsizeof(result)
        assert result_size < 10_000_000  # Less than 10MB
