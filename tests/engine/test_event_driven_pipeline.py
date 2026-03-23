"""
Integration tests for event-driven pipeline flow.

Tests the complete event flow:
    MarketEvent → Strategy → SignalEvent → Risk → OrderEvent → Broker → FillEvent → Account
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.data.indicators.registry import IndicatorRegistry
from src.engine.account.account import AccountState
from src.engine.core.engine import EventDrivenBacktester
from src.strategy.base import StrategyBase
from src.strategy.signal import Signal, TradeSignal


class SimpleTestStrategy(StrategyBase):
    """Simple strategy for testing pipeline."""

    def __init__(self, signals: list[tuple[int, str]]):
        """
        Args:
            signals: List of (bar_index, signal_type) tuples
                    e.g., [(0, "long"), (5, "exit")]
        """
        super().__init__("TestStrategy")
        self.signals = {idx: sig for idx, sig in signals}
        self.bar_count = 0

    def generate_signal(self, bar, position=None, is_warmup=False):
        signal_type = self.signals.get(self.bar_count, "hold")
        self.bar_count += 1

        if signal_type == "hold":
            return TradeSignal()
        elif signal_type == "long":
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=0.0,  # Market order
                stop_loss=bar["close"] - 10,
                take_profit=bar["close"] + 20,
                reason="Test long",
            )
        elif signal_type == "short":
            return TradeSignal(
                signal=Signal.SHORT,
                entry_price=0.0,
                stop_loss=bar["close"] + 10,
                take_profit=bar["close"] - 20,
                reason="Test short",
            )
        elif signal_type == "exit":
            return TradeSignal(signal=Signal.EXIT, reason="Test exit")
        else:
            return TradeSignal()

    def reset(self):
        self.bar_count = 0

    @classmethod
    def build_registry(cls, **params) -> IndicatorRegistry:
        return IndicatorRegistry()


def create_test_data(n_bars: int = 10, start_price: float = 1000.0) -> pd.DataFrame:
    """Create simple test data."""
    timestamps = [datetime(2024, 1, 1, 9, 0) + timedelta(minutes=5 * i) for i in range(n_bars)]

    data = {
        "datetime": timestamps,
        "open": [start_price + i for i in range(n_bars)],
        "high": [start_price + i + 5 for i in range(n_bars)],
        "low": [start_price + i - 5 for i in range(n_bars)],
        "close": [start_price + i + 2 for i in range(n_bars)],
        "volume": [1000] * n_bars,
    }

    return pd.DataFrame(data)


class TestEventDrivenPipeline:
    """Test complete event-driven pipeline."""

    def test_simple_long_trade(self):
        """Test simple long entry and exit."""
        # Strategy: long at bar 0, exit at bar 5
        strategy = SimpleTestStrategy([(0, "long"), (5, "exit")])
        account = AccountState(initial_capital=500_000_000)

        backtester = EventDrivenBacktester(
            strategy=strategy,
            account=account,
            freq_minutes=5,
        )

        data = create_test_data(n_bars=10, start_price=1000.0)
        result = backtester.run(data)

        # Verify trade executed
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.side == "long"
        assert trade.entry_price == pytest.approx(data.iloc[0]["open"], abs=1.0)
        assert trade.exit_price == pytest.approx(data.iloc[5]["close"], abs=2.0)

    def test_simple_short_trade(self):
        """Test simple short entry and exit."""
        strategy = SimpleTestStrategy([(0, "short"), (5, "exit")])
        account = AccountState(initial_capital=500_000_000)

        backtester = EventDrivenBacktester(
            strategy=strategy,
            account=account,
            freq_minutes=5,
        )

        data = create_test_data(n_bars=10, start_price=1000.0)
        result = backtester.run(data)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.side == "short"
        assert trade.entry_price == pytest.approx(data.iloc[0]["open"], abs=1.0)
        assert trade.exit_price == pytest.approx(data.iloc[5]["close"], abs=2.0)

    def test_multiple_trades(self):
        """Test multiple sequential trades."""
        strategy = SimpleTestStrategy(
            [
                (0, "long"),
                (3, "exit"),
                (5, "short"),
                (8, "exit"),
            ]
        )
        account = AccountState(initial_capital=500_000_000)

        backtester = EventDrivenBacktester(
            strategy=strategy,
            account=account,
            freq_minutes=5,
        )

        data = create_test_data(n_bars=10, start_price=1000.0)
        result = backtester.run(data)

        # Should have 2 trades
        assert len(result.trades) == 2
        assert result.trades[0].side == "long"
        assert result.trades[1].side == "short"

    def test_no_double_entry(self):
        """Test risk handler prevents double entry."""
        # Try to enter long twice
        strategy = SimpleTestStrategy(
            [
                (0, "long"),
                (2, "long"),  # Should be blocked
                (5, "exit"),
            ]
        )
        account = AccountState(initial_capital=500_000_000)

        backtester = EventDrivenBacktester(
            strategy=strategy,
            account=account,
            freq_minutes=5,
        )

        data = create_test_data(n_bars=10, start_price=1000.0)
        result = backtester.run(data)

        # Should only have 1 trade (second entry blocked)
        assert len(result.trades) == 1

    def test_stop_loss_triggered(self):
        """Test stop loss is triggered."""
        strategy = SimpleTestStrategy([(0, "long")])
        account = AccountState(initial_capital=500_000_000)

        backtester = EventDrivenBacktester(
            strategy=strategy,
            account=account,
            freq_minutes=5,
        )

        # Create data where price drops below stop loss
        data = create_test_data(n_bars=10, start_price=1000.0)
        # Make bar 3 drop below stop loss (stop at 992, low at 985)
        data.loc[3, "low"] = 985.0
        data.loc[3, "close"] = 990.0

        result = backtester.run(data)

        # Trade should be closed by stop loss
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert "SL" in trade.exit_reason or "Stop" in trade.exit_reason

    def test_take_profit_triggered(self):
        """Test take profit is triggered."""
        strategy = SimpleTestStrategy([(0, "long")])
        account = AccountState(initial_capital=500_000_000)

        backtester = EventDrivenBacktester(
            strategy=strategy,
            account=account,
            freq_minutes=5,
        )

        # Create data where price reaches take profit
        data = create_test_data(n_bars=10, start_price=1000.0)
        # Make bar 3 reach take profit (TP at 1022, high at 1025)
        data.loc[3, "high"] = 1025.0
        data.loc[3, "close"] = 1023.0

        result = backtester.run(data)

        # Trade should be closed by take profit
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert "TP" in trade.exit_reason or "Take" in trade.exit_reason

    def test_equity_curve_tracking(self):
        """Test equity curve is tracked correctly."""
        strategy = SimpleTestStrategy([(0, "long"), (5, "exit")])
        account = AccountState(initial_capital=500_000_000)

        backtester = EventDrivenBacktester(
            strategy=strategy,
            account=account,
            freq_minutes=5,
        )

        data = create_test_data(n_bars=10, start_price=1000.0)
        result = backtester.run(data)

        # Verify equity curve has correct length
        assert len(result.equity_curve) == 10

        # Verify tracked equity values are valid
        assert "equity" in result.equity_curve.columns
        assert result.equity_curve["equity"].notna().all()

    def test_commission_deducted(self):
        """Test commission is deducted from cash."""
        strategy = SimpleTestStrategy([(0, "long"), (5, "exit")])
        account = AccountState(
            initial_capital=500_000_000,
            commission_rate=0.0001,  # 0.01%
        )

        backtester = EventDrivenBacktester(
            strategy=strategy,
            account=account,
            freq_minutes=5,
        )

        data = create_test_data(n_bars=10, start_price=1000.0)
        result = backtester.run(data)

        # Verify commission was charged
        trade = result.trades[0]
        assert trade.commission > 0

    def test_empty_data(self):
        """Test handling of empty data."""
        strategy = SimpleTestStrategy([])
        account = AccountState(initial_capital=500_000_000)

        backtester = EventDrivenBacktester(
            strategy=strategy,
            account=account,
            freq_minutes=5,
        )

        data = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        result = backtester.run(data)

        assert len(result.trades) == 0
        assert len(result.equity_curve) == 0

    def test_progress_callback(self):
        """Test progress callback is called."""
        strategy = SimpleTestStrategy([])
        account = AccountState(initial_capital=500_000_000)

        backtester = EventDrivenBacktester(
            strategy=strategy,
            account=account,
            freq_minutes=5,
        )

        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        data = create_test_data(n_bars=5)
        _ = backtester.run(data, progress_callback=callback)

        # Verify callback was called for each bar
        assert len(progress_calls) == 5
        assert progress_calls[-1] == (5, 5)
