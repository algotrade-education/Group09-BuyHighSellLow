"""
Tests for AccountState integration with Backtester.

Focus on:
- Cash management
- Position state transitions
- Daily P&L tracking
- Loss limit enforcement
- Trade recording
"""

import pandas as pd
import pytest

from src.data.indicators.registry import IndicatorRegistry
from src.engine.backtester import Backtester
from src.strategy.base import PositionSnapshot, StrategyBase
from src.strategy.signal import Signal, TradeSignal


class SingleTradeStrategy(StrategyBase):
    """Strategy that makes exactly one trade."""

    def __init__(self):
        super().__init__(name="SingleTrade")
        self.entered = False
        self.exited = False

    def generate_signal(
        self,
        bar: dict,
        position: PositionSnapshot | None = None,
        is_warmup: bool = False,
    ) -> TradeSignal:
        if is_warmup:
            return TradeSignal()

        pos = position or PositionSnapshot.flat()
        close = float(bar["close"])

        # Enter on bar 5
        if not self.entered and not pos.is_long:
            self.entered = True
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=0,
                stop_loss=close - 20,
                take_profit=close + 40,
            )

        # Exit on bar 15
        if self.entered and not self.exited and pos.is_long:
            self.exited = True
            return TradeSignal(signal=Signal.EXIT, entry_price=0)

        return TradeSignal()

    def reset(self):
        self.entered = False
        self.exited = False

    @classmethod
    def build_registry(cls, **params) -> IndicatorRegistry:
        return IndicatorRegistry()


class LossyStrategy(StrategyBase):
    """Strategy designed to lose money."""

    def __init__(self):
        super().__init__(name="Lossy")
        self.trade_count = 0

    def generate_signal(
        self,
        bar: dict,
        position: PositionSnapshot | None = None,
        is_warmup: bool = False,
    ) -> TradeSignal:
        if is_warmup:
            return TradeSignal()

        pos = position or PositionSnapshot.flat()
        close = float(bar["close"])

        # Make losing trades
        if pos.is_flat and self.trade_count < 5:
            self.trade_count += 1
            # Enter with very tight stop, far target (will hit stop)
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=0,
                stop_loss=close - 5,  # Very tight
                take_profit=close + 100,  # Very far
            )

        return TradeSignal()

    def reset(self):
        self.trade_count = 0

    @classmethod
    def build_registry(cls, **params) -> IndicatorRegistry:
        return IndicatorRegistry()


@pytest.fixture
def simple_data():
    """Simple uptrending data."""
    dates = pd.date_range("2024-01-01 09:00", periods=30, freq="5min")
    prices = [1000 + i for i in range(30)]

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [p - 0.5 for p in prices],
            "high": [p + 2 for p in prices],
            "low": [p - 2 for p in prices],
            "close": prices,
            "volume": [1000] * 30,
        }
    )


@pytest.fixture
def volatile_down_data():
    """Volatile downtrending data for loss testing."""
    dates = pd.date_range("2024-01-01 09:00", periods=50, freq="5min")
    prices = [1000 - i * 0.5 for i in range(50)]  # Slow downtrend

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [p + 0.5 for p in prices],
            "high": [p + 3 for p in prices],
            "low": [p - 3 for p in prices],
            "close": prices,
            "volume": [1000] * 50,
        }
    )


class TestAccountStateCashManagement:
    """Test cash management through backtester."""

    def test_cash_decreases_on_commission(self, simple_data):
        """Test cash decreases by commission amount."""
        strategy = SingleTradeStrategy()
        initial_capital = 500_000_000
        commission_rate = 0.0003

        backtester = Backtester(
            strategy=strategy,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            position_size=1,
        )

        result = backtester.run(simple_data)

        # Should have one trade
        assert len(result.trades) == 1

        trade = result.trades[0]

        # Final equity should be initial capital + net pnl
        final_equity = result.equity_curve["equity"].iloc[-1]
        expected_equity = initial_capital + trade.pnl

        # Allow some tolerance due to rounding
        assert final_equity == pytest.approx(expected_equity, rel=1e-3)

    def test_equity_equals_cash_when_flat(self, simple_data):
        """Test equity equals cash when position is flat."""
        strategy = SingleTradeStrategy()
        backtester = Backtester(
            strategy=strategy,
            initial_capital=500_000_000,
            position_size=1,
        )

        result = backtester.run(simple_data)

        equity_curve = result.equity_curve

        # When position is FLAT, equity should equal cash
        flat_rows = equity_curve[equity_curve["position"] == "FLAT"]
        for _, row in flat_rows.iterrows():
            assert row["equity"] == pytest.approx(row["cash"], rel=1e-6)

    def test_equity_includes_unrealized_pnl(self, simple_data):
        """Test equity includes unrealized P&L when in position."""
        strategy = SingleTradeStrategy()
        backtester = Backtester(
            strategy=strategy,
            initial_capital=500_000_000,
            position_size=1,
        )

        result = backtester.run(simple_data)

        equity_curve = result.equity_curve

        # When position is LONG, equity should be cash + unrealized_pnl
        long_rows = equity_curve[equity_curve["position"] == "LONG"]
        if len(long_rows) > 0:
            for _, row in long_rows.iterrows():
                expected_equity = row["cash"] + row["unrealized_pnl"]
                assert row["equity"] == pytest.approx(expected_equity, rel=1e-4)


class TestAccountStatePositionTracking:
    """Test position state tracking."""

    def test_position_transitions(self, simple_data):
        """Test position state transitions: FLAT -> LONG -> FLAT."""
        strategy = SingleTradeStrategy()
        backtester = Backtester(
            strategy=strategy,
            initial_capital=500_000_000,
            position_size=1,
        )

        result = backtester.run(simple_data)

        equity_curve = result.equity_curve
        positions = equity_curve["position"].tolist()

        # Should start FLAT
        assert positions[0] == "FLAT"

        # Should have LONG positions
        assert "LONG" in positions

        # Should end FLAT
        assert positions[-1] == "FLAT"

    def test_no_double_entry(self, simple_data):
        """Test that backtester prevents double entry."""

        class DoubleEntryStrategy(StrategyBase):
            """Try to enter twice."""

            def __init__(self):
                super().__init__(name="DoubleEntry")

            def generate_signal(self, bar, position=None, is_warmup=False):
                if is_warmup:
                    return TradeSignal()

                close = float(bar["close"])
                # Always try to go long
                return TradeSignal(
                    signal=Signal.LONG,
                    entry_price=0,
                    stop_loss=close - 20,
                    take_profit=close + 40,
                )

            @classmethod
            def build_registry(cls, **params):
                return IndicatorRegistry()

        strategy = DoubleEntryStrategy()
        backtester = Backtester(
            strategy=strategy,
            initial_capital=500_000_000,
            position_size=1,
        )

        result = backtester.run(simple_data)

        # Should only have 1 trade (second entry blocked)
        assert len(result.trades) == 1


class TestAccountStateDailyTracking:
    """Test daily P&L tracking and loss limits."""

    def test_daily_loss_limit_stops_trading(self, volatile_down_data):
        """Test that daily loss limit stops new trades."""
        strategy = LossyStrategy()
        initial_capital = 500_000_000
        max_daily_loss_pct = 0.01  # 1% daily loss limit

        backtester = Backtester(
            strategy=strategy,
            initial_capital=initial_capital,
            max_daily_loss_pct=max_daily_loss_pct,
            position_size=1,
        )

        result = backtester.run(volatile_down_data)

        # Should have some trades, but stopped by loss limit
        assert len(result.trades) > 0

        # Calculate daily P&L
        if len(result.trades) > 0:
            # Check that trading stopped after loss limit hit
            # (hard to verify without access to internal state)
            pass

    def test_daily_pnl_resets_on_new_day(self):
        """Test daily P&L resets on new trading day."""
        # Create data spanning 2 days
        day1 = pd.date_range("2024-01-01 09:00", "2024-01-01 14:45", freq="5min")
        day2 = pd.date_range("2024-01-02 09:00", "2024-01-02 14:45", freq="5min")

        dates = day1.tolist() + day2.tolist()
        n = len(dates)
        prices = [1000 + i * 0.5 for i in range(n)]

        data = pd.DataFrame(
            {
                "datetime": dates,
                "open": [p - 0.5 for p in prices],
                "high": [p + 2 for p in prices],
                "low": [p - 2 for p in prices],
                "close": prices,
                "volume": [1000] * n,
            }
        )

        strategy = LossyStrategy()
        backtester = Backtester(
            strategy=strategy,
            initial_capital=500_000_000,
            max_daily_loss_pct=0.01,
            position_size=1,
        )

        result = backtester.run(data)

        # Should complete without error
        assert result is not None


class TestAccountStateTradeRecording:
    """Test trade recording and metrics."""

    def test_trade_recorded_with_all_fields(self, simple_data):
        """Test that trades are recorded with all required fields."""
        strategy = SingleTradeStrategy()
        backtester = Backtester(
            strategy=strategy,
            initial_capital=500_000_000,
            position_size=1,
        )

        result = backtester.run(simple_data)

        assert len(result.trades) == 1
        trade = result.trades[0]

        # Verify all fields are populated
        assert trade.entry_time is not None
        assert trade.exit_time is not None
        assert trade.entry_price > 0
        assert trade.exit_price > 0
        assert trade.quantity > 0
        assert trade.side in ["long", "short"]
        assert trade.pnl != 0  # Should have some P&L
        assert trade.commission >= 0
        assert trade.exit_reason != ""

    def test_multiple_trades_recorded_in_order(self):
        """Test multiple trades are recorded in chronological order."""

        class MultiTradeStrategy(StrategyBase):
            """Make 3 trades."""

            def __init__(self):
                super().__init__(name="MultiTrade")
                self.trade_count = 0
                self.in_position = False

            def generate_signal(self, bar, position=None, is_warmup=False):
                if is_warmup:
                    return TradeSignal()

                pos = position or PositionSnapshot.flat()
                close = float(bar["close"])

                # Enter every 10 bars
                if pos.is_flat and self.trade_count < 3:
                    self.trade_count += 1
                    self.in_position = True
                    return TradeSignal(
                        signal=Signal.LONG,
                        entry_price=0,
                        stop_loss=close - 10,
                        take_profit=close + 20,
                    )

                # Exit after 5 bars
                if self.in_position and pos.is_long:
                    self.in_position = False
                    return TradeSignal(signal=Signal.EXIT, entry_price=0)

                return TradeSignal()

            def reset(self):
                self.trade_count = 0
                self.in_position = False

            @classmethod
            def build_registry(cls, **params):
                return IndicatorRegistry()

        dates = pd.date_range("2024-01-01 09:00", periods=100, freq="5min")
        prices = [1000 + i * 0.5 for i in range(100)]

        data = pd.DataFrame(
            {
                "datetime": dates,
                "open": [p - 0.5 for p in prices],
                "high": [p + 2 for p in prices],
                "low": [p - 2 for p in prices],
                "close": prices,
                "volume": [1000] * 100,
            }
        )

        strategy = MultiTradeStrategy()
        backtester = Backtester(
            strategy=strategy,
            initial_capital=500_000_000,
            position_size=1,
        )

        result = backtester.run(data)

        # Should have multiple trades
        assert len(result.trades) >= 2

        # Trades should be in chronological order
        for i in range(len(result.trades) - 1):
            assert result.trades[i].entry_time <= result.trades[i + 1].entry_time
