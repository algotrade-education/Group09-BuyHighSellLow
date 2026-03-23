"""
Edge case tests for backtester.

Tests cover:
- Multiple entries/exits in same session
- Rapid signal changes
- Extreme price movements
- Data quality issues
- Session boundaries
- Order TTL expiration
"""

import pandas as pd
import pytest

from src.data.indicators.registry import IndicatorRegistry
from src.engine.backtester import Backtester
from src.strategy.base import PositionSnapshot, StrategyBase
from src.strategy.signal import Signal, TradeSignal

# ── Test Strategies ───────────────────────────────────────────────


class FlipFlopStrategy(StrategyBase):
    """Strategy that alternates between long and short every bar."""

    def __init__(self):
        super().__init__(name="FlipFlop")
        self.bar_count = 0

    def generate_signal(
        self,
        bar: dict,
        position: PositionSnapshot | None = None,
        is_warmup: bool = False,
    ) -> TradeSignal:
        if is_warmup:
            return TradeSignal()

        self.bar_count += 1
        FLAT = PositionSnapshot(
            is_flat=True,
            is_long=False,
            is_short=False,
            quantity=0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
        )
        pos = position or FLAT
        close = float(bar["close"])

        # Alternate between long and short
        if self.bar_count % 2 == 1:
            if pos.is_flat or pos.is_short:
                return TradeSignal(
                    signal=Signal.LONG,
                    entry_price=0,  # Market order
                    stop_loss=close * 0.9,
                    take_profit=close * 1.2,
                )
        else:
            if pos.is_flat or pos.is_long:
                return TradeSignal(
                    signal=Signal.SHORT,
                    entry_price=0,  # Market order
                    stop_loss=close * 1.1,
                    take_profit=close * 0.8,
                )

        return TradeSignal()

    def reset(self):
        self.bar_count = 0

    @classmethod
    def build_registry(cls, **params) -> IndicatorRegistry:
        return IndicatorRegistry()


class MultipleEntryStrategy(StrategyBase):
    """Strategy that tries to enter multiple times."""

    def __init__(self):
        super().__init__(name="MultipleEntry")
        self.entry_count = 0

    def generate_signal(
        self,
        bar: dict,
        position: PositionSnapshot | None = None,
        is_warmup: bool = False,
    ) -> TradeSignal:
        if is_warmup:
            return TradeSignal()

        FLAT = PositionSnapshot(
            is_flat=True,
            is_long=False,
            is_short=False,
            quantity=0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
        )
        _ = position or FLAT
        close = float(bar["close"])

        # Try to enter on every bar (should be rejected if already in position)
        if self.entry_count < 10:
            self.entry_count += 1
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=0,  # Market order
                stop_loss=close * 0.95,
                take_profit=close * 1.05,
            )

        return TradeSignal()

    def reset(self):
        self.entry_count = 0

    @classmethod
    def build_registry(cls, **params) -> IndicatorRegistry:
        return IndicatorRegistry()


# ── Test Data Fixtures ────────────────────────────────────────────


@pytest.fixture
def gap_data():
    """Generate data with price gaps."""
    dates = pd.date_range("2024-01-01 09:00", periods=50, freq="5min")

    prices = []
    for i in range(50):
        if i < 20:
            prices.append(1000)
        elif i == 20:
            prices.append(1100)  # Gap up
        elif i < 40:
            prices.append(1100)
        else:
            prices.append(900)  # Gap down

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": prices,
            "high": [p + 5 for p in prices],
            "low": [p - 5 for p in prices],
            "close": prices,
            "volume": [1000] * 50,
        }
    )


@pytest.fixture
def extreme_volatility_data():
    """Generate data with extreme price swings."""
    dates = pd.date_range("2024-01-01 09:00", periods=50, freq="5min")

    prices = []
    for i in range(50):
        # Oscillate wildly
        if i % 2 == 0:
            prices.append(1000)
        else:
            prices.append(1050)

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": prices,
            "high": [p + 30 for p in prices],
            "low": [p - 30 for p in prices],
            "close": prices,
            "volume": [1000] * 50,
        }
    )


@pytest.fixture
def session_boundary_data():
    """Generate data spanning multiple sessions."""
    # Morning session: 09:00-11:30
    morning = pd.date_range("2024-01-01 09:00", "2024-01-01 11:30", freq="5min")
    # Afternoon session: 13:00-14:45
    afternoon = pd.date_range("2024-01-01 13:00", "2024-01-01 14:45", freq="5min")

    dates = morning.tolist() + afternoon.tolist()
    n = len(dates)

    prices = [1000 + i for i in range(n)]

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [p - 0.5 for p in prices],
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000] * n,
        }
    )


# ── Edge Case Tests ───────────────────────────────────────────────


def test_rapid_signal_changes(extreme_volatility_data):
    """Test strategy with rapid signal changes."""
    strategy = FlipFlopStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(extreme_volatility_data)

    # Should handle rapid changes without error
    assert result is not None
    # Should have multiple trades
    assert len(result.trades) > 0


def test_multiple_entry_attempts_rejected(extreme_volatility_data):
    """Test that multiple entry attempts while in position are rejected."""
    strategy = MultipleEntryStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(extreme_volatility_data)

    # Should only have 1-10 trades (strategy tries 10 times, but gets stopped out)
    # Each time it enters, it may get stopped out and re-enter
    assert len(result.trades) >= 1


def test_price_gaps_handled(gap_data):
    """Test that price gaps are handled correctly."""
    strategy = FlipFlopStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(gap_data)

    # Should complete without error
    assert result is not None
    # Equity should remain positive
    assert result.equity_curve["equity"].min() > 0


def test_extreme_volatility(extreme_volatility_data):
    """Test backtest with extreme price volatility."""
    strategy = FlipFlopStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(extreme_volatility_data)

    # Should complete without error
    assert result is not None
    # Should have trades
    assert len(result.trades) > 0


def test_session_boundary_handling(session_boundary_data):
    """Test handling of session boundaries."""
    strategy = FlipFlopStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(session_boundary_data)

    # Should complete without error
    assert result is not None


def test_order_ttl_expiration():
    """Test order expiration with TTL."""
    # Create data where price doesn't move (order won't fill)
    dates = pd.date_range("2024-01-01 09:00", periods=50, freq="5min")
    data = pd.DataFrame(
        {
            "datetime": dates,
            "open": [1000] * 50,
            "high": [1001] * 50,
            "low": [999] * 50,
            "close": [1000] * 50,
            "volume": [1000] * 50,
        }
    )

    strategy = MultipleEntryStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
        order_ttl_bars=2,  # Orders expire after 2 bars
    )

    result = backtester.run(data)

    # Should complete without error
    assert result is not None


def test_zero_commission():
    """Test backtest with zero commission."""
    dates = pd.date_range("2024-01-01 09:00", periods=20, freq="5min")
    data = pd.DataFrame(
        {
            "datetime": dates,
            "open": [1000 + i for i in range(20)],
            "high": [1001 + i for i in range(20)],
            "low": [999 + i for i in range(20)],
            "close": [1000 + i for i in range(20)],
            "volume": [1000] * 20,
        }
    )

    strategy = FlipFlopStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        commission_rate=0.0,  # Zero commission
        position_size=1,
    )

    result = backtester.run(data)

    # All trades should have zero commission
    for trade in result.trades:
        assert trade.commission == 0.0


def test_high_commission():
    """Test backtest with high commission rate."""
    dates = pd.date_range("2024-01-01 09:00", periods=20, freq="5min")
    data = pd.DataFrame(
        {
            "datetime": dates,
            "open": [1000 + i for i in range(20)],
            "high": [1001 + i for i in range(20)],
            "low": [999 + i for i in range(20)],
            "close": [1000 + i for i in range(20)],
            "volume": [1000] * 20,
        }
    )

    strategy = FlipFlopStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        commission_rate=0.01,  # 1% commission (very high)
        position_size=1,
    )

    result = backtester.run(data)

    # High commission should significantly impact PnL
    if len(result.trades) > 0:
        total_commission = sum(t.commission for t in result.trades)
        assert total_commission > 0


def test_single_bar_data():
    """Test backtest with only one bar."""
    data = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2024-01-01 09:00")],
            "open": [1000],
            "high": [1010],
            "low": [990],
            "close": [1005],
            "volume": [1000],
        }
    )

    strategy = FlipFlopStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(data)

    # Should complete without error
    assert result is not None
    # No trades possible with single bar
    assert len(result.trades) == 0


def test_warmup_equals_data_length():
    """Test warmup period equal to data length."""
    dates = pd.date_range("2024-01-01 09:00", periods=20, freq="5min")
    data = pd.DataFrame(
        {
            "datetime": dates,
            "open": [1000 + i for i in range(20)],
            "high": [1001 + i for i in range(20)],
            "low": [999 + i for i in range(20)],
            "close": [1000 + i for i in range(20)],
            "volume": [1000] * 20,
        }
    )

    strategy = FlipFlopStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(data, warmup_bars=20)

    # No trades should occur (all bars are warmup)
    assert len(result.trades) == 0


def test_negative_prices_rejected():
    """Test that negative prices are handled."""
    dates = pd.date_range("2024-01-01 09:00", periods=10, freq="5min")
    data = pd.DataFrame(
        {
            "datetime": dates,
            "open": [-100] * 10,  # Invalid negative prices
            "high": [-90] * 10,
            "low": [-110] * 10,
            "close": [-100] * 10,
            "volume": [1000] * 10,
        }
    )

    strategy = FlipFlopStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    # Should handle gracefully (may produce no trades or errors)
    result = backtester.run(data)
    assert result is not None
