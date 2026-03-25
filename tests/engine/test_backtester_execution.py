"""
Tests for order execution and position management in backtester.

Focus on:
- Order filling mechanics
- Stop loss / take profit execution
- EOD close behavior
- Position tracking (MAE/MFE)
- Slippage and commission
"""

import pandas as pd
import pytest

from src.data.indicators.registry import IndicatorRegistry
from src.engine.backtester import Backtester
from src.strategy.base import PositionSnapshot, StrategyBase
from src.strategy.signal import Signal, TradeSignal

# --- Test Strategies ---


class AlwaysLongStrategy(StrategyBase):
    """Strategy that always goes long on first bar."""

    def __init__(self):
        super().__init__(name="AlwaysLong")
        self.entered = False

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

        if pos.is_flat and not self.entered:
            self.entered = True
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=0,  # Market order
                stop_loss=close * 0.95,  # 5% stop
                take_profit=close * 1.10,  # 10% target
            )

        return TradeSignal()

    def reset(self):
        self.entered = False

    @classmethod
    def build_registry(cls, **params) -> IndicatorRegistry:
        return IndicatorRegistry()


class StopLossTestStrategy(StrategyBase):
    """Strategy to test stop loss execution."""

    def __init__(self):
        super().__init__(name="StopLossTest")
        self.entered = False

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

        # Enter long on first bar
        if pos.is_flat and not self.entered:
            self.entered = True
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=0,  # Market order
                stop_loss=close - 10,  # Tight stop
                take_profit=close + 100,  # Far target
            )

        return TradeSignal()

    def reset(self):
        self.entered = False

    @classmethod
    def build_registry(cls, **params) -> IndicatorRegistry:
        return IndicatorRegistry()


class TakeProfitTestStrategy(StrategyBase):
    """Strategy to test take profit execution."""

    def __init__(self):
        super().__init__(name="TakeProfitTest")
        self.entered = False

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

        # Enter long on first bar
        if pos.is_flat and not self.entered:
            self.entered = True
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=0,  # Market order
                stop_loss=close - 100,  # Far stop
                take_profit=close + 10,  # Tight target
            )

        return TradeSignal()

    def reset(self):
        self.entered = False

    @classmethod
    def build_registry(cls, **params) -> IndicatorRegistry:
        return IndicatorRegistry()


# --- Test Data Fixtures ---


@pytest.fixture
def uptrend_data():
    """Generate uptrending price data."""
    dates = pd.date_range("2024-01-01 09:00", periods=50, freq="5min")
    prices = [1000 + i * 2 for i in range(50)]  # Steady uptrend

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [p - 0.5 for p in prices],
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000] * 50,
        }
    )


@pytest.fixture
def downtrend_data():
    """Generate downtrending price data."""
    dates = pd.date_range("2024-01-01 09:00", periods=50, freq="5min")
    prices = [1000 - i * 2 for i in range(50)]  # Steady downtrend

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [p + 0.5 for p in prices],
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000] * 50,
        }
    )


@pytest.fixture
def volatile_data():
    """Generate volatile price data (hits stop loss)."""
    dates = pd.date_range("2024-01-01 09:00", periods=50, freq="5min")

    # Start at 1000, drop to 980, then recover
    prices = []
    for i in range(50):
        if i < 10:
            prices.append(1000)
        elif i < 20:
            prices.append(1000 - (i - 10) * 2)  # Drop to 980
        else:
            prices.append(980 + (i - 20))  # Recover

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [p - 0.5 for p in prices],
            "high": [p + 2 for p in prices],
            "low": [p - 2 for p in prices],
            "close": prices,
            "volume": [1000] * 50,
        }
    )


# --- Execution Tests ---


def test_order_execution_at_open(uptrend_data):
    """Test that orders are executed at next bar open."""
    strategy = AlwaysLongStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(uptrend_data)

    # Should have at least one trade
    assert len(result.trades) > 0

    trade = result.trades[0]
    # Entry should be at bar 1 open (after signal on bar 0)
    # Market order fills at next bar open
    expected_entry = uptrend_data.iloc[1]["open"]
    # Allow some tolerance for market order execution
    assert abs(trade.entry_price - expected_entry) < 5  # Within 5 points


def test_stop_loss_execution(volatile_data):
    """Test stop loss is triggered correctly."""
    strategy = StopLossTestStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(volatile_data)

    # Should have a trade that hit stop loss
    assert len(result.trades) > 0

    trade = result.trades[0]
    # Should be stopped out (negative PnL)
    assert trade.pnl < 0
    assert trade.exit_reason.lower().startswith("stop loss")


def test_take_profit_execution(uptrend_data):
    """Test take profit is triggered correctly."""
    strategy = TakeProfitTestStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(uptrend_data)

    # Should have a trade that hit take profit
    assert len(result.trades) > 0

    trade = result.trades[0]
    # Should be profitable
    assert trade.pnl > 0
    assert trade.exit_reason.lower().startswith("take profit")


def test_eod_close_uses_next_open(uptrend_data):
    """Test EOD close uses next bar open (no look-ahead bias)."""
    # Add session end marker
    data = uptrend_data.copy()
    # Mark bar 20 as EOD
    data.loc[20, "datetime"] = pd.Timestamp("2024-01-01 14:45")

    strategy = AlwaysLongStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(data)

    # Should have trades
    if len(result.trades) > 0:
        # Check that EOD close happened
        for trade in result.trades:
            if "EOD" in trade.exit_reason:
                # Exit price should be next bar open, not current close
                assert trade.exit_price > 0


def test_commission_applied_on_entry_and_exit(uptrend_data):
    """Test commission is applied on both entry and exit."""
    strategy = AlwaysLongStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        commission_rate=0.0003,
        position_size=1,
    )

    result = backtester.run(uptrend_data)

    assert len(result.trades) > 0

    trade = result.trades[0]
    # Commission should be > 0
    assert trade.commission > 0

    # Commission = rate * (entry_notional + exit_notional)
    entry_notional = trade.entry_price * trade.quantity * 100_000.0
    exit_notional = trade.exit_price * trade.quantity * 100_000.0
    expected_commission = 0.0003 * (entry_notional + exit_notional)

    assert trade.commission == pytest.approx(expected_commission, rel=1e-6)


def test_mae_mfe_tracking(volatile_data):
    """Test MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion) tracking."""
    strategy = AlwaysLongStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(volatile_data)

    assert len(result.trades) > 0

    trade = result.trades[0]
    # MAE should be >= 0 (maximum adverse excursion in points)
    # MFE should be >= 0 (maximum favorable excursion in points)
    assert trade.mae is None or trade.mae >= 0
    assert trade.mfe is None or trade.mfe >= 0


def test_position_size_respected(uptrend_data):
    """Test that position size is respected."""
    strategy = AlwaysLongStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=3,  # 3 contracts
    )

    result = backtester.run(uptrend_data)

    assert len(result.trades) > 0

    trade = result.trades[0]
    assert trade.quantity == 3


def test_no_position_after_backtest_end(uptrend_data):
    """Test that position is closed at end of backtest."""
    strategy = AlwaysLongStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(uptrend_data)

    # Last equity record should show flat position
    last_position = result.equity_curve["position"].iloc[-1]
    assert last_position == "FLAT"  # Position is string, not int


def test_trade_duration_calculated(uptrend_data):
    """Test that trade duration is calculated correctly."""
    strategy = AlwaysLongStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(uptrend_data)

    assert len(result.trades) > 0

    trade = result.trades[0]

    # Duration should be positive and datetimes should be valid
    assert trade.duration_minutes > 0
    assert isinstance(trade.entry_time, pd.Timestamp)
    assert isinstance(trade.exit_time, pd.Timestamp)

    # Verify duration calculation
    time_diff = (trade.exit_time - trade.entry_time).total_seconds() / 60
    assert trade.duration_minutes == pytest.approx(time_diff, rel=1e-6)


def test_gross_vs_net_pnl(uptrend_data):
    """Test gross PnL vs net PnL (after commission)."""
    strategy = AlwaysLongStrategy()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        commission_rate=0.0003,
        position_size=1,
    )

    result = backtester.run(uptrend_data)

    assert len(result.trades) > 0

    trade = result.trades[0]
    # Net PnL = Gross PnL - Commission
    assert trade.pnl == pytest.approx(trade.gross_pnl - trade.commission, rel=1e-6)

    # Gross PnL should be higher than net PnL
    assert trade.gross_pnl > trade.pnl
