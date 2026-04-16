"""
Integration tests for Backtester - test real backtest flow end-to-end.

Tests cover:
- Complete backtest execution flow
- Order execution and position management
- Trade recording and equity tracking
- EOD close behavior
- Daily loss limits
- Warmup period handling
"""

import pandas as pd
import pytest

from src.data.indicators.registry import IndicatorRegistry, IndicatorSpec
from src.engine.backtester import Backtester
from src.strategy.base import PositionSnapshot, StrategyBase
from src.strategy.signal import Signal, TradeSignal

# --- Test Strategy ---


class SimpleMAStrategy(StrategyBase):
    """Simple moving average crossover strategy for testing."""

    def __init__(self, fast_period: int = 5, slow_period: int = 10):
        super().__init__(name="SimpleMA")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signal(
        self,
        bar: dict,
        position: PositionSnapshot | None = None,
        is_warmup: bool = False,
    ) -> TradeSignal:
        if is_warmup:
            return TradeSignal()

        pos = position or PositionSnapshot.flat()

        # Get MA values
        fast_ma = bar.get(f"sma_{self.fast_period}")
        slow_ma = bar.get(f"sma_{self.slow_period}")

        if fast_ma is None or slow_ma is None:
            return TradeSignal()

        close = float(bar["close"])
        atr = bar.get("atr_14", close * 0.02)  # Default 2% if no ATR

        # Entry signals
        if pos.is_flat:
            if fast_ma > slow_ma:
                return TradeSignal(
                    signal=Signal.LONG,
                    entry_price=close,
                    stop_loss=close - 2 * atr,
                    take_profit=close + 3 * atr,
                )
            elif fast_ma < slow_ma:
                return TradeSignal(
                    signal=Signal.SHORT,
                    entry_price=close,
                    stop_loss=close + 2 * atr,
                    take_profit=close - 3 * atr,
                )

        # Exit signals
        if pos.is_long and fast_ma < slow_ma:
            return TradeSignal(signal=Signal.EXIT, entry_price=close)
        if pos.is_short and fast_ma > slow_ma:
            return TradeSignal(signal=Signal.EXIT, entry_price=close)

        return TradeSignal()

    @classmethod
    def build_registry(cls, **params) -> IndicatorRegistry:
        fast = params.get("fast_period", 5)
        slow = params.get("slow_period", 10)
        registry = IndicatorRegistry()
        registry.register(IndicatorSpec("sma", {"period": fast}, f"sma_{fast}"))
        registry.register(IndicatorSpec("sma", {"period": slow}, f"sma_{slow}"))
        registry.register(IndicatorSpec("atr", {"period": 14}, "atr_14"))
        return registry


# --- Test Data Fixtures ---


@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC data with trend for testing."""
    dates = pd.date_range("2024-01-01 09:00", periods=100, freq="5min")

    # Create uptrend then downtrend
    base_price = 1000.0
    prices = []
    for i in range(100):
        if i < 50:
            # Uptrend
            price = base_price + i * 2
        else:
            # Downtrend
            price = base_price + (100 - i) * 2
        prices.append(price)

    df = pd.DataFrame(
        {
            "datetime": dates,
            "open": [p - 1 for p in prices],
            "high": [p + 2 for p in prices],
            "low": [p - 2 for p in prices],
            "close": prices,
            "volume": [1000] * 100,
        }
    )

    # Add simple moving averages
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["atr_14"] = df["close"].rolling(14).std() * 1.5  # Simplified ATR

    # Reset index to make datetime a column (not index)
    df = df.reset_index(drop=True)

    return df


@pytest.fixture
def flat_data():
    """Generate flat price data (no trend)."""
    dates = pd.date_range("2024-01-01 09:00", periods=50, freq="5min")
    price = 1000.0

    df = pd.DataFrame(
        {
            "datetime": dates,
            "open": [price] * 50,
            "high": [price + 1] * 50,
            "low": [price - 1] * 50,
            "close": [price] * 50,
            "volume": [1000] * 50,
        }
    )

    df["sma_5"] = price
    df["sma_10"] = price
    df["atr_14"] = 5.0

    df = df.reset_index(drop=True)

    return df


# --- Integration Tests ---


def test_backtester_basic_flow(sample_ohlc_data):
    """Test basic backtest execution flow."""
    strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(sample_ohlc_data, warmup_bars=10)

    # Verify result structure
    assert result is not None
    assert hasattr(result, "trades")
    assert hasattr(result, "equity_curve")
    assert hasattr(result, "metrics")

    # Should have some trades (uptrend then downtrend should trigger signals)
    assert len(result.trades) > 0

    # Equity curve should have same length as data
    assert len(result.equity_curve) == len(sample_ohlc_data)

    # Metrics should be calculated
    assert "total_trades" in result.metrics
    assert "sharpe_ratio" in result.metrics


def test_backtester_no_trades_flat_market(flat_data):
    """Test backtest with no signals (flat market)."""
    strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(flat_data, warmup_bars=10)

    # Should have no trades in flat market
    assert len(result.trades) == 0

    # Equity should remain at initial capital
    assert result.equity_curve["equity"].iloc[-1] == pytest.approx(500_000_000, rel=1e-6)


def test_backtester_warmup_period(sample_ohlc_data):
    """Test that warmup period prevents trades."""
    strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    # Run with large warmup
    result = backtester.run(sample_ohlc_data, warmup_bars=90)

    # Should have very few or no trades due to warmup
    assert len(result.trades) < 5


def test_backtester_position_lifecycle(sample_ohlc_data):
    """Test complete position lifecycle: entry -> exit."""
    strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(sample_ohlc_data, warmup_bars=10)

    # Verify trades have proper structure
    for trade in result.trades:
        assert trade.entry_time is not None
        assert trade.exit_time is not None
        assert trade.entry_price > 0
        assert trade.exit_price > 0
        assert trade.quantity > 0
        assert trade.exit_time > trade.entry_time


def test_backtester_commission_calculation(sample_ohlc_data):
    """Test that commission is properly calculated."""
    strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        commission_rate=0.0003,  # 0.03%
        position_size=1,
    )

    result = backtester.run(sample_ohlc_data, warmup_bars=10)

    # All trades should have commission
    for trade in result.trades:
        assert trade.commission > 0
        # Commission should be reasonable (< 1% of notional)
        notional = trade.entry_price * trade.quantity * 100_000
        assert trade.commission < notional * 0.01


def test_backtester_equity_tracking(sample_ohlc_data):
    """Test equity curve tracking."""
    strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(sample_ohlc_data, warmup_bars=10)

    equity_curve = result.equity_curve

    # Verify equity curve structure
    # datetime should be a column, not index
    assert "datetime" in equity_curve.columns
    assert "equity" in equity_curve.columns
    assert "cash" in equity_curve.columns
    assert "position" in equity_curve.columns

    # Equity should never be negative
    assert (equity_curve["equity"] > 0).all()

    # First equity should be initial capital
    assert equity_curve["equity"].iloc[0] == pytest.approx(500_000_000, rel=1e-6)


def test_backtester_max_daily_loss(sample_ohlc_data):
    """Test daily loss limit enforcement."""
    strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        max_daily_loss_pct=2.0,  # 2% daily loss limit
        position_size=1,
    )

    result = backtester.run(sample_ohlc_data, warmup_bars=10)

    # Should complete without error
    assert result is not None
    # Note: Actual loss limit testing requires specific price movements


def test_backtester_order_ttl(sample_ohlc_data):
    """Test order time-to-live expiration."""
    strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        order_ttl_bars=1,  # Orders expire after 1 bar
        position_size=1,
    )

    result = backtester.run(sample_ohlc_data, warmup_bars=10)

    # Should complete without error
    assert result is not None


def test_backtester_progress_callback(sample_ohlc_data):
    """Test progress callback is called."""
    strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    progress_calls = []

    def progress_callback(current, total):
        progress_calls.append((current, total))

    _ = backtester.run(
        sample_ohlc_data,
        warmup_bars=10,
        progress_callback=progress_callback,
    )

    # Progress callback should be called
    assert len(progress_calls) > 0
    # Last call should be (total, total)
    assert progress_calls[-1] == (len(sample_ohlc_data), len(sample_ohlc_data))


def test_backtester_empty_data():
    """Test backtest with empty data raises error."""
    strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Data is empty"):
        backtester.run(empty_df)


def test_backtester_missing_columns():
    """Test backtest with missing required columns raises error."""
    strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    # Missing 'close' column
    bad_df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=10, freq="5min"),
            "open": [1000] * 10,
            "high": [1010] * 10,
            "low": [990] * 10,
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        backtester.run(bad_df)


def test_backtester_result_properties(sample_ohlc_data):
    """Test BacktestResult convenience properties."""
    strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result = backtester.run(sample_ohlc_data, warmup_bars=10)

    # Test properties
    assert result.total_trades >= 0
    assert result.winning_trades >= 0
    assert result.losing_trades >= 0
    assert (
        result.winning_trades + result.losing_trades + result.breakeven_trades
        == result.total_trades
    )

    if result.total_trades > 0:
        assert 0 <= result.win_rate <= 100


def test_backtester_multiple_runs_independent(sample_ohlc_data):
    """Test that multiple backtest runs are independent."""
    strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
    backtester = Backtester(
        strategy=strategy,
        initial_capital=500_000_000,
        position_size=1,
    )

    result1 = backtester.run(sample_ohlc_data, warmup_bars=10)
    result2 = backtester.run(sample_ohlc_data, warmup_bars=10)

    # Results should be identical
    assert len(result1.trades) == len(result2.trades)
    assert result1.equity_curve["equity"].iloc[-1] == result2.equity_curve["equity"].iloc[-1]
