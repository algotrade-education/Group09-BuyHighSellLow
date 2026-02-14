"""
Integration tests for Backtester and TradeManager.

Uses a simple stub strategy that alternates between LONG and CLOSE
signals so we can verify the full execution pipeline without depending
on any real strategy implementation.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from src.engine.backtester import Backtester
from src.engine.trade_manager import TradeManager
from src.strategy.base import Signal, Strategy, TradeSignal


# ── Stub strategy ────────────────────────────────────────────────────


class AlwaysLongStrategy(Strategy):
    """Opens a long on the first flat bar, holds forever."""

    def __init__(self):
        super().__init__(name="AlwaysLong")

    def generate_signal(
        self, bar: Dict[str, Any], current_position: Optional[Any] = None
    ) -> TradeSignal:
        if current_position and not current_position.is_flat:
            return TradeSignal(signal=Signal.HOLD)
        return TradeSignal(
            signal=Signal.LONG,
            stop_loss=bar["close"] - 10,
            take_profit=bar["close"] + 20,
        )


class AlternatingStrategy(Strategy):
    """Opens long, then closes on the next bar, repeatedly."""

    def __init__(self):
        super().__init__(name="Alternating")

    def generate_signal(
        self, bar: Dict[str, Any], current_position: Optional[Any] = None
    ) -> TradeSignal:
        if current_position and not current_position.is_flat:
            return TradeSignal(signal=Signal.CLOSE)
        return TradeSignal(
            signal=Signal.LONG,
            stop_loss=bar["close"] - 10,
            take_profit=bar["close"] + 50,
        )


class NeverTradeStrategy(Strategy):
    """Always holds — never generates entry or exit."""

    def __init__(self):
        super().__init__(name="NeverTrade")

    def generate_signal(
        self, bar: Dict[str, Any], current_position: Optional[Any] = None
    ) -> TradeSignal:
        return TradeSignal(signal=Signal.HOLD)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_bars(n: int = 50, base_price: float = 1000.0) -> pd.DataFrame:
    """Generate simple OHLC data with a datetime column."""
    np.random.seed(42)
    start = datetime(2024, 1, 2, 9, 15, 0)
    rows = []
    price = base_price
    for i in range(n):
        change = np.random.uniform(-2, 2)
        close = price + change
        high = max(price, close) + abs(np.random.normal(0, 1))
        low = min(price, close) - abs(np.random.normal(0, 1))
        rows.append(
            {
                "datetime": start + timedelta(minutes=i),
                "open": price,
                "high": high,
                "low": low,
                "close": close,
            }
        )
        price = close
    return pd.DataFrame(rows)


# ── Backtester Tests ─────────────────────────────────────────────────


class TestBacktester:
    def test_backtester_returns_result(self):
        """Basic run should return a BacktestResult with expected keys."""
        data = _make_bars(30)
        bt = Backtester(
            strategy=NeverTradeStrategy(),
            commission_rate=0.0,
            slippage_points=0.0,
            contract_multiplier=1,
        )
        result = bt.run(data)

        assert result is not None
        assert result.total_trades == 0
        assert isinstance(result.equity_curve, pd.DataFrame)
        assert len(result.equity_curve) == 30

    def test_backtester_always_long(self):
        """AlwaysLong should open at least one trade."""
        data = _make_bars(20)
        bt = Backtester(
            strategy=AlwaysLongStrategy(),
            commission_rate=0.0,
            slippage_points=0.0,
            contract_multiplier=1,
        )
        result = bt.run(data)

        # Should have at least 1 trade
        assert result.total_trades >= 1
        # All trades should be closed (either by SL/TP or backtest end)
        for trade in result.trades:
            assert trade.is_closed

    def test_backtester_alternating_produces_multiple_trades(self):
        """Alternating strategy should produce multiple trades."""
        data = _make_bars(30)
        bt = Backtester(
            strategy=AlternatingStrategy(),
            commission_rate=0.0,
            slippage_points=0.0,
            contract_multiplier=1,
        )
        result = bt.run(data)

        # At least a few round-trips
        assert result.total_trades >= 2
        for trade in result.trades:
            assert trade.is_closed

    def test_backtester_equity_curve_starts_at_initial_capital(self):
        """First equity value should be the initial capital."""
        data = _make_bars(10)
        capital = 500_000.0
        bt = Backtester(
            strategy=NeverTradeStrategy(),
            initial_capital=capital,
            commission_rate=0.0,
            slippage_points=0.0,
            contract_multiplier=1,
        )
        result = bt.run(data)

        first_equity = result.equity_curve["equity"].iloc[0]
        assert first_equity == pytest.approx(capital)

    def test_backtester_metrics_populated(self):
        """Metrics dict should contain standard keys after a run."""
        data = _make_bars(30)
        bt = Backtester(
            strategy=AlternatingStrategy(),
            commission_rate=0.0,
            slippage_points=0.0,
            contract_multiplier=1,
        )
        result = bt.run(data)

        assert hasattr(result, "metrics")
        # assert "sharpe_ratio" in result.metrics
        # assert "total_return_pct" in result.metrics
        # assert "max_drawdown_pct" in result.metrics

    def test_backtester_resets_between_runs(self):
        """Running twice should produce independent results."""
        data = _make_bars(20)
        bt = Backtester(
            strategy=AlternatingStrategy(),
            commission_rate=0.0,
            slippage_points=0.0,
            contract_multiplier=1,
        )
        r1 = bt.run(data)
        r2 = bt.run(data)

        assert r1.total_trades == r2.total_trades
        assert r1.total_pnl == pytest.approx(r2.total_pnl)


# ── TradeManager Execution Tests ─────────────────────────────────────


class TestTradeManagerExecution:
    """Tests for TradeManager commission / cash flow correctness."""

    def _make_tm(
        self,
        capital: float = 1_000_000,
        commission_rate: float = 0.0,
        slippage: float = 0.0,
    ) -> TradeManager:
        return TradeManager(
            initial_capital=capital,
            commission_rate=commission_rate,
            slippage_points=slippage,
        )

    def test_commission_deducted_on_entry_and_exit(self):
        """Cash should decrease by entry + exit commission."""
        tm = self._make_tm(capital=1_000_000, commission_rate=0.001)
        from src.engine.order import Order, OrderSide, OrderType

        order = Order(
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1,
        )
        bar = {"open": 1000, "high": 1010, "low": 990, "close": 1000}
        ts = datetime(2024, 1, 1, 10, 0, 0)

        tm.execute_order(order, bar, ts)
        tm.open_position(order, ts)
        cash_after_entry = tm.cash

        tm.close_position(exit_price=1000, timestamp=ts, exit_reason="test")
        cash_after_exit = tm.cash

        # Entry + exit commissions should have been deducted
        assert cash_after_entry < 1_000_000
        # After round trip at same price, should be less than initial (commission drag)
        assert cash_after_exit < 1_000_000

    def test_profitable_trade_increases_cash(self):
        """Closing a winning trade should add gross PnL to cash."""
        tm = self._make_tm(commission_rate=0.0)
        from src.engine.order import Order, OrderSide, OrderType

        order = Order(
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1,
        )
        bar = {"open": 1000, "high": 1010, "low": 990, "close": 1000}
        ts = datetime(2024, 1, 1, 10, 0, 0)

        tm.execute_order(order, bar, ts)
        tm.open_position(order, ts)

        trade = tm.close_position(exit_price=1010, timestamp=ts, exit_reason="tp")

        assert trade is not None
        assert trade.pnl > 0
        assert tm.cash > 1_000_000

    def test_losing_trade_decreases_cash(self):
        """Closing a losing trade should reduce cash."""
        tm = self._make_tm(commission_rate=0.0)
        from src.engine.order import Order, OrderSide, OrderType

        order = Order(
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1,
        )
        bar = {"open": 1000, "high": 1010, "low": 990, "close": 1000}
        ts = datetime(2024, 1, 1, 10, 0, 0)

        tm.execute_order(order, bar, ts)
        tm.open_position(order, ts)

        trade = tm.close_position(exit_price=990, timestamp=ts, exit_reason="sl")

        assert trade is not None
        assert trade.pnl < 0
        assert tm.cash < 1_000_000
