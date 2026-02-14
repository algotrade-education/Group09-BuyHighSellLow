"""
Unit tests for Order, Trade, Position, and TradeManager.
"""

from datetime import datetime

import pytest

from src.engine.order import Order, OrderSide, OrderStatus, OrderType
from src.engine.position import Position, PositionSide, Trade
from src.engine.trade_manager import TradeManager


# ── Helpers ──────────────────────────────────────────────────────────


def _make_order(
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    quantity: int = 1,
    limit_price: float = 0.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> Order:
    """Create an Order directly (replaces removed OrderFactory)."""
    return Order(
        order_type=order_type,
        side=side,
        quantity=quantity,
        limit_price=limit_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


def _filled_order(
    side: OrderSide = OrderSide.BUY,
    price: float = 100.0,
    **kwargs,
) -> Order:
    """Create and fill an order in one step."""
    order = _make_order(side=side, **kwargs)
    order.fill(price=price, timestamp=datetime.now())
    return order


# ── Order Tests ──────────────────────────────────────────────────────


class TestOrder:
    """Tests for Order class."""

    def test_market_order_creation(self):
        """Test market order creation."""
        order = _make_order(side=OrderSide.BUY, quantity=2)

        assert order.order_type == OrderType.MARKET
        assert order.side == OrderSide.BUY
        assert order.quantity == 2
        assert order.is_buy
        assert not order.is_sell
        assert order.is_pending

    def test_limit_order_creation(self):
        """Test limit order creation."""
        order = _make_order(
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=100.0,
        )

        assert order.order_type == OrderType.LIMIT
        assert order.side == OrderSide.SELL
        assert order.limit_price == 100.0
        assert order.is_sell

    def test_order_fill(self):
        """Test order fill."""
        order = _make_order()
        timestamp = datetime.now()

        order.fill(
            price=100.0,
            timestamp=timestamp,
            commission=0.15,
            slippage=0.5,
        )

        assert order.is_filled
        assert order.filled_price == 100.0
        assert order.filled_at == timestamp
        assert order.commission == 0.15

    def test_order_cancel(self):
        """Test order cancellation."""
        order = _make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=100.0,
        )
        order.cancel(reason="Test cancel")

        assert order.status == OrderStatus.CANCELLED
        assert order.reason == "Test cancel"

    def test_order_with_stops(self):
        """Test order with stop loss and take profit."""
        order = _make_order(stop_loss=95.0, take_profit=110.0)

        assert order.stop_loss == 95.0
        assert order.take_profit == 110.0


# ── Trade Tests ──────────────────────────────────────────────────────


class TestTrade:
    """Tests for Trade class."""

    def test_trade_creation(self):
        """Test trade creation."""
        trade = Trade(
            trade_id=1,
            side=PositionSide.LONG,
            entry_time=datetime(2024, 1, 1, 10, 0),
            entry_price=100.0,
            quantity=1,
        )

        assert trade.trade_id == 1
        assert trade.side == PositionSide.LONG
        assert not trade.is_closed

    def test_trade_close_long_profit(self):
        """Test closing long trade with profit."""
        trade = Trade(
            trade_id=1,
            side=PositionSide.LONG,
            entry_time=datetime(2024, 1, 1, 10, 0),
            entry_price=100.0,
            quantity=1,
            commission=0.15,
        )

        trade.close(
            exit_price=110.0,
            exit_time=datetime(2024, 1, 1, 14, 0),
            commission=0.15,
            exit_reason="Take profit",
        )

        assert trade.is_closed
        assert trade.is_winner
        # PnL = (110 - 100) * 1 - 0.30 = 9.70
        assert trade.pnl == pytest.approx(9.70)

    def test_trade_close_long_loss(self):
        """Test closing long trade with loss."""
        trade = Trade(
            trade_id=1,
            side=PositionSide.LONG,
            entry_time=datetime(2024, 1, 1, 10, 0),
            entry_price=100.0,
            quantity=1,
        )

        trade.close(
            exit_price=95.0,
            exit_time=datetime(2024, 1, 1, 14, 0),
        )

        assert trade.is_closed
        assert not trade.is_winner
        assert trade.pnl == -5.0

    def test_trade_close_short_profit(self):
        """Test closing short trade with profit."""
        trade = Trade(
            trade_id=1,
            side=PositionSide.SHORT,
            entry_time=datetime(2024, 1, 1, 10, 0),
            entry_price=100.0,
            quantity=1,
        )

        trade.close(
            exit_price=90.0,
            exit_time=datetime(2024, 1, 1, 14, 0),
        )

        assert trade.is_winner
        assert trade.pnl == 10.0

    def test_trade_duration(self):
        """Test trade duration calculation."""
        entry = datetime(2024, 1, 1, 10, 0, 0)
        exit = datetime(2024, 1, 1, 14, 30, 0)

        trade = Trade(
            trade_id=1,
            side=PositionSide.LONG,
            entry_time=entry,
            entry_price=100.0,
        )
        trade.close(exit_price=105.0, exit_time=exit)

        # Duration: 4.5 hours = 16200 seconds
        assert trade.duration == 16200


# ── Position Tests ───────────────────────────────────────────────────


class TestPosition:
    """Tests for Position class (pure state container)."""

    def test_position_initially_flat(self):
        """Test position starts flat."""
        position = Position()

        assert position.is_flat
        assert not position.is_long
        assert not position.is_short
        assert position.quantity == 0

    def test_open_long_position(self):
        """Test opening long position."""
        position = Position()
        order = _filled_order(stop_loss=95.0, take_profit=110.0)

        position.open(order, datetime.now())

        assert position.is_long
        assert position.entry_price == 100.0
        assert position.stop_loss == 95.0
        assert position.take_profit == 110.0

    def test_open_short_position(self):
        """Test opening short position."""
        position = Position()
        order = _filled_order(side=OrderSide.SELL)

        position.open(order, datetime.now())

        assert position.is_short

    def test_close_position(self):
        """Test closing position resets state."""
        position = Position()
        order = _filled_order()

        position.open(order, datetime.now())
        position.close()

        assert position.is_flat
        assert position.entry_price == 0.0
        assert position.quantity == 0

    def test_unrealized_pnl_long(self):
        """Test unrealized P&L for long position."""
        position = Position()
        order = _filled_order()

        position.open(order, datetime.now())

        # Price moved up
        pnl = position.update_unrealized_pnl(110.0)
        assert pnl == 10.0

        # Price moved down
        pnl = position.update_unrealized_pnl(95.0)
        assert pnl == -5.0

    def test_unrealized_pnl_short(self):
        """Test unrealized P&L for short position."""
        position = Position()
        order = _filled_order(side=OrderSide.SELL)

        position.open(order, datetime.now())

        # Price moved down (profit for short)
        pnl = position.update_unrealized_pnl(90.0)
        assert pnl == 10.0

        # Price moved up (loss for short)
        pnl = position.update_unrealized_pnl(105.0)
        assert pnl == -5.0

    def test_check_stop_loss_long(self):
        """Test stop loss check for long position."""
        position = Position()
        order = _filled_order(stop_loss=95.0)

        position.open(order, datetime.now())

        assert not position.check_stop_loss(96.0)
        assert position.check_stop_loss(95.0)
        assert position.check_stop_loss(90.0)

    def test_check_take_profit_long(self):
        """Test take profit check for long position."""
        position = Position()
        order = _filled_order(take_profit=110.0)

        position.open(order, datetime.now())

        assert not position.check_take_profit(109.0)
        assert position.check_take_profit(110.0)
        assert position.check_take_profit(115.0)

    def test_cannot_open_when_in_position(self):
        """Test cannot open new position when already in one."""
        position = Position()
        order = _filled_order()

        position.open(order, datetime.now())

        order2 = _filled_order(side=OrderSide.SELL)
        with pytest.raises(ValueError):
            position.open(order2, datetime.now())

    def test_reset(self):
        """Test position reset."""
        position = Position()
        order = _filled_order()

        position.open(order, datetime.now())
        position.reset()

        assert position.is_flat
        assert position.quantity == 0


# ── TradeManager Tests ───────────────────────────────────────────────


class TestTradeManager:
    """Tests for trade history management (now owned by TradeManager)."""

    def _make_tm(self) -> TradeManager:
        return TradeManager(
            initial_capital=1_000_000,
            commission_rate=0.0,
            slippage_points=0.0,
        )

    def test_open_position_creates_trade(self):
        """Test that opening a position creates a Trade record."""
        tm = self._make_tm()
        order = _filled_order()

        trade = tm.open_position(order, datetime.now())

        assert trade is not None
        assert trade.trade_id == 1
        assert not trade.is_closed
        assert len(tm.trades) == 1

    def test_close_position_finalizes_trade(self):
        """Test that closing a position finalizes the Trade."""
        tm = self._make_tm()
        order = _filled_order()

        tm.open_position(order, datetime.now())
        trade = tm.close_position(
            exit_price=110.0,
            timestamp=datetime.now(),
            exit_reason="Take profit",
        )

        assert trade is not None
        assert trade.is_closed
        assert trade.pnl == 10.0
        assert tm.position.is_flat

    def test_trade_history(self):
        """Test trade history tracking across multiple trades."""
        tm = self._make_tm()

        # Trade 1: long
        order1 = _filled_order()
        tm.open_position(order1, datetime.now())
        tm.close_position(exit_price=110.0, timestamp=datetime.now())

        # Trade 2: short
        order2 = _filled_order(side=OrderSide.SELL, price=110.0)
        tm.open_position(order2, datetime.now())
        tm.close_position(exit_price=100.0, timestamp=datetime.now())

        assert len(tm.trades) == 2
        total_pnl = sum(t.pnl for t in tm.trades)
        assert total_pnl == 20.0  # 10 + 10

    def test_reset_clears_trades(self):
        """Test that reset clears trade history."""
        tm = self._make_tm()
        order = _filled_order()

        tm.open_position(order, datetime.now())
        tm.close_position(exit_price=110.0, timestamp=datetime.now())

        tm.reset()

        assert tm.position.is_flat
        assert len(tm.trades) == 0
