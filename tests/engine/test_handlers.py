"""
Unit tests for event handlers - strategy, risk, broker, account.
"""

from datetime import datetime
from unittest.mock import Mock

from src.engine.account.account import AccountState
from src.engine.core.event_bus import EventBus
from src.engine.core.events import (
    EventType,
    FillEvent,
    MarketEvent,
    SignalEvent,
)
from src.engine.core.handlers import (
    AccountHandler,
    RiskHandler,
    StrategyHandler,
)
from src.strategy.signal import Signal, TradeSignal


class TestStrategyHandler:
    """Test StrategyHandler signal generation."""

    def test_strategy_handler_emits_signal(self):
        """Test strategy handler emits signal on market event."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)

        # Mock strategy
        strategy = Mock()
        strategy.generate_signal.return_value = TradeSignal(
            signal=Signal.LONG,
            entry_price=1000.0,
            stop_loss=990.0,
            take_profit=1020.0,
            reason="Test signal",
        )

        handler = StrategyHandler(strategy, account, bus)

        # Capture emitted signals
        received_signals = []
        bus.subscribe(EventType.SIGNAL, lambda e: received_signals.append(e))

        # Emit market event
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            bar={"open": 1000, "close": 1005},
        )
        handler.on_market(market_event)

        # Verify signal emitted
        assert len(received_signals) == 1
        signal = received_signals[0]
        assert signal.signal_type == "long"
        assert signal.entry_price == 1000.0
        assert signal.stop_loss == 990.0
        assert signal.take_profit == 1020.0

    def test_strategy_handler_hold_signal_not_emitted(self):
        """Test hold signal is not emitted."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)

        strategy = Mock()
        strategy.generate_signal.return_value = TradeSignal()  # Default is HOLD

        handler = StrategyHandler(strategy, account, bus)

        received_signals = []
        bus.subscribe(EventType.SIGNAL, lambda e: received_signals.append(e))

        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            bar={"open": 1000, "close": 1005},
        )
        handler.on_market(market_event)

        # No signal should be emitted for hold
        assert len(received_signals) == 0

    def test_strategy_handler_exception_handling(self):
        """Test strategy handler handles exceptions gracefully."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)

        strategy = Mock()
        strategy.generate_signal.side_effect = ValueError("Strategy error")

        handler = StrategyHandler(strategy, account, bus)

        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            bar={"open": 1000, "close": 1005},
        )

        # Should not raise
        handler.on_market(market_event)


class TestRiskHandler:
    """Test RiskHandler order creation and risk checks."""

    def test_risk_handler_creates_order(self):
        """Test risk handler creates order from signal."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        current_bar = {"open": 1000, "close": 1005}

        handler = RiskHandler(account, bus, current_bar)

        received_orders = []
        bus.subscribe(EventType.ORDER, lambda e: received_orders.append(e))

        signal_event = SignalEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            signal_type="long",
            entry_price=1000.0,
            stop_loss=990.0,
            take_profit=1020.0,
        )
        handler.on_signal(signal_event)

        # Verify order created
        assert len(received_orders) == 1
        order = received_orders[0]
        assert order.side == "buy"
        assert order.order_type == "limit"
        assert order.limit_price == 1000.0

    def test_risk_handler_blocks_when_position_open(self):
        """Test risk handler blocks signal when position already open."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        current_bar = {"open": 1000, "close": 1005}

        # Open a position
        from src.engine.execution.order import Order, OrderSide, OrderType

        order = Order(
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1,
            stop_loss=990.0,
            take_profit=1020.0,
        )
        order.fill(1000.0, datetime(2024, 1, 1, 9, 0), commission=15.0)
        account._open_position(order, datetime(2024, 1, 1, 9, 0))

        handler = RiskHandler(account, bus, current_bar)

        received_orders = []
        bus.subscribe(EventType.ORDER, lambda e: received_orders.append(e))

        signal_event = SignalEvent(
            timestamp=datetime(2024, 1, 1, 9, 5),
            signal_type="long",
            entry_price=1010.0,
        )
        handler.on_signal(signal_event)

        # No order should be created
        assert len(received_orders) == 0

    def test_risk_handler_blocks_on_daily_loss(self):
        """Test risk handler blocks signal when daily loss limit hit."""
        bus = EventBus()
        account = AccountState(
            initial_capital=500_000_000,
            max_daily_loss_pct=0.02,  # 2% daily loss limit
        )
        current_bar = {"open": 1000, "close": 1005}

        # Simulate daily loss hit
        account.risk_manager._daily_loss_hit = True

        handler = RiskHandler(account, bus, current_bar)

        received_orders = []
        bus.subscribe(EventType.ORDER, lambda e: received_orders.append(e))

        signal_event = SignalEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            signal_type="long",
            entry_price=1000.0,
        )
        handler.on_signal(signal_event)

        # No order should be created
        assert len(received_orders) == 0

    def test_risk_handler_market_order_for_zero_entry_price(self):
        """Test risk handler creates market order when entry_price is 0."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        current_bar = {"open": 1000, "close": 1005}

        handler = RiskHandler(account, bus, current_bar)

        received_orders = []
        bus.subscribe(EventType.ORDER, lambda e: received_orders.append(e))

        signal_event = SignalEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            signal_type="short",
            entry_price=0.0,  # Market order
            stop_loss=1010.0,
        )
        handler.on_signal(signal_event)

        assert len(received_orders) == 1
        order = received_orders[0]
        assert order.order_type == "market"
        assert order.side == "sell"
        assert order.limit_price is None


class TestAccountHandler:
    """Test AccountHandler position updates."""

    def test_account_handler_updates_position(self):
        """Test account handler updates position on fill."""
        account = AccountState(initial_capital=500_000_000)
        handler = AccountHandler(account)

        initial_cash = account.cash

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            side="buy",
            quantity=1,
            fill_price=1000.0,
            commission=15.0,
            stop_loss=990.0,
            take_profit=1020.0,
        )
        handler.on_fill(fill_event)

        # Verify position opened
        assert account.position.is_long
        assert account.position.quantity == 1
        assert account.position.entry_price == 1000.0
        assert account.position.stop_loss == 990.0
        assert account.position.take_profit == 1020.0

        # Verify cash deducted
        assert account.cash == initial_cash - 15.0

    def test_account_handler_short_position(self):
        """Test account handler opens short position."""
        account = AccountState(initial_capital=500_000_000)
        handler = AccountHandler(account)

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            side="sell",
            quantity=2,
            fill_price=1000.0,
            commission=30.0,
        )
        handler.on_fill(fill_event)

        assert account.position.is_short
        assert account.position.quantity == 2
        assert account.position.entry_price == 1000.0
