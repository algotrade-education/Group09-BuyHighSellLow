"""
Unit tests for Event types - immutable event dataclasses.
"""

from datetime import datetime

import pytest

from src.engine.core.events import (
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)


class TestMarketEvent:
    """Test MarketEvent creation and immutability."""

    def test_create_market_event(self):
        """Test creating a market event."""
        timestamp = datetime(2024, 1, 1, 9, 0)
        bar = {"open": 1000, "high": 1010, "low": 990, "close": 1005, "volume": 100}

        event = MarketEvent(timestamp=timestamp, bar=bar)

        assert event.timestamp == timestamp
        assert event.bar == bar
        assert event.symbol == "VN30F1M"
        assert event.event_type == EventType.MARKET

    def test_market_event_custom_symbol(self):
        """Test market event with custom symbol."""
        event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            bar={"open": 1000, "close": 1005},
            symbol="VN30F2M",
        )

        assert event.symbol == "VN30F2M"

    def test_market_event_immutable(self):
        """Test market event is immutable."""
        event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            bar={"open": 1000, "close": 1005},
        )

        with pytest.raises(AttributeError):
            event.timestamp = datetime(2024, 1, 2, 9, 0)


class TestSignalEvent:
    """Test SignalEvent creation and validation."""

    def test_create_signal_event(self):
        """Test creating a signal event."""
        timestamp = datetime(2024, 1, 1, 9, 0)

        event = SignalEvent(
            timestamp=timestamp,
            signal_type="long",
            entry_price=1000.0,
            stop_loss=990.0,
            take_profit=1020.0,
            reason="ORB breakout",
        )

        assert event.timestamp == timestamp
        assert event.signal_type == "long"
        assert event.entry_price == 1000.0
        assert event.stop_loss == 990.0
        assert event.take_profit == 1020.0
        assert event.reason == "ORB breakout"
        assert event.event_type == EventType.SIGNAL

    def test_signal_event_defaults(self):
        """Test signal event with default values."""
        event = SignalEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            signal_type="short",
        )

        assert event.entry_price == 0.0
        assert event.stop_loss == 0.0
        assert event.take_profit == 0.0
        assert event.reason == ""
        assert event.metadata == {}

    def test_signal_event_with_metadata(self):
        """Test signal event with metadata."""
        metadata = {"atr": 10.5, "confidence": 0.85}

        event = SignalEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            signal_type="long",
            metadata=metadata,
        )

        assert event.metadata == metadata


class TestOrderEvent:
    """Test OrderEvent creation and validation."""

    def test_create_market_order(self):
        """Test creating a market order event."""
        timestamp = datetime(2024, 1, 1, 9, 0)

        event = OrderEvent(
            timestamp=timestamp,
            order_type="market",
            side="buy",
            quantity=1,
        )

        assert event.timestamp == timestamp
        assert event.order_type == "market"
        assert event.side == "buy"
        assert event.quantity == 1
        assert event.limit_price is None
        assert event.event_type == EventType.ORDER

    def test_create_limit_order(self):
        """Test creating a limit order event."""
        event = OrderEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            order_type="limit",
            side="sell",
            quantity=2,
            limit_price=1010.0,
            stop_loss=1020.0,
            take_profit=990.0,
        )

        assert event.order_type == "limit"
        assert event.side == "sell"
        assert event.quantity == 2
        assert event.limit_price == 1010.0
        assert event.stop_loss == 1020.0
        assert event.take_profit == 990.0


class TestFillEvent:
    """Test FillEvent creation and validation."""

    def test_create_fill_event(self):
        """Test creating a fill event."""
        timestamp = datetime(2024, 1, 1, 9, 0)

        event = FillEvent(
            timestamp=timestamp,
            side="buy",
            quantity=1,
            fill_price=1000.0,
            commission=15.0,
            slippage=0.5,
        )

        assert event.timestamp == timestamp
        assert event.side == "buy"
        assert event.quantity == 1
        assert event.fill_price == 1000.0
        assert event.commission == 15.0
        assert event.slippage == 0.5
        assert event.event_type == EventType.FILL

    def test_fill_event_defaults(self):
        """Test fill event with default values."""
        event = FillEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            side="sell",
            quantity=1,
            fill_price=1000.0,
        )

        assert event.commission == 0.0
        assert event.slippage == 0.0
        assert event.stop_loss is None
        assert event.take_profit is None
        assert event.order_id == 0

    def test_fill_event_with_sl_tp(self):
        """Test fill event with stop loss and take profit."""
        event = FillEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            side="buy",
            quantity=1,
            fill_price=1000.0,
            stop_loss=990.0,
            take_profit=1020.0,
            order_id=123,
        )

        assert event.stop_loss == 990.0
        assert event.take_profit == 1020.0
        assert event.order_id == 123
