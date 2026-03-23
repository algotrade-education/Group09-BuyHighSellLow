"""
Unit tests for EventBus - publish-subscribe event routing.
"""

from datetime import datetime

from src.engine.core.event_bus import EventBus
from src.engine.core.events import EventType, MarketEvent, SignalEvent


class TestEventBus:
    """Test EventBus subscription and event routing."""

    def test_subscribe_and_emit(self):
        """Test basic subscribe and emit functionality."""
        bus = EventBus()
        received_events = []

        def handler(event):
            received_events.append(event)

        bus.subscribe(EventType.MARKET, handler)

        event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            bar={"open": 1000, "high": 1010, "low": 990, "close": 1005},
        )
        bus.emit(event)

        assert len(received_events) == 1
        assert received_events[0] == event

    def test_multiple_handlers_same_event(self):
        """Test multiple handlers for same event type."""
        bus = EventBus()
        handler1_calls = []
        handler2_calls = []

        def handler1(event):
            handler1_calls.append(event)

        def handler2(event):
            handler2_calls.append(event)

        bus.subscribe(EventType.MARKET, handler1)
        bus.subscribe(EventType.MARKET, handler2)

        event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            bar={"open": 1000, "close": 1005},
        )
        bus.emit(event)

        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1

    def test_handler_execution_order(self):
        """Test handlers execute in registration order."""
        bus = EventBus()
        execution_order = []

        def handler1(event):
            execution_order.append(1)

        def handler2(event):
            execution_order.append(2)

        def handler3(event):
            execution_order.append(3)

        bus.subscribe(EventType.MARKET, handler1)
        bus.subscribe(EventType.MARKET, handler2)
        bus.subscribe(EventType.MARKET, handler3)

        event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            bar={"open": 1000, "close": 1005},
        )
        bus.emit(event)

        assert execution_order == [1, 2, 3]

    def test_nested_event_emission(self):
        """Test handler can emit new events during processing."""
        bus = EventBus()
        received_signals = []

        def market_handler(event):
            # Handler emits new event
            bus.emit(
                SignalEvent(
                    timestamp=event.timestamp,
                    signal_type="long",
                    entry_price=1000,
                )
            )

        def signal_handler(event):
            received_signals.append(event)

        bus.subscribe(EventType.MARKET, market_handler)
        bus.subscribe(EventType.SIGNAL, signal_handler)

        event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            bar={"open": 1000, "close": 1005},
        )
        bus.emit(event)

        assert len(received_signals) == 1
        assert received_signals[0].signal_type == "long"

    def test_handler_exception_isolation(self):
        """Test handler exception doesn't stop other handlers."""
        bus = EventBus()
        handler2_called = []

        def failing_handler(event):
            raise ValueError("Handler failed")

        def working_handler(event):
            handler2_called.append(True)

        bus.subscribe(EventType.MARKET, failing_handler)
        bus.subscribe(EventType.MARKET, working_handler)

        event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            bar={"open": 1000, "close": 1005},
        )
        bus.emit(event)

        # Second handler should still execute
        assert len(handler2_called) == 1

    def test_unsubscribe(self):
        """Test unsubscribe removes handler."""
        bus = EventBus()
        received_events = []

        def handler(event):
            received_events.append(event)

        bus.subscribe(EventType.MARKET, handler)
        bus.unsubscribe(EventType.MARKET, handler)

        event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            bar={"open": 1000, "close": 1005},
        )
        bus.emit(event)

        assert len(received_events) == 0

    def test_no_handlers_for_event_type(self):
        """Test emitting event with no handlers doesn't crash."""
        bus = EventBus()

        event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            bar={"open": 1000, "close": 1005},
        )
        bus.emit(event)  # Should not raise

    def test_subscriber_count(self):
        """Test subscriber_count property."""
        bus = EventBus()

        def handler1(event):
            pass

        def handler2(event):
            pass

        bus.subscribe(EventType.MARKET, handler1)
        bus.subscribe(EventType.SIGNAL, handler1)
        bus.subscribe(EventType.SIGNAL, handler2)

        counts = bus.subscriber_count
        assert counts["market"] == 1
        assert counts["signal"] == 2

    def test_clear(self):
        """Test clear removes all subscriptions."""
        bus = EventBus()
        received_events = []

        def handler(event):
            received_events.append(event)

        bus.subscribe(EventType.MARKET, handler)
        bus.clear()

        event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            bar={"open": 1000, "close": 1005},
        )
        bus.emit(event)

        assert len(received_events) == 0
        assert bus.subscriber_count == {}
