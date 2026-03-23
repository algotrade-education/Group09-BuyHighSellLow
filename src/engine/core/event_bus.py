"""
Publish-subscribe event bus for event-driven architecture.

The EventBus is the central nervous system of the event-driven backtester.
Handlers register interest in specific EventTypes, and the bus dispatches
events to all registered handlers.

Design Principles:
    - Synchronous dispatch (no async needed for backtesting)
    - Ordered execution (handlers called in registration order)
    - Nested event support (handlers can emit new events during processing)
    - Error isolation (handler exceptions don't crash the bus)

Example:
    ```python
    bus = EventBus()

    # Register handlers
    bus.subscribe(EventType.MARKET, strategy.on_market)
    bus.subscribe(EventType.SIGNAL, risk_manager.on_signal)
    bus.subscribe(EventType.ORDER, broker.on_order)
    bus.subscribe(EventType.FILL, account.on_fill)

    # Emit event - bus automatically routes to registered handlers
    bus.emit(MarketEvent(timestamp, bar))
    ```
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from src.engine.core.events import BaseEvent, EventType

logger = logging.getLogger(__name__)

Handler = Callable[[Any], None]  # Accept any event type, not just BaseEvent


class EventBus:
    """
    Synchronous publish-subscribe event bus.

    The EventBus manages event routing between decoupled components.
    Handlers register for specific event types, and the bus ensures
    events are delivered to all interested parties.

    Features:
        - Type-safe event routing via EventType enum
        - Queue-based processing to handle nested event emissions
        - Error isolation (one handler failure doesn't affect others)
        - Debug logging for event flow visibility

    Thread Safety:
        Not thread-safe. Designed for single-threaded backtesting.
        For multi-threaded use, add locking around emit/subscribe.

    Usage:
        ```python
        bus = EventBus()

        # Subscribe handlers to event types
        bus.subscribe(EventType.MARKET, strategy.on_market)
        bus.subscribe(EventType.SIGNAL, risk_manager.on_signal)

        # Emit events - bus routes to registered handlers
        bus.emit(MarketEvent(timestamp, bar))

        # Check subscription status
        print(bus.subscriber_count)  # {'market': 1, 'signal': 1}
        ```
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._queue: deque[BaseEvent] = deque()
        self._processing: bool = False

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """
        Register a handler for a specific event type.

        Args:
            event_type: The type of event to listen for
            handler: Callable that accepts a BaseEvent subclass

        Note:
            Handlers are called in registration order when events are emitted.
            The same handler can be registered multiple times (will be called multiple times).
        """
        self._handlers[event_type].append(handler)

        owner = getattr(handler, "__self__", None)
        owner_name = owner.__class__.__name__ if owner else "fn"
        handler_name = getattr(handler, "__name__", str(handler))

        logger.debug(
            "Subscribed %s.%s to %s",
            owner_name,
            handler_name,
            event_type,
        )

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        """
        Unregister a handler from an event type.

        Args:
            event_type: The event type to unsubscribe from
            handler: The handler to remove

        Note:
            If handler was registered multiple times, only first occurrence is removed.
        """
        handlers = self._handlers[event_type]
        if handler in handlers:
            handlers.remove(handler)

    def emit(self, event: BaseEvent) -> None:
        """
        Emit an event to all registered handlers.

        If called during event processing (handler emits new event),
        the new event is queued and processed after current handler completes.
        This prevents recursive dispatch and maintains predictable execution order.

        Args:
            event: Event to emit (must be BaseEvent subclass)

        Example:
            ```python
            # Handler can emit new events
            def on_signal(self, event: SignalEvent) -> None:
                if self._validate(event):
                    self._bus.emit(OrderEvent(...))  # Queued, not recursive
            ```
        """
        self._queue.append(event)
        if not self._processing:
            self._drain()

    def _drain(self) -> None:
        """
        Process all queued events.

        Events are processed FIFO. New events emitted during processing
        are added to the queue and processed in order.
        """
        self._processing = True
        try:
            while self._queue:
                event = self._queue.popleft()
                self._dispatch(event)
        finally:
            self._processing = False

    def _dispatch(self, event: BaseEvent) -> None:
        """
        Dispatch an event to all registered handlers.

        Handler exceptions are caught and logged but don't stop
        other handlers from executing.

        Args:
            event: Event to dispatch
        """
        handlers = self._handlers.get(event.event_type, []).copy()
        if not handlers:
            logger.debug("No handlers for event type: %s", event.event_type)
            return

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    "Handler %s failed for event %s: %s",
                    getattr(handler, "__name__", str(handler)),
                    event.event_type,
                    e,
                    exc_info=True,
                )

    def clear(self) -> None:
        """
        Clear all subscriptions and queued events.

        Useful for resetting the bus between test runs or backtest sessions.
        """
        self._handlers.clear()
        self._queue.clear()
        self._processing = False

    @property
    def subscriber_count(self) -> dict[str, int]:
        """
        Get count of subscribers per event type.

        Returns:
            Dict mapping event type name to subscriber count

        Example:
            ```python
            print(bus.subscriber_count)
            # {'market': 1, 'signal': 1, 'order': 1, 'fill': 1}
            ```
        """
        return {k.value: len(v) for k, v in self._handlers.items() if v}
