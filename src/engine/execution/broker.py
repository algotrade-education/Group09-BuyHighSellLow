"""
Base broker interface for order execution.

This module defines the abstract broker interface that all broker implementations
must follow. Brokers are responsible for executing orders and emitting fill events.

Broker Types:
    - SimBroker: Simulated execution for backtesting
    - PaperBroker: Real broker connection with paper trading account
    - LiveBroker: Real broker connection with live trading account
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.core.event_bus import EventBus
    from src.engine.core.events import OrderEvent


class BaseBroker(ABC):
    """
    Abstract base class for all broker implementations.

    A broker receives OrderEvents and is responsible for:
        1. Executing the order (simulated or real)
        2. Emitting FillEvent when order is filled
        3. Handling order lifecycle (pending, filled, rejected, etc.)

    Subclasses must implement:
        - on_order(): Handle incoming order events
        - update_bar(): Update current market data (for sim broker)
    """

    def __init__(self, bus: EventBus) -> None:
        """
        Initialize broker.

        Args:
            bus: Event bus for emitting fill events
        """
        self._bus = bus

    @abstractmethod
    def on_order(self, event: OrderEvent) -> None:
        """
        Handle incoming order event.

        This method should:
            1. Validate the order
            2. Execute the order (simulated or real)
            3. Emit FillEvent if order is filled
            4. Handle order rejection/cancellation if needed

        Args:
            event: Order event to execute
        """
        pass

    @abstractmethod
    def update_bar(self, bar: dict) -> None:
        """
        Update current market data.

        For SimBroker: Updates current bar for order execution
        For PaperBroker: May not be needed (uses real-time data)

        Args:
            bar: Current bar OHLCV data
        """
        pass

    def cancel_pending_orders(self) -> None:
        """
        Cancel all pending orders.

        For SimBroker (T+0): No-op since orders execute immediately
        For SimBrokerT1: Clears pending order queue
        For PaperBroker: Sends cancel requests to broker

        Default implementation is no-op. Override if needed.
        """
        return None
