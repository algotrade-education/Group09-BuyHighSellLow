"""
Event types for event-driven trading architecture.

Event Flow:
    MarketEvent -> [Strategy] -> SignalEvent
    SignalEvent -> [RiskManager] -> OrderEvent
    OrderEvent  -> [Broker] -> FillEvent
    FillEvent   -> [AccountState] -> (state update)

All events are immutable dataclasses - no modifications after emission.
This ensures thread-safety and predictable behavior in the event pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    MARKET = "market"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class BaseEvent:
    timestamp: datetime
    event_type: EventType


@dataclass(frozen=True, slots=True)
class MarketEvent(BaseEvent):
    """
    Market data event emitted when a new bar is available.

    Triggers strategy signal generation based on latest market data.

    Attributes:
        timestamp: Bar timestamp
        bar: OHLCV data with indicators (dict with keys: open, high, low, close, volume, etc.)
        symbol: Trading symbol (default: VN30F1M)
    """

    bar: dict[str, Any]
    symbol: str = "VN30F1M"
    event_type: EventType = field(default=EventType.MARKET, init=False)


@dataclass(frozen=True, slots=True)
class SignalEvent(BaseEvent):
    """
    Trading signal event emitted by strategy after market analysis.

    RiskManager receives this event and decides whether to create an OrderEvent.

    Attributes:
        timestamp: Signal generation time
        signal_type: Signal direction ("long", "short", "hold", "exit")
        entry_price: Desired entry price (0 = market order)
        stop_loss: Stop loss price (0 = no stop)
        take_profit: Take profit price (0 = no target)
        reason: Signal generation reason/logic
        symbol: Trading symbol
        metadata: Additional signal metadata (indicators, confidence, etc.)
    """

    signal_type: str
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    reason: str = ""
    symbol: str = "VN30F1M"
    metadata: dict = field(default_factory=dict)
    event_type: EventType = field(default=EventType.SIGNAL, init=False)


@dataclass(frozen=True, slots=True)
class OrderEvent(BaseEvent):
    """
    Order event emitted by RiskManager after signal validation.

    Broker receives this event and executes the order.

    Attributes:
        timestamp: Order creation time
        order_type: Order type ("market", "limit")
        side: Order side ("buy", "sell")
        quantity: Order quantity (number of contracts)
        limit_price: Limit price for limit orders (None for market orders)
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        symbol: Trading symbol
    """

    order_type: str
    side: str
    quantity: int = 1
    limit_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    symbol: str = "VN30F1M"
    event_type: EventType = field(default=EventType.ORDER, init=False)


@dataclass(frozen=True, slots=True)
class FillEvent(BaseEvent):
    """
    Fill event emitted by broker after order execution.

    AccountState receives this event to update cash, position, and trade records.

    Attributes:
        timestamp: Execution time
        side: Execution side ("buy", "sell")
        quantity: Filled quantity
        fill_price: Actual execution price
        commission: Commission cost
        slippage: Slippage cost (in price points)
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        symbol: Trading symbol
        order_id: Order identifier
    """

    side: str
    quantity: int
    fill_price: float
    commission: float = 0.0
    slippage: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    symbol: str = "VN30F1M"
    order_id: int = 0
    event_type: EventType = field(default=EventType.FILL, init=False)
