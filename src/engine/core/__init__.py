"""
Core engine components for event-driven architecture.

This module provides foundational components for building event-driven
trading systems including event bus, events, handlers, and backtester.

Components:
    - EventBus: Publish-subscribe event router
    - Events: MarketEvent, SignalEvent, OrderEvent, FillEvent
    - Handlers: StrategyHandler, RiskHandler, SimBrokerHandler, AccountHandler
    - EventDrivenBacktester: Event-driven backtesting engine

Usage:
    ```python
    from src.engine.core import EventDrivenBacktester
    from src.engine.account import AccountState

    strategy = MyStrategy()
    account = AccountState(initial_capital=500_000_000)
    backtester = EventDrivenBacktester(strategy, account)
    result = backtester.run(data)
    ```
"""

from src.engine.core.engine import EventDrivenBacktester
from src.engine.core.event_bus import EventBus
from src.engine.core.events import (
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from src.engine.core.handlers import (
    AccountHandler,
    RiskHandler,
    StrategyHandler,
)
from src.engine.execution.sim_broker import SimBroker, SimBrokerT1

__all__ = [
    # Engine
    "EventDrivenBacktester",
    # Event Bus
    "EventBus",
    # Events
    "EventType",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    # Handlers
    "StrategyHandler",
    "RiskHandler",
    "AccountHandler",
    # Brokers
    "SimBroker",
    "SimBrokerT1",
]
