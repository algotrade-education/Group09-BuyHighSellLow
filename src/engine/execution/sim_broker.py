"""
Simulated broker for backtesting.

This module provides SimBroker class that simulates order execution
for backtesting purposes. Orders are filled based on bar OHLC data
without any real broker connection.

Execution Logic:
    - Market orders: Filled at bar open
    - Limit orders: Filled if price reached within bar range
    - Applies slippage model
    - Calculates commission
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.engine.core.events import FillEvent
from src.engine.execution.broker import BaseBroker
from src.engine.execution.order import OrderSide

if TYPE_CHECKING:
    from src.engine.account.account import AccountState
    from src.engine.core.event_bus import EventBus
    from src.engine.core.events import OrderEvent

logger = logging.getLogger(__name__)


class SimBroker(BaseBroker):
    """
    Simulated broker for backtesting.

    Simulates order execution based on bar OHLC data:
        - Market orders: Filled at bar open price
        - Limit buy: Filled if bar low <= limit price
        - Limit sell: Filled if bar high >= limit price

    Features:
        - Realistic fill price determination
        - Slippage application
        - Commission calculation
        - No look-ahead bias

    Usage:
        ```python
        broker = SimBroker(account, bus)

        # Update bar each iteration
        broker.update_bar(bar)

        # Broker automatically handles OrderEvents via event bus
        bus.subscribe(EventType.ORDER, broker.on_order)
        ```

    Note:
        This is for backtesting only. For paper/live trading,
        use PaperBroker or LiveBroker instead.
    """

    def __init__(self, account: AccountState, bus: EventBus) -> None:
        """
        Initialize simulated broker.

        Args:
            account: Account state for slippage/commission models
            bus: Event bus for emitting fill events
        """
        super().__init__(bus)
        self._account = account
        self._current_bar: dict[str, Any] = {}

    def update_bar(self, bar: dict[str, Any]) -> None:
        """
        Update current bar for order execution.

        This should be called before emitting MarketEvent each bar
        so that orders can be filled against current bar data.

        Args:
            bar: Current bar OHLCV data
        """
        self._current_bar = bar

    def on_order(self, event: OrderEvent) -> None:
        """
        Simulate order execution and emit fill event.

        Execution logic:
            1. Determine fill price based on order type and bar data
            2. Apply slippage model
            3. Calculate commission
            4. Emit FillEvent

        Args:
            event: Order event to execute
        """
        if not self._current_bar:
            logger.warning("No current bar data - cannot execute order")
            return

        # Determine fill price
        fill_price = self._get_fill_price(event)
        if fill_price is None:
            # Limit price not reached
            return

        # Apply slippage
        side_enum = OrderSide.BUY if event.side == "buy" else OrderSide.SELL
        fill_price, slippage = self._account.slippage_model.calculate(fill_price, side_enum)

        # Calculate commission
        commission = self._calc_commission(fill_price, event.quantity)

        # Emit fill event
        self._bus.emit(
            FillEvent(
                timestamp=event.timestamp,
                side=event.side,
                quantity=event.quantity,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage,
                stop_loss=event.stop_loss,
                take_profit=event.take_profit,
                symbol=event.symbol,
            )
        )

        logger.debug(
            "Order filled: %s %d @ %.2f (slippage=%.2f, commission=%.0f)",
            event.side,
            event.quantity,
            fill_price,
            slippage,
            commission,
        )

    def _get_fill_price(self, event: OrderEvent) -> float | None:
        """
        Determine fill price based on order type and bar data.

        Logic:
            - Market order: Fill at bar open
            - Limit buy: Fill if bar low <= limit price
                - If open <= limit: Fill at open
                - Else: Fill at limit price
            - Limit sell: Fill if bar high >= limit price
                - If open >= limit: Fill at open
                - Else: Fill at limit price

        Args:
            event: Order event

        Returns:
            Fill price if order can be filled, None if limit not reached
        """
        bar = self._current_bar

        # Market order: fill at open
        if event.order_type == "market":
            return float(bar["open"])

        # Limit order
        limit_price = event.limit_price
        if limit_price is None:
            logger.error("Limit order missing limit_price: %s", event)
            return None

        if event.side == "buy":
            # Buy limit: fill if bar low <= limit price
            if bar["low"] > limit_price:
                return None  # Price didn't reach limit

            # If opened below limit, fill at open; otherwise fill at limit
            return bar["open"] if bar["open"] <= limit_price else limit_price

        else:  # sell
            # Sell limit: fill if bar high >= limit price
            if bar["high"] < limit_price:
                return None  # Price didn't reach limit

            # If opened above limit, fill at open; otherwise fill at limit
            return bar["open"] if bar["open"] >= limit_price else limit_price

    def _calc_commission(self, price: float, quantity: int) -> float:
        """
        Calculate commission for order execution.

        Args:
            price: Execution price
            quantity: Order quantity

        Returns:
            Commission amount
        """
        notional = price * quantity * self._account.contract_multiplier
        return notional * self._account.commission_rate

    def cancel_pending_orders(self) -> None:
        """
        Cancel all pending orders.

        For SimBroker (T+0), this is a no-op since orders are executed immediately.
        Overridden in SimBrokerT1 to clear pending order queue.
        """
        pass  # No pending orders in T+0 execution


class SimBrokerT1(SimBroker):
    """
    Simulated broker with T+1 execution.

    Orders submitted at bar T are filled at bar T+1 open.
    This simulates more realistic execution where orders
    cannot be filled at the same bar they are submitted.

    Usage:
        ```python
        broker = SimBrokerT1(account, bus)

        # Orders are queued and filled next bar
        broker.update_bar(bar)  # Fills pending orders from previous bar
        bus.emit(OrderEvent(...))  # Queued for next bar
        ```
    """

    def __init__(self, account: AccountState, bus: EventBus) -> None:
        """
        Initialize T+1 simulated broker.

        Args:
            account: Account state for slippage/commission models
            bus: Event bus for emitting fill events
        """
        super().__init__(account, bus)
        self._pending_orders: list[OrderEvent] = []

    def on_order(self, event: OrderEvent) -> None:
        """
        Queue order for execution at next bar.

        Args:
            event: Order event to queue
        """
        self._pending_orders.append(event)
        logger.debug("Order queued for T+1 execution: %s", event)

    def update_bar(self, bar: dict[str, Any]) -> None:
        """
        Update bar and execute pending orders.

        Args:
            bar: Current bar OHLCV data
        """
        super().update_bar(bar)

        # Execute pending orders from previous bar
        for order in self._pending_orders:
            super().on_order(order)

        # Clear pending orders
        self._pending_orders.clear()

    def cancel_pending_orders(self) -> None:
        """
        Cancel all pending orders.

        This is useful for EOD close scenarios where we want to
        clear the order queue without executing pending orders.
        """
        if self._pending_orders:
            logger.debug("Cancelling %d pending orders", len(self._pending_orders))
            self._pending_orders.clear()
