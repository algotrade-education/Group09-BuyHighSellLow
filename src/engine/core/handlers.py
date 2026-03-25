"""
Event handlers for event-driven backtesting architecture.

Each handler is a focused component that:
    - Subscribes to specific event types
    - Performs domain-specific logic
    - Emits new events or updates state

Handlers are completely decoupled - they don't know about each other.
Communication happens only through the event bus.

Note:
    SimBrokerHandler is deprecated. Use SimBroker from src.engine.execution.sim_broker instead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.engine.core.events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from src.engine.execution.order import Order, OrderSide, OrderType

if TYPE_CHECKING:
    from src.engine.account.account import AccountState
    from src.engine.core.event_bus import EventBus
    from src.strategy.base import StrategyBase

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Core Trading Handlers
# ------------------------------------------------------------------


class StrategyHandler:
    """
    Strategy signal generation handler.

    Receives: MarketEvent
    Emits: SignalEvent

    Equivalent to: signal = strategy.generate_signal(bar, position)

    This handler runs the trading strategy logic and emits trading signals
    based on market data and current position state.
    """

    def __init__(self, strategy: StrategyBase, account: AccountState, bus: EventBus) -> None:
        """
        Initialize strategy handler.

        Args:
            strategy: Trading strategy instance
            account: Account state for position info
            bus: Event bus for emitting signals
        """
        self._strategy = strategy
        self._account = account
        self._bus = bus

    def on_market(self, event: MarketEvent) -> None:
        """
        Handle market data event and generate trading signal.

        Args:
            event: Market event with bar data
        """
        try:
            signal = self._strategy.generate_signal(
                bar=event.bar,
                position=self._account.position_snapshot,
                is_warmup=event.is_warmup,
            )
            if signal.is_hold:
                return

            self._bus.emit(
                SignalEvent(
                    timestamp=event.timestamp,
                    signal_type=signal.signal.value,
                    ord_type=signal.ord_type.lower(),
                    is_warmup=event.is_warmup,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    reason=signal.reason,
                    symbol=event.symbol,
                    metadata=signal.metadata,
                )
            )
        except Exception as e:
            logger.error("StrategyHandler error: %s", e, exc_info=True)


class RiskHandler:
    """
    Risk management and order creation handler.

    Receives: SignalEvent
    Emits: OrderEvent

    Equivalent to: order = account.create_order(signal, bar, timestamp)

    Validates trading signals against risk rules:
        - Daily loss limit enforcement
        - Position state check (no double entries)
        - Margin availability verification
        - Position sizing calculation

    Only emits OrderEvent if all risk checks pass.
    """

    def __init__(self, account: AccountState, bus: EventBus, current_bar: dict[str, Any]) -> None:
        """
        Initialize risk handler.

        Args:
            account: Account state for risk checks
            bus: Event bus for emitting orders
            current_bar: Reference to current bar dict (updated each bar)
        """
        self._account = account
        self._bus = bus
        self._current_bar = current_bar  # Mutable reference - updated externally

    def on_signal(self, event: SignalEvent) -> None:
        """
        Validate signal and emit order if risk checks pass.

        Args:
            event: Signal event from strategy
        """
        # Risk check: daily loss limit
        if self._account.is_daily_loss_hit:
            return

        # Warmup phase: allow strategy state updates, but do not create orders
        if event.is_warmup:
            return

        # EXIT signal: allow while position is open
        if event.signal_type == "exit":
            if not self._account.position.is_flat:
                close_price = self._current_bar.get("close")
                if close_price is not None:
                    self._account.close_position(
                        close_price,
                        event.timestamp,
                        event.reason or "Exit signal",
                    )
            return

        # Risk check: position already open (entry signals only)
        if not self._account.position.is_flat:
            return

        # Calculate position size
        check_price = (
            event.entry_price if event.entry_price > 0 else self._current_bar.get("close", 0)
        )
        quantity = self._account.position_sizer.calculate_size(
            equity=self._account.equity,
            entry_price=check_price,
            stop_loss=event.stop_loss,
            contract_multiplier=self._account.contract_multiplier,
        )

        # Risk check: margin availability
        max_qty = self._account._max_affordable_quantity(check_price)
        if max_qty <= 0 or quantity <= 0:
            return
        quantity = min(quantity, max_qty)

        side = "buy" if event.signal_type == "long" else "sell"

        requested_order_type = (event.ord_type or "limit").lower()

        # Auto-convert to market order when entry_price is 0
        if event.entry_price == 0.0 and event.signal_type in ("long", "short"):
            requested_order_type = "market"

        if requested_order_type not in ("limit", "market"):
            logger.warning("Unknown ord_type '%s', fallback to market", event.ord_type)
            requested_order_type = "market"

        limit_price = (
            event.entry_price if requested_order_type == "limit" and event.entry_price > 0 else None
        )

        self._bus.emit(
            OrderEvent(
                timestamp=event.timestamp,
                order_type=requested_order_type,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                stop_loss=event.stop_loss or None,
                take_profit=event.take_profit or None,
                symbol=event.symbol,
            )
        )


class SimBrokerHandler:
    """
    Simulated broker execution handler.

    Receives: OrderEvent
    Emits: FillEvent

    Equivalent to: account.execute_order(order, bar, timestamp)

    Simulates order execution for backtesting:
        - Market orders: filled at bar open
        - Limit orders: filled if price reached within bar range
        - Applies slippage model
        - Calculates commission

    For paper/live trading, replace with PaperBrokerHandler that:
        - Sends FIX messages to broker
        - Awaits execution reports
        - Emits FillEvent when order filled
    """

    def __init__(
        self,
        account: AccountState,
        bus: EventBus,
        current_bar: dict[str, Any],
    ) -> None:
        """
        Initialize simulated broker handler.

        Args:
            account: Account state for slippage/commission models
            bus: Event bus for emitting fills
            current_bar: Reference to current bar dict (updated each bar)
        """
        self._account = account
        self._bus = bus
        self._current_bar = current_bar

    def on_order(self, event: OrderEvent) -> None:
        """
        Simulate order execution and emit fill event.

        Args:
            event: Order event to execute
        """
        bar = self._current_bar
        exec_price = self._determine_price(event, bar)
        if exec_price is None:
            return  # Limit price not reached

        # Apply slippage
        side_enum = OrderSide.BUY if event.side == "buy" else OrderSide.SELL
        exec_price, slippage = self._account.slippage_model.calculate(exec_price, side_enum)

        # Calculate commission
        commission = self._account._calc_commission(exec_price, event.quantity)

        self._bus.emit(
            FillEvent(
                timestamp=event.timestamp,
                side=event.side,
                quantity=event.quantity,
                fill_price=exec_price,
                commission=commission,
                slippage=slippage,
                stop_loss=event.stop_loss,
                take_profit=event.take_profit,
                symbol=event.symbol,
            )
        )

    @staticmethod
    def _determine_price(event: OrderEvent, bar: dict[str, Any]) -> float | None:
        """
        Determine execution price based on order type and bar data.

        Args:
            event: Order event
            bar: Current bar OHLC data

        Returns:
            Execution price if order can be filled, None otherwise
        """
        if event.order_type == "market":
            return float(bar["open"])

        lp = event.limit_price
        if lp is None:
            return None

        # Buy limit: filled if bar low <= limit price
        if event.side == "buy":
            if bar["low"] > lp:
                return None  # Price didn't reach limit
            return bar["open"] if bar["open"] <= lp else lp

        # Sell limit: filled if bar high >= limit price
        if bar["high"] < lp:
            return None  # Price didn't reach limit
        return bar["open"] if bar["open"] >= lp else lp


class AccountHandler:
    """
    Account state update handler.

    Receives: FillEvent
    Emits: Nothing (terminal handler)

    Equivalent to: account._open_position(order, timestamp)

    Updates account state after order execution:
        - Deducts commission from cash
        - Opens position with fill details
        - Creates trade record
    """

    def __init__(self, account: AccountState) -> None:
        """
        Initialize account handler.

        Args:
            account: Account state to update
        """
        self._account = account

    def on_fill(self, event: FillEvent) -> None:
        """
        Update account state with fill details.

        Args:
            event: Fill event from broker
        """
        # Create synthetic Order to reuse _open_position logic
        side = OrderSide.BUY if event.side == "buy" else OrderSide.SELL
        order = Order(
            order_type=OrderType.MARKET,
            side=side,
            quantity=event.quantity,
            stop_loss=event.stop_loss,
            take_profit=event.take_profit,
            symbol=event.symbol,
        )
        order.fill(
            price=event.fill_price,
            timestamp=event.timestamp,
            commission=event.commission,
            slippage=event.slippage,
        )

        # Update account
        self._account.cash -= event.commission
        self._account._open_position(order, event.timestamp)
