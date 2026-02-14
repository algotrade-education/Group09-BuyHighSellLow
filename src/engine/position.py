"""
Position management for the trading engine
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from src.engine.order import Order


class PositionSide(Enum):
    """Position direction."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Trade:
    """
    Represents a completed round-trip trade.

    Attributes:
        trade_id: Unique identifier for the trade
        side: Direction of the trade (LONG or SHORT)
        entry_time: Timestamp when the position was opened
        entry_price: Price at which the position was opened
        exit_time: Timestamp when the position was closed
        exit_price: Price at which the position was closed
        quantity: Quantity traded
        pnl: Profit or loss from the trade
        pnl_pct: Profit or loss as a percentage of the entry price
        commission: Total commission paid for the trade
        exit_reason: Reason for exiting the trade (e.g., "Take Profit", "Stop Loss", "Signal")
    """

    trade_id: int
    side: PositionSide
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: int = 1
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    exit_reason: str = ""

    multiplier = 1.0  # Multiplier for calculating PnL (e.g., contract size for futures)

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_time is not None

    @property
    def is_winner(self) -> bool:
        """Check if trade is profitable."""
        return self.pnl > 0

    @property
    def gross_pnl(self) -> float:
        """Get gross P&L (excluding commission)."""
        return self.pnl + self.commission

    @property
    def duration(self) -> Optional[float]:
        """Get trade duration in seconds."""
        if self.exit_time and self.entry_time:
            return (self.exit_time - self.entry_time).total_seconds()
        return None

    def close(
        self,
        exit_time: datetime,
        exit_price: float,
        commission: float = 0.0,
        exit_reason: str = "",
    ) -> None:
        """
        Close the trade and calculate P&L.

        Args:
            exit_time: Timestamp when the position is closed
            exit_price: Price at which the position is closed
            commission: Total commission paid for the trade
            exit_reason: Reason for exiting the trade (e.g., "Take Profit", "Stop Loss", "Signal")
        """
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.commission = commission
        self.exit_reason = exit_reason

        # Calculate P&L
        if self.side == PositionSide.LONG:
            self.pnl = (
                exit_price - self.entry_price
            ) * self.quantity * self.multiplier - commission
        elif self.side == PositionSide.SHORT:
            self.pnl = (
                self.entry_price - exit_price
            ) * self.quantity * self.multiplier - commission

        # Calculate P&L percentage
        notional = self.entry_price * self.quantity * self.multiplier
        self.pnl_pct = (self.pnl / notional) * 100 if notional > 1e-6 else 0.0


@dataclass
class Position:
    """
    Manages the current trading position state

    Attributes:
        side: Current position side (LONG, SHORT, or FLAT)
        entry_time: Timestamp when the current position was opened
        entry_price: Price at which the current position was opened
        quantity: Quantity of the current position
        stop_loss: Stop loss price for the current position (optional)
        take_profit: Take profit price for the current position (optional)
        unrealized_pnl: Unrealized profit or loss for the current position
    """

    side: PositionSide = PositionSide.FLAT
    entry_time: Optional[datetime] = None
    entry_price: float = 0.0
    quantity: int = 0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0

    multiplier: float = 1.0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.side == PositionSide.FLAT or self.quantity == 0

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == PositionSide.LONG and self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == PositionSide.SHORT and self.quantity > 0

    def open(self, order: Order, timestamp: datetime) -> None:
        """
        Open a new position based on the filled order.

        Args:
            order: The filled order to open the position with
            timestamp: The timestamp when the position is opened
        """
        if not self.is_flat:
            raise ValueError("Cannot open position: already have position")
        if not order.is_filled:
            raise ValueError("Cannot open position: order not filled")

        self.side = PositionSide.LONG if order.is_buy else PositionSide.SHORT
        self.entry_price = order.filled_price or 0.0
        self.quantity = order.quantity
        self.entry_time = timestamp
        self.stop_loss = order.stop_loss
        self.take_profit = order.take_profit

    def close(self) -> None:
        """
        Close the current position and reset all attributes.
        """

        self.side = PositionSide.FLAT
        self.entry_price = 0.0
        self.quantity = 0
        self.entry_time = None
        self.stop_loss = None
        self.take_profit = None
        self.unrealized_pnl = 0.0

    def update_unrealized_pnl(self, current_price: float) -> float:
        """
        Update unrealized P&L based on the current market price.

        Args:
            current_price: Current market price

        Returns:
            Updated unrealized P&L
        """
        if self.is_flat:
            self.unrealized_pnl = 0.0
        elif self.is_long:
            self.unrealized_pnl = (
                (current_price - self.entry_price) * self.quantity * self.multiplier
            )
        else:  # SHORT
            self.unrealized_pnl = (
                (self.entry_price - current_price) * self.quantity * self.multiplier
            )

        return self.unrealized_pnl

    def check_stop_loss(self, current_price: float) -> bool:
        """
        Check if the current price has hit the stop loss level.

        Args:
            current_price: Current market price

        Returns:
            True if stop loss is hit, False otherwise
        """
        if self.stop_loss is None or self.is_flat:
            return False

        if self.is_long:
            return current_price <= self.stop_loss
        else:  # SHORT
            return current_price >= self.stop_loss

    def check_take_profit(self, current_price: float) -> bool:
        """
        Check if the current price has hit the take profit level.

        Args:
            current_price: Current market price

        Returns:
            True if take profit is hit, False otherwise
        """
        if self.take_profit is None or self.is_flat:
            return False

        if self.is_long:
            return current_price >= self.take_profit
        else:  # SHORT
            return current_price <= self.take_profit

    def reset(self) -> None:
        """Reset position state."""
        self.side = PositionSide.FLAT
        self.entry_price = 0.0
        self.quantity = 0
        self.entry_time = None
        self.stop_loss = None
        self.take_profit = None
        self.unrealized_pnl = 0.0
