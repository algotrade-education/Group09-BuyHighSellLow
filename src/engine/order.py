"""
Order domain models for the trading engine.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import itertools
from typing import ClassVar, Optional


class OrderType(Enum):
    """Order types supported by the backtester."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderSide(Enum):
    """Order sides supported by the backtester."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order lifecycle statuses supported by the backtester."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    """
    Represents a trading order.

    Attributes:
        order_id: Unique identifier for the order
        order_type: Type of the order (e.g., MARKET, LIMIT)
        side: Side of the order (e.g., BUY, SELL)
        quantity: Quantity of the order
        limit_price: Limit price for LIMIT orders (None for MARKET orders)
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        status: Current status of the order (e.g., PENDING, FILLED)
        created_at: Timestamp when the order was created
        filled_at: Timestamp when the order was filled (optional)
        filled_price: Price at which the order was filled (optional)
        commission: Commission paid for the order
        slippage: Slippage incurred for the order
        reason: Reason for order rejection or cancellation (if applicable)
    """

    _id_counter: ClassVar[itertools.count] = itertools.count(1)

    order_type: OrderType
    side: OrderSide

    order_id: int = field(default_factory=lambda: next(Order._id_counter))
    quantity: int = 1
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None

    commission: float = 0.0
    slippage: float = 0.0
    reason: str = ""

    @classmethod
    def reset_id_counter(cls) -> None:
        """
        Reset the global auto-incrementing order ID counter to 1.
        Used at the start of new backtests to ensure ID predictability.
        """
        cls._id_counter = itertools.count(1)

    @property
    def is_buy(self) -> bool:
        """Return whether this is a buy order."""
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        """Return whether this is a sell order."""
        return self.side == OrderSide.SELL

    @property
    def is_filled(self) -> bool:
        """Return whether this order has been filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_pending(self) -> bool:
        """Return whether this order is pending."""
        return self.status == OrderStatus.PENDING

    @property
    def total_cost(self) -> Optional[float]:
        """
        Calculate the total transaction cost incurred by this order.
        Includes both explicit commission and estimated slippage impact.
        """
        if self.filled_price is None:
            return 0.0

        return self.commission + abs(self.slippage)

    def fill(
        self,
        price: float,
        timestamp: datetime,
        commission: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        """
        Fills the order with the given price and timestamp.

        Args:
            price: Price at which the order is filled
            timestamp: Timestamp when the order is filled
            commission: Commission paid for the order (optional)
            slippage: Slippage incurred for the order (optional)
        """
        self.status = OrderStatus.FILLED
        self.filled_at = timestamp
        self.filled_price = price
        self.commission = commission
        self.slippage = slippage

    def cancel(self, reason: str = "") -> None:
        """
        Cancels the order with an optional reason.

        Args:
            reason: Reason for order cancellation (optional)
        """
        self.status = OrderStatus.CANCELLED
        self.reason = reason

    def reject(self, reason: str = "") -> None:
        """
        Rejects the order with an optional reason.

        Args:
            reason: Reason for order rejection (optional)
        """
        self.status = OrderStatus.REJECTED
        self.reason = reason

    def expire(self) -> None:
        """
        Mark the order as expired.
        Typically used when an order's Time-To-Live (TTL) is exceeded
        before it can be filled.
        """
        self.status = OrderStatus.EXPIRED

    def __repr__(self) -> str:
        price_str = f"@{self.limit_price:.2f}" if self.limit_price else ""
        return (
            f"Order(id={self.order_id}, side={self.side.value}, "
            f"quantity={self.quantity}, type={self.order_type.value}"
            f"{price_str}, status={self.status.value})"
        )
