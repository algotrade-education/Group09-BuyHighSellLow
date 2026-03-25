"""
Order data structure and lifecycle management for trading execution engine.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import ClassVar


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass(slots=True)
class Order:
    """Trading order - immutable after creation except status/fill fields."""

    _id_counter: ClassVar[itertools.count] = itertools.count(1)

    order_type: OrderType
    side: OrderSide

    order_id: int = field(default_factory=lambda: next(Order._id_counter))
    symbol: str = "VN30F1M"
    quantity: int = 1
    limit_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None

    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime | None = None
    filled_at: datetime | None = None
    filled_price: float | None = None

    commission: float = 0.0
    slippage: float = 0.0
    reason: str = ""

    # --- Class methods ---

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._id_counter = itertools.count(1)

    # --- Properties ---

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_pending(self) -> bool:
        return self.status == OrderStatus.PENDING

    # --- Lifecycle ---

    def fill(
        self,
        price: float,
        timestamp: datetime,
        commission: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        """
        Fill the order with execution details.

        Args:
            price: The execution price.
            timestamp: The time of execution.
            commission: The commission cost for this fill.
            slippage: The slippage cost for this fill.
        """
        self.status = OrderStatus.FILLED
        self.filled_at = timestamp
        self.filled_price = price
        self.commission = commission
        self.slippage = slippage

    def cancel(self, reason: str = "") -> None:
        """
        Cancel the order.

        Args:
            reason: The reason for cancellation.
        """
        self.status = OrderStatus.CANCELLED
        self.reason = reason

    def reject(self, reason: str = "") -> None:
        """
        Reject the order.

        Args:
            reason: The reason for rejection.
        """
        self.status = OrderStatus.REJECTED
        self.reason = reason

    def expire(self) -> None:
        self.status = OrderStatus.EXPIRED

    def __repr__(self) -> str:
        price_str = f"@{self.limit_price:.2f}" if self.limit_price else ""
        return (
            f"Order(id={self.order_id}, {self.symbol}, {self.side}/{self.order_type}"
            f"{price_str}, qty={self.quantity}, {self.status})"
        )
