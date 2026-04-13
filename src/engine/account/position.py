"""
Position state - track open position, unrealized PnL, MAE/MFE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from src.engine.execution.order import Order
from src.strategy.base import PositionSnapshot


class PositionSide(StrEnum):
    """Position side."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Position:
    """
    Current open position state.

    MAE/MFE:
        mae: Maximum Adverse Excursion - price has moved against us compared to entry (points)
        mfe: Maximum Favorable Excursion - profit potential if we closed at the best price (points)
        Both will be reset when we close the position, and updated intrabar by update_unrealized_pnl().
    """

    multiplier: float = 1.0

    # --- State ---
    side: PositionSide = PositionSide.FLAT
    entry_time: datetime | None = None
    entry_price: float = 0.0
    quantity: int = 0
    stop_loss: float | None = None
    take_profit: float | None = None
    unrealized_pnl: float = 0.0

    # --- MAE/MFE tracking ---
    _best_price: float | None = field(default=None, repr=False)
    _worst_price: float | None = field(default=None, repr=False)

    # --- Properties ---

    @property
    def is_flat(self) -> bool:
        return self.side == PositionSide.FLAT or self.quantity == 0

    @property
    def is_long(self) -> bool:
        return self.side == PositionSide.LONG and self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.side == PositionSide.SHORT and self.quantity > 0

    @property
    def mae(self) -> float | None:
        """Maximum Adverse Excursion in points. None if no data available."""
        if self._worst_price is None or self.entry_price == 0:
            return None
        if self.is_long:
            return max(0.0, self.entry_price - self._worst_price)
        if self.is_short:
            return max(0.0, self._worst_price - self.entry_price)
        return None

    @property
    def mfe(self) -> float | None:
        """Maximum Favorable Excursion in points. None if no data available."""
        if self._best_price is None or self.entry_price == 0:
            return None
        if self.is_long:
            return max(0.0, self._best_price - self.entry_price)
        if self.is_short:
            return max(0.0, self.entry_price - self._best_price)
        return None

    # --- Lifecycle ---

    def open(self, order: Order, timestamp: datetime) -> None:
        """
        Open position from filled order.
        Raise error if already have open position or order not filled.

        Args:
            order: The filled order to open position from.
            timestamp: The time of opening the position.

        Raises:
            ValueError: If already have open position or order not filled.
        """
        if not self.is_flat:
            raise ValueError("Cannot open: already have position.")
        if not order.is_filled:
            raise ValueError("Cannot open: order not filled.")

        self.side = PositionSide.LONG if order.is_buy else PositionSide.SHORT
        self.entry_price = order.filled_price or 0.0
        self.quantity = order.quantity
        self.entry_time = timestamp
        self.stop_loss = order.stop_loss
        self.take_profit = order.take_profit
        self.unrealized_pnl = 0.0
        self._best_price = self.entry_price
        self._worst_price = self.entry_price

    def close(self) -> None:
        """Reset to FLAT."""
        self.side = PositionSide.FLAT
        self.entry_price = 0.0
        self.quantity = 0
        self.entry_time = None
        self.stop_loss = None
        self.take_profit = None
        self.unrealized_pnl = 0.0
        self._best_price = None
        self._worst_price = None

    def reset(self) -> None:
        """Alias for close() - use between backtests."""
        self.close()

    # --- Updates ---

    def update_unrealized_pnl(
        self,
        current_price: float,
        high: float | None = None,
        low: float | None = None,
    ) -> float:
        """
        Update unrealized P&L and track MAE/MFE.

        Unrealized P&L is the "floating" profit/loss if we closed the position
        at the current price. It changes every bar as the market moves.

        This is different from realized P&L (in TradeRecorder) which is calculated
        only once when the trade actually closes, using actual execution prices.

        MAE/MFE tracking:
        - MAE (Maximum Adverse Excursion): Worst price against us during the trade
        - MFE (Maximum Favorable Excursion): Best price in our favor during the trade
        - Used for post-trade analysis to optimize stop placement and profit targets

        Args:
            current_price: Current market price (typically close price)
            high: High price of current bar (optional, for better MAE/MFE tracking)
            low: Low price of current bar (optional, for better MAE/MFE tracking)

        Returns:
            Updated unrealized P&L
        """
        if self.is_flat:
            self.unrealized_pnl = 0.0
            return 0.0

        # Update MAE/MFE with intrabar extremes if provided
        # This gives more accurate worst/best price tracking than using only close
        prices_to_check = [current_price]
        if high is not None:
            prices_to_check.append(high)
        if low is not None:
            prices_to_check.append(low)

        for price in prices_to_check:
            if self._best_price is None or price > self._best_price:
                self._best_price = price
            if self._worst_price is None or price < self._worst_price:
                self._worst_price = price

        # Calculate unrealized P&L (mark-to-market)
        if self.is_long:
            self.unrealized_pnl = (
                (current_price - self.entry_price) * self.quantity * self.multiplier
            )
        else:
            self.unrealized_pnl = (
                (self.entry_price - current_price) * self.quantity * self.multiplier
            )

        return self.unrealized_pnl

    # --- SL/TP checks ---

    def check_stop_loss(self, price: float) -> bool:
        if self.stop_loss is None or self.is_flat:
            return False
        return price <= self.stop_loss if self.is_long else price >= self.stop_loss

    def check_take_profit(self, price: float) -> bool:
        if self.take_profit is None or self.is_flat:
            return False
        return price >= self.take_profit if self.is_long else price <= self.take_profit

    def __repr__(self) -> str:
        if self.is_flat:
            return "Position(FLAT)"
        mae_str = f", mae={self.mae:.2f}" if self.mae is not None else ""
        mfe_str = f", mfe={self.mfe:.2f}" if self.mfe is not None else ""
        return (
            f"Position({self.side}, entry={self.entry_price:.2f}, "
            f"qty={self.quantity}, upnl={self.unrealized_pnl:.0f}{mae_str}{mfe_str})"
        )

    def to_snapshot(self) -> PositionSnapshot:
        """Convert to immutable snapshot for strategy signal generation.

        Returns:
            PositionSnapshot with current position state.
        """
        return PositionSnapshot(
            is_flat=self.is_flat,
            is_long=self.is_long,
            is_short=self.is_short,
            quantity=self.quantity,
            entry_price=self.entry_price,
            stop_loss=self.stop_loss or 0.0,
            take_profit=self.take_profit or 0.0,
        )
