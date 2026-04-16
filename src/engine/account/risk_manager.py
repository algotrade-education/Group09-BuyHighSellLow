"""
Risk management - margin checks and daily loss limits.

Separated from AccountState to follow Single Responsibility Principle.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from src.engine.account.position import Position

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages risk limits and margin calculations.

    Responsibilities:
    - Calculate maximum affordable quantity
    - Track daily P&L and loss limits
    - Enforce risk constraints

    Usage:
        risk = RiskManager(margin_rate=0.18, max_daily_loss_pct=2.0)  # 2% daily loss limit
        max_qty = risk.max_affordable_quantity(price, cash, position)
        risk.record_trade_pnl(pnl, equity)
        if risk.is_daily_loss_hit:
            # Stop trading for the day
    """

    def __init__(
        self,
        margin_rate: float,
        contract_multiplier: float,
        max_daily_loss_pct: float = 0.0,
    ) -> None:
        """
        Args:
            margin_rate: Margin requirement as fraction (e.g., 0.18 = 18%)
            contract_multiplier: Contract multiplier for futures
            max_daily_loss_pct: Max daily loss as % of equity (e.g. 2.0 = 2%, 0 = disabled)
        """
        self.margin_rate = margin_rate
        self.contract_multiplier = contract_multiplier
        self.max_daily_loss_pct = max_daily_loss_pct

        self._daily_pnl: float = 0.0
        self._current_date: date | None = None
        self._daily_loss_hit: bool = False

    def reset(self) -> None:
        """Reset daily tracking."""
        self._daily_pnl = 0.0
        self._current_date = None
        self._daily_loss_hit = False

    def update_daily(self, timestamp: datetime) -> None:
        """
        Reset daily P&L tracking when new trading day starts.

        Args:
            timestamp: Current timestamp
        """
        trading_date = timestamp.date() if hasattr(timestamp, "date") else None
        if trading_date and trading_date != self._current_date:
            self._current_date = trading_date
            self._daily_pnl = 0.0
            self._daily_loss_hit = False

    def record_trade_pnl(self, pnl: float, equity: float) -> None:
        """
        Record trade P&L and check daily loss limit.

        Args:
            pnl: Trade P&L (net after commission)
            equity: Current equity
        """
        self._daily_pnl += pnl

        if self.max_daily_loss_pct > 0:
            # max_daily_loss_pct is human % (e.g. 2.0 = 2%)
            limit = -((self.max_daily_loss_pct / 100.0) * equity)
            if self._daily_pnl < limit:
                self._daily_loss_hit = True
                logger.info(
                    "Max daily loss hit: daily_pnl=%.0f, limit=%.0f",
                    self._daily_pnl,
                    limit,
                )

    def max_affordable_quantity(
        self,
        price: float,
        available_cash: float,
        position: Position,
    ) -> int:
        """
        Calculate maximum affordable quantity based on available margin.

        Uses cash instead of equity for conservative margin calculation.
        Equity includes unrealized P&L which can be negative.

        Args:
            price: Entry price
            available_cash: Available cash
            position: Current position

        Returns:
            Maximum quantity that can be afforded
        """
        if price <= 0 or self.margin_rate <= 0:
            return 0

        # Calculate used margin from existing position
        used_margin = 0.0
        if not position.is_flat:
            used_margin = (
                position.entry_price
                * position.quantity
                * self.contract_multiplier
                * self.margin_rate
            )

        # Calculate available margin
        available = max(0.0, available_cash - used_margin)
        margin_per_contract = price * self.contract_multiplier * self.margin_rate

        if margin_per_contract <= 0:
            return 0

        return int(available // margin_per_contract)

    @property
    def is_daily_loss_hit(self) -> bool:
        """Check if daily loss limit has been hit."""
        return self._daily_loss_hit

    @property
    def daily_pnl(self) -> float:
        """Get current daily P&L."""
        return self._daily_pnl
