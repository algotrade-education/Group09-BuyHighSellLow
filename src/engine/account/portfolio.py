"""
Portfolio state management - tracks cash, equity, and margin.

Enhanced version with:
- Margin tracking (used vs available)
- Cash constraint validation
- State invariants checking
- Portfolio-level metrics

Separated from AccountState to follow Single Responsibility Principle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.account.position import Position

logger = logging.getLogger(__name__)


class PortfolioState:
    """
    Manages cash, equity, and margin calculations.

    Responsibilities:
    - Track cash balance with validation
    - Calculate equity (cash + unrealized P&L)
    - Track margin usage
    - Validate portfolio state invariants
    - Provide portfolio-level metrics

    Design:
    - Single position only (current system design)
    - Cash can go negative (margin trading)
    - Equity must stay positive (margin call if negative)

    Usage:
        portfolio = PortfolioState(initial_capital=500_000_000, margin_rate=0.18)
        portfolio.deduct_cash(commission)
        portfolio.add_cash(realized_pnl)
        portfolio.update_equity(position)

        # Check state
        if portfolio.is_margin_call:
            # Handle margin call

        # Metrics
        print(f"Leverage: {portfolio.leverage:.2f}x")
        print(f"Margin utilization: {portfolio.margin_utilization:.1f}%")
    """

    def __init__(
        self,
        initial_capital: float,
        margin_rate: float = 0.18,
        contract_multiplier: float = 100_000.0,
    ) -> None:
        """
        Args:
            initial_capital: Starting capital
            margin_rate: Margin requirement as fraction (e.g., 0.18 = 18%)
            contract_multiplier: Contract multiplier for futures
        """
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be > 0, got {initial_capital}")
        if not 0 < margin_rate <= 1:
            raise ValueError(f"margin_rate must be in (0, 1], got {margin_rate}")
        if contract_multiplier <= 0:
            raise ValueError(f"contract_multiplier must be > 0, got {contract_multiplier}")

        self.initial_capital = initial_capital
        self.margin_rate = margin_rate
        self.contract_multiplier = contract_multiplier

        # State
        self.cash: float = initial_capital
        self.equity: float = initial_capital
        self._used_margin: float = 0.0

    def reset(self) -> None:
        """Reset to initial state."""
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self._used_margin = 0.0

    # --- Cash operations ---

    def deduct_cash(self, amount: float) -> None:
        """
        Deduct cash (e.g., commission, margin).

        Cash can go negative in margin trading, but equity must stay positive.

        Args:
            amount: Amount to deduct (must be >= 0)
        """
        if amount < 0:
            raise ValueError(f"Cannot deduct negative amount: {amount}")

        self.cash -= amount

        # Warn if cash goes negative (using margin)
        if self.cash < 0:
            logger.warning(
                "Cash is negative: %.0f (using margin). Equity: %.0f",
                self.cash,
                self.equity,
            )

    def add_cash(self, amount: float) -> None:
        """
        Add cash (e.g., realized P&L).

        Args:
            amount: Amount to add (can be negative for losses)
        """
        self.cash += amount

    # --- Equity and margin ---

    def update_equity(self, position: Position) -> None:
        """
        Update equity based on position's unrealized P&L.

        Equity = Cash + Unrealized P&L

        Also updates used margin based on current position.

        Args:
            position: Current position
        """
        self.equity = self.cash + position.unrealized_pnl

        # Update used margin
        if position.is_flat:
            self._used_margin = 0.0
        else:
            self._used_margin = (
                position.entry_price
                * position.quantity
                * self.contract_multiplier
                * self.margin_rate
            )

        # Check margin call condition
        if self.is_margin_call:
            logger.error(
                "MARGIN CALL: Equity %.0f <= 0. Cash: %.0f, Unrealized P&L: %.0f",
                self.equity,
                self.cash,
                position.unrealized_pnl,
            )

    # --- Properties and metrics ---

    @property
    def available_cash(self) -> float:
        """
        Get available cash for new positions.

        Returns max(0, cash) to prevent negative available cash.
        Used margin is tracked separately.
        """
        return max(0.0, self.cash)

    @property
    def used_margin(self) -> float:
        """Get margin currently used by open position."""
        return self._used_margin

    @property
    def available_margin(self) -> float:
        """
        Get available margin for new positions.

        Available margin = cash - used_margin
        Can be negative if using leverage.
        """
        return self.cash - self._used_margin

    @property
    def total_margin_capacity(self) -> float:
        """
        Get total margin capacity based on equity.

        This is the maximum margin that can be used given current equity.
        Typically equity / margin_rate (e.g., 500M equity / 0.18 = 2.78B capacity).
        """
        if self.margin_rate <= 0:
            return 0.0
        return self.equity / self.margin_rate

    @property
    def margin_utilization(self) -> float:
        """
        Get margin utilization as percentage.

        Returns:
            Percentage of margin capacity being used (0-100+)
            > 100% means over-leveraged (should not happen with proper checks)
        """
        capacity = self.total_margin_capacity
        if capacity <= 0:
            return 0.0
        return (self._used_margin / capacity) * 100.0

    @property
    def leverage(self) -> float:
        """
        Get current leverage ratio.

        Leverage = Position Notional / Equity

        Returns:
            Leverage ratio (e.g., 2.0 = 2x leverage)
            0.0 if no position or equity <= 0
        """
        if self.equity <= 0:
            return 0.0

        # Position notional = used_margin / margin_rate
        if self.margin_rate <= 0:
            return 0.0

        position_notional = self._used_margin / self.margin_rate
        return position_notional / self.equity

    @property
    def is_margin_call(self) -> bool:
        """
        Check if margin call condition is met.

        Margin call occurs when equity <= 0 (losses exceed capital).
        """
        return self.equity <= 0

    @property
    def return_pct(self) -> float:
        """
        Get return percentage from initial capital.

        Returns:
            Return as percentage (e.g., 5.0 = 5% gain)
        """
        if self.initial_capital <= 0:
            return 0.0
        return ((self.equity - self.initial_capital) / self.initial_capital) * 100.0

    # --- Validation ---

    def validate_state(self) -> list[str]:
        """
        Validate portfolio state invariants.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check equity calculation
        # Note: We can't validate equity = cash + unrealized_pnl here
        # because we don't have access to position

        # Check margin consistency
        if self._used_margin < 0:
            errors.append(f"Used margin cannot be negative: {self._used_margin}")

        # Check margin call
        if self.is_margin_call:
            errors.append(f"Margin call: equity {self.equity:.0f} <= 0")

        # Check extreme leverage
        if self.leverage > 10.0:
            errors.append(f"Extreme leverage: {self.leverage:.1f}x (> 10x)")

        return errors

    def __repr__(self) -> str:
        return (
            f"PortfolioState(cash={self.cash:.0f}, equity={self.equity:.0f}, "
            f"used_margin={self._used_margin:.0f}, leverage={self.leverage:.2f}x)"
        )
