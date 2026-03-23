"""
Position sizing strategies for determining how many contracts to trade based on equity, risk, and other factors.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PositionSizer(ABC):
    """
    Abstract base class for position sizing strategies.
    Subclasses must implement calculate_size() method.
    """

    @abstractmethod
    def calculate_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float | None = None,
        contract_multiplier: float = 1.0,
        **kwargs: float,
    ) -> int:
        """
        Calculate position size based on given parameters.
        """
        ...


class FixedSizer(PositionSizer):
    """Always returns same number of contracts."""

    def __init__(self, size: int = 1) -> None:
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")

        self.size = size

    def calculate_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float | None = None,
        contract_multiplier: float = 1.0,
        **kwargs: float,
    ) -> int:
        return self.size


class PercentRiskSizer(PositionSizer):
    """
    Size position so SL hit risks a fixed % of equity.
    Formula: size = (equity * risk%) / (|entry - SL| * multiplier)
    """

    def __init__(
        self,
        risk_per_trade_pct: float = 2.0,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        if not 0 < risk_per_trade_pct <= 100:
            raise ValueError(f"Risk % must be (0, 100], got {risk_per_trade_pct}")
        if min_size <= 0:
            raise ValueError(f"min_size ({min_size}) must be positive")
        if max_size < min_size:
            raise ValueError(f"max_size ({max_size}) must be >= min_size ({min_size})")

        self.risk_per_trade_pct = risk_per_trade_pct
        self.min_size = min_size
        self.max_size = max_size

    def calculate_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float | None = None,
        contract_multiplier: float = 1.0,
        **kwargs: float,
    ) -> int:
        if stop_loss is None or stop_loss <= 0:
            logger.debug("No valid stop_loss, returning min_size=%d", self.min_size)
            return self.min_size

        risk_per_contract = abs(entry_price - stop_loss) * contract_multiplier
        if risk_per_contract == 0:
            return self.min_size

        risk_amount = equity * (self.risk_per_trade_pct / 100)
        size = int(risk_amount / risk_per_contract)
        return max(self.min_size, min(size, self.max_size))


class PercentEquitySizer(PositionSizer):
    """Size as % of equity notional."""

    def __init__(
        self,
        equity_pct: float = 10.0,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        self.equity_pct = equity_pct
        self.min_size = min_size
        self.max_size = max_size

    def calculate_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float | None = None,
        contract_multiplier: float = 1.0,
        **kwargs: float,
    ) -> int:
        if entry_price <= 0:
            return self.min_size

        allocation = equity * (self.equity_pct / 100)
        notional = entry_price * contract_multiplier
        size = int(allocation / notional)
        return max(self.min_size, min(size, self.max_size))


class KellySizer(PositionSizer):
    """
    Kelly Criterion position sizing.
    f* = (win_rate * payoff_ratio - loss_rate) / payoff_ratio
    Fractional Kelly (default 0.25) to reduce variance.

    Usage:
        sizer = KellySizer(kelly_fraction=0.25)
        size = sizer.calculate_size(
            equity=100000,
            entry_price=1000,
            win_rate=0.55,  # 55% win rate
            payoff_ratio=1.5,  # avg_win / avg_loss = 1.5
        )

    Args:
        kelly_fraction: Fraction of Kelly to use (0.1-0.5 recommended, default 0.25)
        min_size: Minimum position size
        max_size: Maximum position size
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        if not 0 < kelly_fraction <= 1.0:
            raise ValueError(f"Kelly fraction must be (0, 1], got {kelly_fraction}")

        if kelly_fraction > 0.5:
            logger.warning("Kelly fraction %.2f > 0.5 may lead to high variance", kelly_fraction)

        self.kelly_fraction = kelly_fraction
        self.min_size = min_size
        self.max_size = max_size

    def calculate_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float | None = None,
        contract_multiplier: float = 1.0,
        win_rate: float = 0.0,
        payoff_ratio: float = 0.0,
        **kwargs: float,
    ) -> int:
        """
        Calculate position size using Kelly Criterion.

        Args:
            equity: Current account equity
            entry_price: Entry price
            stop_loss: Not used by Kelly sizer
            contract_multiplier: Contract multiplier
            win_rate: Historical win rate (0-1), e.g., 0.55 for 55%
            payoff_ratio: avg_win / avg_loss, e.g., 1.5 means wins are 1.5x losses

        Returns:
            Position size in contracts
        """
        if win_rate <= 0 or payoff_ratio <= 0 or entry_price <= 0:
            logger.debug(
                "Invalid Kelly inputs (win_rate=%.2f, payoff=%.2f), using min_size",
                win_rate,
                payoff_ratio,
            )
            return self.min_size

        loss_rate = 1.0 - win_rate
        kelly_f = (win_rate * payoff_ratio - loss_rate) / payoff_ratio
        kelly_f = max(0.0, kelly_f * self.kelly_fraction)

        allocation = equity * kelly_f
        notional = entry_price * contract_multiplier
        size = int(allocation / notional) if notional > 0 else self.min_size
        size = max(self.min_size, min(size, self.max_size))

        return size


class VolatilityAdjustedSizer(PositionSizer):
    """Reduce size when ATR is high relative to target volatility."""

    def __init__(
        self,
        base_size: int = 1,
        target_volatility_pct: float = 2.0,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        self.base_size = base_size
        self.target_volatility_pct = target_volatility_pct
        self.min_size = min_size
        self.max_size = max_size

    def calculate_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float | None = None,
        contract_multiplier: float = 1.0,
        atr: float | None = None,
        **kwargs: float,
    ) -> int:
        if atr is None or atr <= 0 or entry_price <= 0:
            return self.base_size

        current_vol_pct = (atr / entry_price) * 100
        adjustment = self.target_volatility_pct / current_vol_pct
        size = int(self.base_size * adjustment)
        return max(self.min_size, min(size, self.max_size))
