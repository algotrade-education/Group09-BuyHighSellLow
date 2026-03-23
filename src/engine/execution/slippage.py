"""
Slippage models for simulating execution price impact.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from src.engine.execution.order import OrderSide

logger = logging.getLogger(__name__)


class SlippageModel(ABC):
    """Abstract base class for slippage models."""

    @abstractmethod
    def calculate(
        self,
        price: float,
        side: OrderSide,
        volume: float = 0.0,
        atr: float = 0.0,
    ) -> tuple[float, float]:
        """
        Calculate execution price after slippage.

        Args:
            price: Original order price
            side: Order side (BUY/SELL)
            volume: Order volume (optional, for volume-based models)
            atr: Average True Range (optional, for volatility-based models)

        Returns:
            Tuple of (execution_price, slippage_amount)
            - execution_price: Price after slippage
            - slippage_amount: Absolute slippage cost (always positive)
        """
        ...


class FixedSlippage(SlippageModel):
    """
    Fixed slippage in price points.

    BUY:  price + points
    SELL: price - points

    Args:
        points: Fixed slippage in price points (default: 0.5)
    """

    def __init__(self, points: float = 0.5) -> None:
        if points < 0:
            raise ValueError(f"Slippage points must be >= 0, got {points}")
        self.points = points

    def calculate(
        self,
        price: float,
        side: OrderSide,
        volume: float = 0.0,
        atr: float = 0.0,
    ) -> tuple[float, float]:
        if price <= 0:
            logger.warning("Invalid price %.2f, returning original", price)
            return price, 0.0

        if side == OrderSide.BUY:
            execution_price = price + self.points
        else:
            execution_price = price - self.points

        return execution_price, self.points


class VolatilityAdjustedSlippage(SlippageModel):
    """
    Slippage scaled by volatility (ATR) - wider in volatile periods.

    BUY: price + ATR * multiplier
    SELL: price - ATR * multiplier
    """

    def __init__(self, atr_multiplier: float = 0.05, fallback_points: float = 0.5) -> None:
        """
        Args:
            atr_multiplier: Multiplier for ATR (default: 0.05 = 5% of ATR)
            fallback_points: Fallback slippage when ATR not available (default: 0.5)
        """

        if atr_multiplier < 0:
            raise ValueError(f"ATR multiplier must be >= 0, got {atr_multiplier}")
        if fallback_points < 0:
            raise ValueError(f"Fallback points must be >= 0, got {fallback_points}")

        self.atr_multiplier = atr_multiplier
        self.fallback_points = fallback_points

    def calculate(
        self,
        price: float,
        side: OrderSide,
        volume: float = 0.0,
        atr: float = 0.0,
    ) -> tuple[float, float]:
        if price <= 0:
            logger.warning("Invalid price %.2f, returning original", price)
            return price, 0.0

        if atr > 0:
            slip = atr * self.atr_multiplier
        else:
            slip = self.fallback_points
            logger.debug("No ATR provided, using fallback slippage %.2f", slip)

        if side == OrderSide.BUY:
            execution_price = price + slip
        else:
            execution_price = price - slip

        return execution_price, slip


class PercentageSlippage(SlippageModel):
    """
    Slippage as percentage of price.

    BUY:  price * (1 + pct)
    SELL: price * (1 - pct)
    """

    def __init__(self, percentage: float = 0.001) -> None:
        """
        Args:
            percentage: Slippage as percentage (e.g., 0.001 = 0.1%)
        """

        if percentage < 0:
            raise ValueError(f"Slippage percentage must be >= 0, got {percentage}")

        if percentage > 0.1:
            logger.warning("Slippage percentage %.2f%% seems high", percentage * 100)

        self.percentage = percentage

    def calculate(
        self,
        price: float,
        side: OrderSide,
        volume: float = 0.0,
        atr: float = 0.0,
    ) -> tuple[float, float]:
        if price <= 0:
            logger.warning("Invalid price %.2f, returning original", price)
            return price, 0.0

        slip = price * self.percentage

        if side == OrderSide.BUY:
            execution_price = price + slip
        else:
            execution_price = price - slip

        return execution_price, slip


class VolumeBasedSlippage(SlippageModel):
    """
    Slippage increases with order volume (market impact).

    slippage = base_points + (volume * volume_multiplier)

    """

    def __init__(self, base_points: float = 0.5, volume_multiplier: float = 0.1) -> None:
        """
        Args:
            base_points: Base slippage in points (default: 0.5)
            volume_multiplier: Additional slippage per unit volume (default: 0.1)
        """
        if base_points < 0:
            raise ValueError(f"Base points must be >= 0, got {base_points}")
        if volume_multiplier < 0:
            raise ValueError(f"Volume multiplier must be >= 0, got {volume_multiplier}")

        self.base_points = base_points
        self.volume_multiplier = volume_multiplier

    def calculate(
        self,
        price: float,
        side: OrderSide,
        volume: float = 0.0,
        atr: float = 0.0,
    ) -> tuple[float, float]:
        if price <= 0:
            logger.warning("Invalid price %.2f, returning original", price)
            return price, 0.0

        slip = self.base_points + (volume * self.volume_multiplier)

        if side == OrderSide.BUY:
            execution_price = price + slip
        else:
            execution_price = price - slip

        return execution_price, slip


class ZeroSlippage(SlippageModel):
    """
    No slippage - execution price equals order price.
    Useful for testing or idealized scenarios.
    """

    def calculate(
        self,
        price: float,
        side: OrderSide,
        volume: float = 0.0,
        atr: float = 0.0,
    ) -> tuple[float, float]:
        return price, 0.0
