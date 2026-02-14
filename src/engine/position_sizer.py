"""
Position sizing strategies for dynamic position management.

Position sizers calculate the optimal number of contracts to trade
based on various risk management approaches.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class PositionSizer(ABC):
    """
    Abstract base class for position sizing strategies.
    """

    @abstractmethod
    def calculate_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        contract_multiplier: float = 1.0,
        **kwargs,
    ) -> int:
        """
        Calculate position size.

        Args:
            equity: Current account equity
            entry_price: Intended entry price
            stop_loss: Stop loss price (if applicable)
            contract_multiplier: Contract multiplier
            **kwargs: Additional parameters

        Returns:
            Number of contracts to trade (minimum 1)
        """
        pass


class FixedSizer(PositionSizer):
    """
    Fixed position size - always trades the same number of contracts.

    Simplest approach, good for consistent testing.
    """

    def __init__(self, size: int = 1):
        """
        Initialize fixed sizer.

        Args:
            size: Number of contracts to trade (default: 1)
        """
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")
        self.size = size

    def calculate_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        contract_multiplier: float = 1.0,
        **kwargs,
    ) -> int:
        """Return fixed size."""
        return self.size


class PercentRiskSizer(PositionSizer):
    """
    Risk-based position sizing.

    Sizes position so that a stop loss hit risks a fixed % of equity.
    Formula: Position Size = (Equity * Risk%) / (|Entry - Stop| * Multiplier)
    """

    def __init__(
        self,
        risk_per_trade_pct: float = 2.0,
        min_size: int = 1,
        max_size: int = 10,
    ):
        """
        Initialize percent risk sizer.

        Args:
            risk_per_trade_pct: Percentage of equity to risk per trade (default: 2%)
            min_size: Minimum position size (default: 1)
            max_size: Maximum position size (default: 10)
        """
        if not 0 < risk_per_trade_pct <= 100:
            raise ValueError(
                f"Risk percentage must be between 0 and 100, got {risk_per_trade_pct}"
            )
        if min_size <= 0:
            raise ValueError(f"Min size must be positive, got {min_size}")
        if max_size < min_size:
            raise ValueError(f"Max size ({max_size}) must be >= min size ({min_size})")

        self.risk_per_trade_pct = risk_per_trade_pct
        self.min_size = min_size
        self.max_size = max_size

    def calculate_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        contract_multiplier: float = 1.0,
        **kwargs,
    ) -> int:
        """
        Calculate position size based on risk percentage.

        If no stop loss provided, returns min_size.
        """
        if stop_loss is None or stop_loss <= 0:
            logger.debug("No stop loss provided, using minimum size")
            return self.min_size

        # Calculate risk per contract
        risk_per_contract = abs(entry_price - stop_loss) * contract_multiplier

        if risk_per_contract == 0:
            logger.warning("Stop loss equals entry price, using minimum size")
            return self.min_size

        # Calculate maximum risk amount
        risk_amount = equity * (self.risk_per_trade_pct / 100)

        # Calculate position size
        size = int(risk_amount / risk_per_contract)

        # Apply limits
        size = max(self.min_size, min(size, self.max_size))

        logger.debug(
            "Risk sizer: Equity=%.2f, Risk=%.2f%%, Size=%d contracts",
            equity,
            self.risk_per_trade_pct,
            size,
        )

        return size


class PercentEquitySizer(PositionSizer):
    """
    Equity-based position sizing.

    Sizes position as a percentage of total equity.
    Formula: Position Size = (Equity * %) / (Entry Price * Multiplier)
    """

    def __init__(
        self,
        equity_pct: float = 10.0,
        min_size: int = 1,
        max_size: int = 10,
    ):
        """
        Initialize percent equity sizer.

        Args:
            equity_pct: Percentage of equity to use per trade (default: 10%)
            min_size: Minimum position size (default: 1)
            max_size: Maximum position size (default: 10)
        """
        if not 0 < equity_pct <= 100:
            raise ValueError(
                f"Equity percentage must be between 0 and 100, got {equity_pct}"
            )
        if min_size <= 0:
            raise ValueError(f"Min size must be positive, got {min_size}")
        if max_size < min_size:
            raise ValueError(f"Max size ({max_size}) must be >= min size ({min_size})")

        self.equity_pct = equity_pct
        self.min_size = min_size
        self.max_size = max_size

    def calculate_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        contract_multiplier: float = 1.0,
        **kwargs,
    ) -> int:
        """Calculate position size based on equity percentage."""
        if entry_price <= 0:
            logger.warning("Invalid entry price, using minimum size")
            return self.min_size

        # Calculate allocation amount
        allocation = equity * (self.equity_pct / 100)

        # Calculate notional value per contract
        notional_per_contract = entry_price * contract_multiplier

        # Calculate position size
        size = int(allocation / notional_per_contract)

        # Apply limits
        size = max(self.min_size, min(size, self.max_size))

        logger.debug(
            "Equity sizer: Equity=%.2f, Allocation=%.2f%%, Size=%d contracts",
            equity,
            self.equity_pct,
            size,
        )

        return size


class VolatilityAdjustedSizer(PositionSizer):
    """
    Volatility-adjusted position sizing.

    Reduces position size in high volatility, increases in low volatility.
    Requires ATR (Average True Range) as input.
    """

    def __init__(
        self,
        base_size: int = 1,
        target_volatility_pct: float = 2.0,
        min_size: int = 1,
        max_size: int = 10,
    ):
        """
        Initialize volatility-adjusted sizer.

        Args:
            base_size: Base position size for normal volatility
            target_volatility_pct: Target volatility as % of price (default: 2%)
            min_size: Minimum position size (default: 1)
            max_size: Maximum position size (default: 10)
        """
        if base_size <= 0:
            raise ValueError(f"Base size must be positive, got {base_size}")
        if target_volatility_pct <= 0:
            raise ValueError(
                f"Target volatility must be positive, got {target_volatility_pct}"
            )

        self.base_size = base_size
        self.target_volatility_pct = target_volatility_pct
        self.min_size = min_size
        self.max_size = max_size

    def calculate_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        contract_multiplier: float = 1.0,
        atr: Optional[float] = None,
        **kwargs,
    ) -> int:
        """
        Calculate position size adjusted for volatility.

        Args:
            atr: Average True Range (required via kwargs)
        """
        if entry_price <= 0:
            logger.warning("Invalid entry price, using base size")
            return self.base_size

        if atr is None or atr <= 0:
            logger.debug("No ATR provided, using base size")
            return self.base_size

        # Calculate current volatility as % of price
        current_vol_pct = (atr / entry_price) * 100

        # Adjust size: if volatility is 2x target, use 0.5x size
        vol_adjustment = self.target_volatility_pct / current_vol_pct
        adjusted_size = int(self.base_size * vol_adjustment)

        # Apply limits
        size = max(self.min_size, min(adjusted_size, self.max_size))

        logger.debug(
            "Volatility sizer: ATR=%.2f, Vol=%.2f%%, Adj=%.2fx, Size=%d",
            atr,
            current_vol_pct,
            vol_adjustment,
            size,
        )

        return size
