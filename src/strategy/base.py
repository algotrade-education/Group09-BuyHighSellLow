"""
Abstract base classes for all trading strategies.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.data.indicators.registry import IndicatorRegistry
from src.strategy.signal import TradeSignal

# --- Position Snapshot ---
# Engine pass object into strategy.generate_signal() with current position info.
# This allows strategy to make informed decisions based on current position, e.g.:
# - If already long, maybe only look for exit signals, not new long entries.
# - If already short, maybe only look for exit signals, not new short entries.
# - If flat, look for both long and short entries.


@dataclass(frozen=True)
class PositionSnapshot:
    """
    Read-only snapshot of current position.

    Invariants:
        - Exactly one of is_flat, is_long, is_short must be True
        - If is_flat, quantity must be 0
        - If is_long or is_short, quantity must be > 0
    """

    is_flat: bool
    is_long: bool
    is_short: bool
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float

    def __post_init__(self) -> None:
        """Validate position invariants."""
        # Check exactly one direction flag is True
        flags = [self.is_flat, self.is_long, self.is_short]
        if sum(flags) != 1:
            raise ValueError(
                f"Exactly one of is_flat/is_long/is_short must be True, got: "
                f"flat={self.is_flat}, long={self.is_long}, short={self.is_short}"
            )

        # Check quantity consistency
        if self.is_flat and self.quantity != 0:
            raise ValueError(f"Flat position must have quantity=0, got {self.quantity}")
        if (self.is_long or self.is_short) and self.quantity <= 0:
            raise ValueError(f"Non-flat position must have quantity>0, got {self.quantity}")

    @classmethod
    def flat(cls) -> PositionSnapshot:
        """Create a flat (no position) snapshot."""
        return cls(
            is_flat=True,
            is_long=False,
            is_short=False,
            quantity=0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
        )


# --- Strategy Base Class ---


class StrategyBase(ABC):
    """
    Minimal interface for all trading strategies.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate_signal(
        self, bar: dict[str, Any], position: PositionSnapshot | None = None, is_warmup: bool = False
    ) -> TradeSignal:
        """
        Receive one bar, returns TradeSignal

        Args:
            bar:        Dict with OHLCV and indicator columns (if provided)
                        Keys: "datetime", "open", "high", "low", "close", "volume",
                            "atr_14",...
            position:   Current position snapshot. Returns FLAT if provided None
            is_warmup:  True in warmup period (first N bars).
                        Strategy can use this to skip signal generation
                        until indicators are ready.

        Returns:
            TradeSignal: Signal object with direction, price, SL/TP, etc.
        """
        ...

    @classmethod
    @abstractmethod
    def build_registry(cls, **params: Any) -> IndicatorRegistry:
        """
        Return IndicatorRegistry with indicators required by this strategy.
        This allows engine to precompute all indicators before backtest starts.

        Args:
            params: Strategy parameters (e.g. lookback periods) that may affect which indicators are needed.

        Returns:
            IndicatorRegistry: Registry with all indicators this strategy needs.
        """
        ...

    def reset(self) -> None:
        """
        Reset any internal state of the strategy.
        Called by engine before each backtest or paper trading session.
        Default no-op, override if your strategy has state to reset.
        """
        return None

    def save_state(self) -> dict[str, Any]:
        """
        Return a dict representing the current state of the strategy.
        This can be used for checkpointing during backtests or paper trading.
        Default returns empty dict, override if your strategy has state to save.
        """
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        """
        Load strategy state from a dict.
        This can be used for checkpointing during backtests or paper trading.
        Default no-op, override if your strategy has state to load.
        """
        return None

    @staticmethod
    def validate_bar(
        bar: dict[str, Any],
        required_fields: list[str],
        raise_on_error: bool = False,
    ) -> bool:
        """
        Validate that the input bar has all required keys and valid values.

        Args:
            bar: Bar dictionary to validate
            required_fields: List of required field names
            raise_on_error: If True, raise ValueError on validation failure

        Returns:
            True if valid, False otherwise (when raise_on_error=False)

        Raises:
            ValueError: If validation fails and raise_on_error=True
        """
        for field in required_fields:
            if field not in bar:
                if raise_on_error:
                    raise ValueError(f"Missing required field: {field}")
                return False

            val = bar[field]
            if val is None:
                if raise_on_error:
                    raise ValueError(f"Field {field} is None")
                return False

            # Check numeric fields are not NaN/inf and are positive
            if field in ("open", "high", "low", "close", "volume"):
                try:
                    fval = float(val)
                    if math.isnan(fval) or math.isinf(fval):
                        if raise_on_error:
                            raise ValueError(f"Field {field} is NaN or Inf: {fval}")
                        return False
                    if fval <= 0:
                        if raise_on_error:
                            raise ValueError(f"Field {field} must be positive: {fval}")
                        return False
                except (ValueError, TypeError) as e:
                    if raise_on_error:
                        raise ValueError(f"Field {field} is not numeric: {val}") from e
                    return False

        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
