"""
Abstract base classes for all trading strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.data.indicators.registry import IndicatorRegistry
from strategy.signal import TradeSignal

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
    """

    is_flat: bool
    is_long: bool
    is_short: bool
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float


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

    @abstractmethod
    def reset(self) -> None:
        """
        Reset any internal state of the strategy.
        Called by engine before each backtest or paper trading session.
        Default no-op, override if your strategy has state to reset.
        """
        ...

    @abstractmethod
    def save_state(self) -> dict[str, Any]:
        """
        Return a dict representing the current state of the strategy.
        This can be used for checkpointing during backtests or paper trading.
        Default returns empty dict, override if your strategy has state to save.
        """
        return {}

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        """
        Load strategy state from a dict.
        This can be used for checkpointing during backtests or paper trading.
        Default no-op, override if your strategy has state to load.
        """
        ...

    @staticmethod
    def validate_bar(
        bar: dict[str, Any],
        required_fields: list[str],
    ) -> bool:
        """
        Validate that the input bar has all required keys.
        Raises ValueError if any key is missing.
        """
        for field in required_fields:
            if field not in bar:
                return False
            val = bar[field]
            if val is None:
                return False

            # Check numeric fields are not NaN
            if field in ("open", "high", "low", "close", "volume"):
                try:
                    if float(val) <= 0:
                        return False
                except (ValueError, TypeError):
                    return False

        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
