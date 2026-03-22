"""
TradeSignal and Signal enum: common interface between strategy, engine, and paper trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class Signal(StrEnum):
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"
    EXIT = "exit"  # Explicit exit request from strategy (Not SL/TP)


@dataclass
class TradeSignal:
    """
    Output of strategy.generate_signal().

    Fields:
        signal:       Direction or HOLD/EXIT.
        entry_price:  0.0 = market order (fill at open of next bar).
                      > 0 = limit order at specific price.
        stop_loss:    0.0 = no SL (not recommended).
        take_profit:  0.0 = no TP.
        ord_type:     "LIMIT" or "MARKET".
        reason:       Human-readable reason - used for logging and debug.
        metadata:     Extra info for analysis (range size, ATR, etc.)
                      Does not affect execution.
    """

    signal: Signal = Signal.HOLD
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    ord_type: Literal["LIMIT", "MARKET"] = "LIMIT"
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate signal fields."""
        if self.ord_type not in ("LIMIT", "MARKET"):
            raise ValueError(f"ord_type must be 'LIMIT' or 'MARKET', got: {self.ord_type}")

        # Validate prices are non-negative
        if self.entry_price < 0:
            raise ValueError(f"entry_price must be >= 0, got: {self.entry_price}")
        if self.stop_loss < 0:
            raise ValueError(f"stop_loss must be >= 0, got: {self.stop_loss}")
        if self.take_profit < 0:
            raise ValueError(f"take_profit must be >= 0, got: {self.take_profit}")

    # ── Convenience properties ───

    @property
    def is_entry(self) -> bool:
        return self.signal in (Signal.LONG, Signal.SHORT)

    @property
    def is_long(self) -> bool:
        return self.signal == Signal.LONG

    @property
    def is_short(self) -> bool:
        return self.signal == Signal.SHORT

    @property
    def is_hold(self) -> bool:
        return self.signal == Signal.HOLD

    @property
    def is_exit(self) -> bool:
        return self.signal == Signal.EXIT

    def __repr__(self) -> str:
        if self.is_hold:
            reason = f" ({self.reason})" if self.reason else ""
            return f"TradeSignal(HOLD{reason})"

        return (
            f"TradeSignal({self.signal.upper()}, "
            f"entry={self.entry_price}, sl={self.stop_loss:.2f}, tp={self.take_profit:.2f})"
        )
