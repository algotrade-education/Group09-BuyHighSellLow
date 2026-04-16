"""
Simple Moving Average (SMA) indicator.
"""

from __future__ import annotations

from typing import Any

from src.data.indicators.base import IndicatorBase


class SMA(IndicatorBase):
    """
    Simple moving average over a specified period.
    Can be applied to any column (e.g., close, ATR, volume).
    """

    def __init__(self, period: int, source_col: str = "close") -> None:
        """
        Args:
            period: Number of bars for the moving average.
            source_col: Column name to calculate SMA on (default: "close").
        """
        if period < 1:
            raise ValueError(f"SMA period must be >= 1, got {period}")

        super().__init__()
        self._period = period
        self._source_col = source_col
        self._values: list[float] = []
        self._sum = 0.0

        # Set required inputs and warm-up
        self.warm_up_required = period
        self.required_inputs = frozenset({source_col})

    def update(self, **kwargs: Any) -> float | None:
        """
        Update SMA with a new bar.

        Args:
            **kwargs: Bar data containing the source column.

        Returns:
            Current SMA value if ready, else None.
        """
        val = kwargs.get(self._source_col)
        if val is None:
            return self._current_value

        try:
            val = float(val)
        except (TypeError, ValueError):
            return self._current_value

        self._values.append(val)
        self._sum += val
        self._count += 1

        # Remove oldest value if we exceed period
        if len(self._values) > self._period:
            oldest = self._values.pop(0)
            self._sum -= oldest

        # Update current value
        if self.is_ready:
            self._current_value = self._sum / len(self._values)

        return self._current_value

    @property
    def is_ready(self) -> bool:
        """SMA is ready when we have at least period values."""
        return len(self._values) >= self._period

    def _reset_state(self) -> None:
        """Reset indicator state."""
        self._values.clear()
        self._sum = 0.0
        self._count = 0
        self._current_value = None

    def _get_state(self) -> dict:
        """Serialize indicator state."""
        return {
            "class": self.__class__.__name__,
            "period": self._period,
            "source_col": self._source_col,
            "values": self._values.copy(),
            "sum": self._sum,
            "count": self._count,
            "current_value": self._current_value,
        }

    def _set_state(self, state: dict) -> None:
        """Restore indicator state."""
        if state.get("class") != self.__class__.__name__:
            raise ValueError(
                f"State class mismatch: expected {self.__class__.__name__}, got {state.get('class')}"
            )
        self._period = state["period"]
        self._source_col = state["source_col"]
        self._values = state["values"].copy()
        self._sum = state["sum"]
        self._count = state["count"]
        self._current_value = state["current_value"]
