from typing import Any

from src.data.indicators.base import IndicatorBase


class WilderATR(IndicatorBase):
    """
    ATR indicator using Wilder's smoothing method.
    Differ from pandas EWM (asymptotic convergence).

    Formula:
        ATR[0] = mean(TR, period) # SMA seed
        ATR[n] = (ATR[n-1] * (period - 1) + TR[n]) / period
    """

    required_inputs = frozenset({"high", "low", "close"})

    def __init__(self, period: int = 14) -> None:
        super().__init__()

        self.period = period
        self.warm_up_required = period
        self._tr_values: list[float] = []
        self._atr_value: float | None = None
        self._prev_close: float | None = None

    def update(self, **kwargs: Any) -> float | None:
        high = kwargs.get("high")
        low = kwargs.get("low")
        close = kwargs.get("close")

        if high is None or low is None or close is None:
            raise ValueError("Missing required bar data for ATR calculation")

        # Calculate True Range
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))

        self._prev_close = close
        self._count += 1

        # Seed phase: collect TR values for initial SMA
        if self._count <= self.period:
            self._tr_values.append(tr)
            if self._count == self.period:
                self._atr_value = sum(self._tr_values) / self.period
                return self._set_value(self._atr_value)
            return None

        # Wilder's smoothing
        if self._atr_value is None:
            return None

        self._atr_value = (self._atr_value * (self.period - 1) + tr) / self.period
        return self._set_value(self._atr_value)

    def _reset_state(self) -> None:
        self._tr_values = []
        self._atr_value = None
        self._prev_close = None

    def _get_state(self) -> dict:
        return {
            "period": self.period,
            "tr_values": self._tr_values,
            "atr_value": self._atr_value,
            "prev_close": self._prev_close,
        }

    def _set_state(self, state: dict) -> None:
        self.period = state.get("period", 14)
        self._tr_values = state.get("tr_values", [])
        self._atr_value = state.get("atr_value")
        self._prev_close = state.get("prev_close")
