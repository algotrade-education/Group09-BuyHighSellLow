from typing import Any

from src.data.indicators.base import IndicatorBase


class WilderADX(IndicatorBase):
    """
    ADX indicator using Wilder's smoothing method.

    ADX + DI+/DI- using Wilder's smoothing method.
    Depend on WilderATR internally for True Range calculation.
    """

    required_inputs = frozenset({"high", "low", "close"})

    def __init__(self, period: int = 14) -> None:
        super().__init__()
        self.period = period
        # Two-phase seeding:
        #   Phase 1 (bars 1..period+1): Seed smoothed DM+, DM-, TR via SMA.
        #   Phase 2 (bars period+2..period*2): Accumulate DX values to seed ADX via SMA.
        # Total warm-up = period * 2 bars before ADX output is reliable.
        self.warm_up_required = period * 2

        self._prev_high: float | None = None
        self._prev_low: float | None = None
        self._prev_close: float | None = None

        # For smoothing +DM, -DM, TR
        self._dm_plus_values: list[float] = []
        self._dm_minus_values: list[float] = []
        self._tr_values: list[float] = []

        self._smoothed_dm_plus: float | None = None
        self._smoothed_dm_minus: float | None = None
        self._smoothed_tr: float | None = None

        # For DI and DX
        self._di_plus_value: float | None = None
        self._di_minus_value: float | None = None
        self._dx_values: list[float] = []

        self._adx_value: float | None = None

    def update(self, **kwargs: Any) -> float | None:
        high = kwargs.get("high")
        low = kwargs.get("low")
        close = kwargs.get("close")

        if high is None or low is None or close is None:
            raise ValueError("Missing required bar data for ADX calculation")

        self._count += 1

        # First bar: just store values
        if self._prev_high is None:
            self._prev_high = high
            self._prev_low = low
            self._prev_close = close
            return None

        prev_high = self._prev_high
        prev_low = self._prev_low
        prev_close = self._prev_close
        if prev_low is None or prev_close is None:
            self._prev_high = high
            self._prev_low = low
            self._prev_close = close
            return None

        # Calculate directional movement
        up_move = high - prev_high
        down_move = prev_low - low

        dm_plus = up_move if up_move > down_move and up_move > 0 else 0.0
        dm_minus = down_move if down_move > up_move and down_move > 0 else 0.0

        # Calculate True Range
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))

        # Seed phase: collect values for initial SMA
        if self._count <= self.period + 1:
            self._dm_plus_values.append(dm_plus)
            self._dm_minus_values.append(dm_minus)
            self._tr_values.append(tr)

            if self._count == self.period + 1:
                self._smoothed_dm_plus = sum(self._dm_plus_values) / self.period
                self._smoothed_dm_minus = sum(self._dm_minus_values) / self.period
                self._smoothed_tr = sum(self._tr_values) / self.period
        else:
            if (
                self._smoothed_dm_plus is None
                or self._smoothed_dm_minus is None
                or self._smoothed_tr is None
            ):
                self._prev_high = high
                self._prev_low = low
                self._prev_close = close
                return None

            # Wilder's smoothing
            self._smoothed_dm_plus = (
                self._smoothed_dm_plus * (self.period - 1) + dm_plus
            ) / self.period
            self._smoothed_dm_minus = (
                self._smoothed_dm_minus * (self.period - 1) + dm_minus
            ) / self.period
            self._smoothed_tr = (self._smoothed_tr * (self.period - 1) + tr) / self.period

        # Calculate DI+ and DI-
        if (
            self._smoothed_tr is not None
            and self._smoothed_tr > 0
            and self._smoothed_dm_plus is not None
            and self._smoothed_dm_minus is not None
        ):
            self._di_plus_value = 100 * self._smoothed_dm_plus / self._smoothed_tr
            self._di_minus_value = 100 * self._smoothed_dm_minus / self._smoothed_tr

            # Calculate DX
            di_sum = self._di_plus_value + self._di_minus_value
            if di_sum > 0:
                dx = 100 * abs(self._di_plus_value - self._di_minus_value) / di_sum

                # Seed ADX with SMA of DX
                if self._count <= self.warm_up_required:
                    self._dx_values.append(dx)
                    if self._count == self.warm_up_required:
                        self._adx_value = sum(self._dx_values) / self.period
                        self._prev_high = high
                        self._prev_low = low
                        self._prev_close = close
                        return self._set_value(self._adx_value)
                else:
                    # Smooth ADX using Wilder's method
                    if self._adx_value is None:
                        self._prev_high = high
                        self._prev_low = low
                        self._prev_close = close
                        return None

                    self._adx_value = (self._adx_value * (self.period - 1) + dx) / self.period
                    self._prev_high = high
                    self._prev_low = low
                    self._prev_close = close
                    return self._set_value(self._adx_value)

        self._prev_high = high
        self._prev_low = low
        self._prev_close = close
        return None

    @property
    def di_plus(self) -> float | None:
        return self._di_plus_value if self.is_ready else None

    @property
    def di_minus(self) -> float | None:
        return self._di_minus_value if self.is_ready else None

    def _reset_state(self) -> None:
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None
        self._dm_plus_values = []
        self._dm_minus_values = []
        self._tr_values = []
        self._smoothed_dm_plus = None
        self._smoothed_dm_minus = None
        self._smoothed_tr = None
        self._di_plus_value = None
        self._di_minus_value = None
        self._dx_values = []
        self._adx_value = None

    def _get_state(self) -> dict:
        return {
            "period": self.period,
            "prev_high": self._prev_high,
            "prev_low": self._prev_low,
            "prev_close": self._prev_close,
            "dm_plus_values": self._dm_plus_values,
            "dm_minus_values": self._dm_minus_values,
            "tr_values": self._tr_values,
            "smoothed_dm_plus": self._smoothed_dm_plus,
            "smoothed_dm_minus": self._smoothed_dm_minus,
            "smoothed_tr": self._smoothed_tr,
            "di_plus_value": self._di_plus_value,
            "di_minus_value": self._di_minus_value,
            "dx_values": self._dx_values,
            "adx_value": self._adx_value,
        }

    def _set_state(self, state: dict) -> None:
        self.period = state.get("period", 14)
        self._prev_high = state.get("prev_high")
        self._prev_low = state.get("prev_low")
        self._prev_close = state.get("prev_close")
        self._dm_plus_values = state.get("dm_plus_values", [])
        self._dm_minus_values = state.get("dm_minus_values", [])
        self._tr_values = state.get("tr_values", [])
        self._smoothed_dm_plus = state.get("smoothed_dm_plus")
        self._smoothed_dm_minus = state.get("smoothed_dm_minus")
        self._smoothed_tr = state.get("smoothed_tr")
        self._di_plus_value = state.get("di_plus_value")
        self._di_minus_value = state.get("di_minus_value")
        self._dx_values = state.get("dx_values", [])
        self._adx_value = state.get("adx_value")
