from collections import deque
from typing import Any

from src.data.indicators.base import IndicatorBase


class VolumeMA(IndicatorBase):
    """
    Simple moving average of volume over a specified period.
    """

    required_inputs = frozenset({"volume"})

    def __init__(self, period: int = 20) -> None:
        super().__init__()
        self.period = period
        self.warm_up_required = period
        self._buffer: deque[float] = deque(maxlen=period)
        self._sum: float = 0.0

    def update(self, **kwargs: Any) -> float | None:
        volume = kwargs.get("volume")

        if volume is None:
            raise ValueError("Missing required volume data for VolumeMA calculation")

        self._count += 1

        # Add new volume to buffer
        if len(self._buffer) == self.period:
            # Remove oldest value from sum
            self._sum -= self._buffer[0]

        self._buffer.append(volume)
        self._sum += volume

        # Calculate average once we have enough data
        if len(self._buffer) == self.period:
            avg = self._sum / self.period
            return self._set_value(avg)

        return None

    def _reset_state(self) -> None:
        self._buffer.clear()
        self._sum = 0.0

    def _get_state(self) -> dict:
        return {
            "period": self.period,
            "buffer": list(self._buffer),
            "sum": self._sum,
        }

    def _set_state(self, state: dict) -> None:
        self.period = state.get("period", 20)
        buffer_list = state.get("buffer", [])
        self._buffer = deque(buffer_list, maxlen=self.period)
        self._sum = state.get("sum", 0.0)
