"""
Abstract base class for indicators.
All indicators should inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class IndicatorBase(ABC):
    """
    Stateful indicator baseclass.
    Update state for each new bar, and generate signal based on the updated state.

    Subclass needs to implement:
    - warm_up_required: number of bars required to warm up the indicator before the output is reliable
    - required_inputs: frozenset of bar field names needed (e.g., {"close"}, {"high", "low", "close"})
    - update(**kwargs): receive raw bar values (high, low, close, volume, etc.) and update internal state. Return None if in warm-up period, otherwise return the current indicator value.
    - _get_state(): return a dictionary representing the current state of the indicator. This can be used for debugging or for saving state during paper trading.
    - _set_state(): load the indicator state from a dictionary. Should be the inverse of _get_state.
    """

    # Number of bars required to warm up the indicator before the output is reliable.
    # Subclasses should set this to the appropriate number based on the indicator's requirements.
    warm_up_required: int = 0

    # Set of bar field names required by this indicator (e.g., {"close"}, {"high", "low", "close"})
    # Subclasses MUST override this to declare their input requirements
    required_inputs: frozenset[str] = frozenset({"close"})

    def __init__(self) -> None:
        self._count: int = 0  # Number of bars updated so far. Used to track warm-up period.
        self._current_value: float | None = (
            None  # Current indicator value. None if in warm-up period.
        )

    # --- Core Interface ---

    @abstractmethod
    def update(self, **kwargs: Any) -> float | None:
        """
        Receive raw bar values (high, low, close, volume, etc.) and update internal state.

        Returns:
            float: Current indicator value if out of warm-up period.
            None: If still in warm-up period.

        Example:
            atr.update(high=100, low=90, close=95, volume=1000)
        """
        pass

    @property
    def value(self) -> float | None:
        """
        Return the current indicator value.

        Returns None if still in warm-up period.
        Subclasses can override if they need custom behavior.
        """
        return self._current_value

    @property
    def is_ready(self) -> bool:
        # True if the indicator has received enough bars to be out of warm-up period.
        return self._count >= self.warm_up_required and self._current_value is not None

    @property
    def bar_count(self) -> int:
        # Return the number of bars that have been updated so far.
        return self._count

    # --- State Management ---

    def reset(self) -> None:
        # Reset internal state to initial conditions.
        # Useful for backtesting or paper trading when you want to restart the strategy.
        self._count = 0
        self._current_value = None
        self._reset_state()

    @abstractmethod
    def _reset_state(self) -> None:
        # Reset any additional state variables specific to the indicator.
        pass

    def save_state(self) -> dict:
        """
        Serialize state to a dictionary.
        This can be used for debugging or for saving state during paper trading.

        Returns:
            dict: A dictionary representing the current state of the indicator.
        """
        return {
            "class": self.__class__.__name__,
            "count": self._count,
            "current_value": self._current_value,
            **self._get_state(),
        }

    def load_state(self, state: dict) -> None:
        """
        Load state from a dictionary. Should be the inverse of save_state.

        Raises:
            ValueError: If the state dictionary does not match the expected format or class.
        """
        if state.get("class") != self.__class__.__name__:
            raise ValueError(
                f"State class {state.get('class')} does not match {self.__class__.__name__}"
            )

        self._count = state.get("count", 0)
        self._current_value = state.get("current_value")
        self._set_state(state)

    @abstractmethod
    def _get_state(self) -> dict:
        # Return a dictionary representing the current state of the indicator. This can be used for debugging or for saving state during paper trading.
        pass

    @abstractmethod
    def _set_state(self, state: dict) -> None:
        # Load the indicator state from a dictionary. Should be the inverse of _get_state.
        pass

    # --- Vectorized Interface ---

    def compute_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute indicator over entire DataFrame at once using numpy.

        Subclasses should override this for significant speedup during optimization.
        Default fallback: runs bar-by-bar update() loop (same as DataPipeline._compute).

        Returns:
            pd.Series aligned with df.index, NaN during warm-up period.
        """
        self.reset()
        results: list[float | None] = []
        for bar in df.to_dict("records"):
            results.append(self.update(**{k: bar[k] for k in self.required_inputs if k in bar}))
        return pd.Series(
            [np.nan if v is None else v for v in results],
            index=df.index,
            dtype=np.float64,
        )

    # --- Helpers ---
    def _set_value(self, value: float) -> float:
        # Helper method for subclasses to set the current indicator value and increment bar count.
        self._current_value = value
        return value

    def __repr__(self) -> str:
        status = f"value={self._current_value:.4f}" if self.is_ready else "warming up"
        return f"<{self.__class__.__name__}({status}, bars={self._count})>"
