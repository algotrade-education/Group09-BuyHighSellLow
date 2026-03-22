"""
Longest Drawdown metric.
"""

from typing import Any

import numpy as np
import pandas as pd

from .base import DrawdownMetric


class LongestDrawdown(DrawdownMetric):
    """
    Longest Drawdown Duration metric.

    Measures the longest period of time the portfolio spent underwater
    (below its previous peak).
    """

    def __init__(self) -> None:
        """Initialize Longest Drawdown metric."""
        super().__init__(
            name="Longest Drawdown",
            description="Longest underwater period",
        )

    def calculate(  # type: ignore
        self,
        equity: pd.Series | np.ndarray | list[float],
        **kwargs: Any,
    ) -> int:
        """
        Calculate longest drawdown duration.

        Args:
            equity: Equity curve values

        Returns:
            Number of periods in longest drawdown
        """
        equity = self._to_series(equity).dropna()

        if len(equity) < 2:
            return 0

        drawdown = self._calculate_drawdown_series(equity)
        periods = self._calculate_underwater_periods(drawdown)

        return max(periods) if periods else 0

    def calculate_all_periods(
        self,
        equity: pd.Series | np.ndarray | list[float],
    ) -> list[int]:
        """
        Get all underwater period lengths.

        Args:
            equity: Equity curve values

        Returns:
            list of underwater period lengths
        """
        equity = self._to_series(equity).dropna()

        if len(equity) < 2:
            return []

        drawdown = self._calculate_drawdown_series(equity)
        return self._calculate_underwater_periods(drawdown)

    def calculate_average_underwater(
        self,
        equity: pd.Series | np.ndarray | list[float],
    ) -> float:
        """
        Calculate average underwater period.

        Args:
            equity: Equity curve values

        Returns:
            Average underwater period length
        """
        periods = self.calculate_all_periods(equity)
        return np.mean(periods) if periods else 0.0

    def calculate_time_underwater(
        self,
        equity: pd.Series | np.ndarray | list[float],
    ) -> float:
        """
        Calculate percentage of time spent underwater.

        Args:
            equity: Equity curve values

        Returns:
            Percentage of time underwater (0-100)
        """
        equity = self._to_series(equity).dropna()

        if len(equity) < 2:
            return 0.0

        drawdown = self._calculate_drawdown_series(equity)
        underwater_periods = (drawdown < 0).sum()

        return float((underwater_periods / len(equity)) * 100)


def calculate_longest_drawdown(
    equity: pd.Series | np.ndarray | list[float],
) -> int:
    """
    Convenience function to calculate longest drawdown duration.

    Args:
        equity: Equity curve values

    Returns:
        Longest drawdown duration in periods
    """
    metric = LongestDrawdown()
    return metric.calculate(equity)
