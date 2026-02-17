"""
Maximum Drawdown metric.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .base import DrawdownMetric


@dataclass
class DrawdownInfo:
    """Information about a drawdown period."""

    max_drawdown: float
    peak_idx: int
    trough_idx: int
    recovery_idx: Optional[int]
    peak_value: float
    trough_value: float
    duration: int
    recovery_duration: Optional[int]


class MaximumDrawdown(DrawdownMetric):
    """
    Maximum Drawdown (MDD) metric.

    Measures the largest peak-to-trough decline in portfolio value.

    Formula:
        MDD = (Trough Value - Peak Value) / Peak Value
    """

    def __init__(self):
        """Initialize Maximum Drawdown metric."""
        super().__init__(
            name="Maximum Drawdown",
            description="Largest peak-to-trough decline",
        )

    def calculate(
        self,
        equity: Union[pd.Series, np.ndarray, List[float]],
        as_percentage: bool = True,
    ) -> float:
        """
        Calculate Maximum Drawdown.

        Args:
            equity: Equity curve values (not returns!)
            as_percentage: Return as percentage (e.g., -15.0 for 15% drawdown)

        Returns:
            Maximum drawdown value (negative number)
        """
        equity = self._to_series(equity).dropna()

        if len(equity) < 2:
            return 0.0

        drawdown = self._calculate_drawdown_series(equity)
        mdd = drawdown.min()

        if as_percentage:
            mdd *= 100

        return mdd

    def calculate_with_info(
        self,
        equity: Union[pd.Series, np.ndarray, List[float]],
    ) -> DrawdownInfo:
        """
        Calculate Maximum Drawdown with additional information.

        Args:
            equity: Equity curve values

        Returns:
            DrawdownInfo with peak, trough, recovery details
        """
        equity = self._to_series(equity).dropna().reset_index(drop=True)

        if len(equity) < 2:
            return DrawdownInfo(
                max_drawdown=0.0,
                peak_idx=0,
                trough_idx=0,
                recovery_idx=None,
                peak_value=equity.iloc[0] if len(equity) > 0 else 0,
                trough_value=equity.iloc[0] if len(equity) > 0 else 0,
                duration=0,
                recovery_duration=None,
            )

        # Calculate drawdown series
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max

        # Find maximum drawdown point
        trough_idx = drawdown.idxmin()
        mdd = drawdown.iloc[trough_idx]

        # Find peak before trough
        peak_idx = equity.iloc[: trough_idx + 1].idxmax()

        # Find recovery point (if any)
        recovery_idx = None
        recovery_duration = None
        peak_value = equity.iloc[peak_idx]

        post_trough = equity.iloc[trough_idx:]
        recovery_mask = post_trough >= peak_value
        if recovery_mask.any():
            recovery_idx = recovery_mask.idxmax()
            recovery_duration = recovery_idx - trough_idx

        return DrawdownInfo(
            max_drawdown=mdd * 100,
            peak_idx=peak_idx,
            trough_idx=trough_idx,
            recovery_idx=recovery_idx,
            peak_value=peak_value,
            trough_value=equity.iloc[trough_idx],
            duration=trough_idx - peak_idx,
            recovery_duration=recovery_duration,
        )

    def calculate_all_drawdowns(
        self,
        equity: Union[pd.Series, np.ndarray, List[float]],
        threshold: float = -5.0,
    ) -> List[DrawdownInfo]:
        """
        Find all significant drawdowns above threshold.

        Args:
            equity: Equity curve values
            threshold: Minimum drawdown percentage to include (e.g., -5.0)

        Returns:
            List of DrawdownInfo for each significant drawdown
        """
        equity = self._to_series(equity).dropna().reset_index(drop=True)

        if len(equity) < 2:
            return []

        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100

        drawdowns = []
        in_drawdown = False
        peak_idx = 0
        trough_idx = 0
        min_dd = 0.0

        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                # Start of new drawdown
                in_drawdown = True
                peak_idx = i - 1 if i > 0 else 0
                min_dd = dd
                trough_idx = i
            elif in_drawdown:
                if dd < min_dd:
                    min_dd = dd
                    trough_idx = i
                if dd == 0:
                    # Recovery
                    if min_dd <= threshold:
                        drawdowns.append(
                            DrawdownInfo(
                                max_drawdown=min_dd,
                                peak_idx=peak_idx,
                                trough_idx=trough_idx,
                                recovery_idx=i,
                                peak_value=equity.iloc[peak_idx],
                                trough_value=equity.iloc[trough_idx],
                                duration=trough_idx - peak_idx,
                                recovery_duration=i - trough_idx,
                            )
                        )
                    in_drawdown = False

        # Handle final unrecovered drawdown
        if in_drawdown and min_dd <= threshold:
            drawdowns.append(
                DrawdownInfo(
                    max_drawdown=min_dd,
                    peak_idx=peak_idx,
                    trough_idx=trough_idx,
                    recovery_idx=None,
                    peak_value=equity.iloc[peak_idx],
                    trough_value=equity.iloc[trough_idx],
                    duration=trough_idx - peak_idx,
                    recovery_duration=None,
                )
            )

        return drawdowns


def calculate_max_drawdown(
    equity: Union[pd.Series, np.ndarray, List[float]],
) -> float:
    """
    Convenience function to calculate Maximum Drawdown.

    Args:
        equity: Equity curve values

    Returns:
        Maximum drawdown as percentage (negative number)
    """
    metric = MaximumDrawdown()
    return metric.calculate(equity)
