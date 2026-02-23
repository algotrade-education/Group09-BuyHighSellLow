"""
Simple Moving Average (SMA) indicator.
"""

from typing import Union

import numpy as np
import pandas as pd


class SMA:
    """
    Simple Moving Average indicator.

    Calculates the arithmetic mean of prices over a specified period.
    """

    def __init__(self, period: int = 20):
        """
        Initialize SMA indicator.

        Args:
            period: Number of periods for the moving average
        """
        if period < 1:
            raise ValueError("Period must be at least 1")
        self.period = period

    def calculate(
        self,
        data: Union[pd.Series, np.ndarray, list],
    ) -> pd.Series:
        """
        Calculate SMA for the given data.

        Args:
            data: Price data (Series, array, or list)

        Returns:
            Series with SMA values
        """
        if isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)

        return data.rolling(window=self.period).mean()

    def calculate_slope(
        self,
        data: Union[pd.Series, np.ndarray, list],
        lookback: int = 1,
    ) -> pd.Series:
        """
        Calculate SMA slope (change from previous period).

        Args:
            data: Price data
            lookback: Number of periods to look back

        Returns:
            Series with slope values (positive = uptrend, negative = downtrend)
        """
        sma = self.calculate(data)
        return sma - sma.shift(lookback)

    def is_uptrend(
        self,
        data: Union[pd.Series, np.ndarray, list],
        lookback: int = 1,
    ) -> pd.Series:
        """
        Check if SMA is in uptrend.

        Args:
            data: Price data
            lookback: Periods to check trend over

        Returns:
            Boolean Series (True = uptrend)
        """
        slope = self.calculate_slope(data, lookback)
        return slope > 0

    def is_downtrend(
        self,
        data: Union[pd.Series, np.ndarray, list],
        lookback: int = 1,
    ) -> pd.Series:
        """
        Check if SMA is in downtrend.

        Args:
            data: Price data
            lookback: Periods to check trend over

        Returns:
            Boolean Series (True = downtrend)
        """
        slope = self.calculate_slope(data, lookback)
        return slope < 0


def calculate_sma_slope(
    data: Union[pd.Series, np.ndarray, list],
    period: int = 20,
    lookback: int = 1,
) -> pd.Series:
    """
    Convenience function to calculate SMA slope.

    Args:
        data: Price data
        period: SMA period
        lookback: Periods to calculate slope over

    Returns:
        Series with slope values
    """
    sma = SMA(period)
    return sma.calculate_slope(data, lookback)
