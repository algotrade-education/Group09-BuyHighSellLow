"""
Bollinger Bands indicator.
"""

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class BollingerBandsResult:
    """Container for Bollinger Bands calculation results."""

    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    bandwidth: pd.Series
    percent_b: pd.Series


class BollingerBands:
    """
    Bollinger Bands indicator.

    Calculates upper, middle (SMA), and lower bands based on
    standard deviation from the moving average.
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
    ):
        """
        Initialize Bollinger Bands indicator.

        Args:
            period: Period for moving average and standard deviation
            std_dev: Number of standard deviations for bands
        """
        if period < 1:
            raise ValueError("Period must be at least 1")
        if std_dev <= 0:
            raise ValueError("Standard deviation multiplier must be positive")

        self.period = period
        self.std_dev = std_dev

    def calculate(
        self,
        data: Union[pd.Series, np.ndarray, list],
    ) -> BollingerBandsResult:
        """
        Calculate Bollinger Bands.

        Args:
            data: Price data (typically close prices)

        Returns:
            BollingerBandsResult with all band values
        """
        if isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)

        # Calculate middle band (SMA)
        middle = data.rolling(window=self.period).mean()

        # Calculate rolling standard deviation
        rolling_std = data.rolling(window=self.period).std()

        # Calculate upper and lower bands
        upper = middle + (self.std_dev * rolling_std)
        lower = middle - (self.std_dev * rolling_std)

        # Calculate bandwidth (volatility measure)
        bandwidth = (upper - lower) / middle

        # Calculate %B (where price is relative to bands)
        percent_b = (data - lower) / (upper - lower)

        return BollingerBandsResult(
            upper=upper,
            middle=middle,
            lower=lower,
            bandwidth=bandwidth,
            percent_b=percent_b,
        )

    def get_bands(
        self,
        data: Union[pd.Series, np.ndarray, list],
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Get upper, middle, and lower bands as tuple.

        Args:
            data: Price data

        Returns:
            Tuple of (upper, middle, lower) bands
        """
        result = self.calculate(data)
        return result.upper, result.middle, result.lower

    def is_above_upper(
        self,
        price: Union[pd.Series, float],
        data: Union[pd.Series, np.ndarray, list],
    ) -> Union[pd.Series, bool]:
        """
        Check if price is above upper band.

        Args:
            price: Current price(s) to check
            data: Historical price data for band calculation

        Returns:
            Boolean indicating if price is above upper band
        """
        result = self.calculate(data)
        return price > result.upper

    def is_below_lower(
        self,
        price: Union[pd.Series, float],
        data: Union[pd.Series, np.ndarray, list],
    ) -> Union[pd.Series, bool]:
        """
        Check if price is below lower band.

        Args:
            price: Current price(s) to check
            data: Historical price data for band calculation

        Returns:
            Boolean indicating if price is below lower band
        """
        result = self.calculate(data)
        return price < result.lower

    def touches_middle(
        self,
        high: Union[pd.Series, float],
        low: Union[pd.Series, float],
        data: Union[pd.Series, np.ndarray, list],
    ) -> Union[pd.Series, bool]:
        """
        Check if price bar touches middle band.

        Args:
            high: High price(s)
            low: Low price(s)
            data: Historical price data for band calculation

        Returns:
            Boolean indicating if bar touches middle band
        """
        result = self.calculate(data)
        return (low <= result.middle) & (high >= result.middle)
