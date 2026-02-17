"""
Sortino Ratio metric.
"""

from typing import List, Union

import numpy as np
import pandas as pd

from .base import RiskAdjustedMetric


class SortinoRatio(RiskAdjustedMetric):
    """
    Sortino Ratio - Downside risk-adjusted return metric.

    Similar to Sharpe but uses downside deviation instead of
    standard deviation, focusing only on negative volatility.

    Formula:
        Sortino = (Mean Return - MAR) / Downside Deviation

    Where MAR is Minimum Acceptable Return (often risk-free rate).
    """

    def __init__(
        self,
        annualization_factor: float = 252.0,
        minimum_acceptable_return: float = 0.0,
    ):
        """
        Initialize Sortino Ratio metric.

        Args:
            annualization_factor: Factor for annualizing
            minimum_acceptable_return: MAR (annualized)
        """
        super().__init__(
            name="Sortino Ratio",
            annualization_factor=annualization_factor,
            risk_free_rate=minimum_acceptable_return,
        )
        self.mar = minimum_acceptable_return

    def calculate(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        annualized: bool = True,
    ) -> float:
        """
        Calculate Sortino Ratio.

        Args:
            returns: Series of returns
            annualized: Whether to annualize the ratio

        Returns:
            Sortino Ratio value
        """
        returns = self._to_series(returns).dropna()

        if not self._validate_returns(returns):
            return 0.0

        # Daily MAR
        daily_mar = self.mar / self.annualization_factor

        # Calculate downside deviation (only negative returns)
        downside_returns = returns[returns < daily_mar] - daily_mar

        if len(downside_returns) == 0:
            return float("inf")  # No downside risk

        downside_deviation = np.sqrt((downside_returns**2).mean())

        if downside_deviation == 0:
            return float("inf")

        # Calculate Sortino
        excess_return = returns.mean() - daily_mar
        sortino = excess_return / downside_deviation

        if annualized:
            sortino *= np.sqrt(self.annualization_factor)

        return sortino

    def calculate_downside_deviation(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        annualized: bool = True,
    ) -> float:
        """
        Calculate downside deviation only.

        Args:
            returns: Series of returns
            annualized: Whether to annualize

        Returns:
            Downside deviation
        """
        returns = self._to_series(returns).dropna()
        daily_mar = self.mar / self.annualization_factor

        downside_returns = returns[returns < daily_mar] - daily_mar

        if len(downside_returns) == 0:
            return 0.0

        dd = np.sqrt((downside_returns**2).mean())

        if annualized:
            dd *= np.sqrt(self.annualization_factor)

        return dd


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray, List[float]],
    minimum_acceptable_return: float = 0.0,
    annualization_factor: float = 252.0,
) -> float:
    """
    Convenience function to calculate Sortino Ratio.

    Args:
        returns: Series of returns
        minimum_acceptable_return: MAR (annualized)
        annualization_factor: Annualization factor

    Returns:
        Annualized Sortino Ratio
    """
    metric = SortinoRatio(
        annualization_factor=annualization_factor,
        minimum_acceptable_return=minimum_acceptable_return,
    )
    return metric.calculate(returns)
