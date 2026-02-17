"""
Information Ratio metric.
"""

from typing import List, Union

import numpy as np
import pandas as pd

from .base import RiskAdjustedMetric


class InformationRatio(RiskAdjustedMetric):
    """
    Information Ratio metric.

    Measures active return (excess over benchmark) per unit of
    tracking error (volatility of active returns).

    Formula:
        IR = (Portfolio Return - Benchmark Return) / Tracking Error
    """

    def __init__(
        self,
        annualization_factor: float = 252.0,
    ):
        """
        Initialize Information Ratio metric.

        Args:
            annualization_factor: Factor for annualizing
        """
        super().__init__(
            name="Information Ratio",
            annualization_factor=annualization_factor,
        )

    def calculate(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        benchmark_returns: Union[pd.Series, np.ndarray, List[float]],
        annualized: bool = True,
    ) -> float:
        """
        Calculate Information Ratio.

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            annualized: Whether to annualize

        Returns:
            Information Ratio value
        """
        returns = self._to_series(returns).dropna()
        benchmark = self._to_series(benchmark_returns).dropna()

        # Align series
        aligned = pd.concat([returns, benchmark], axis=1).dropna()
        if len(aligned) < 2:
            return 0.0

        returns = aligned.iloc[:, 0]
        benchmark = aligned.iloc[:, 1]

        # Calculate active returns
        active_returns = returns - benchmark

        # Calculate tracking error
        tracking_error = active_returns.std()

        if tracking_error == 0:
            return 0.0

        # Calculate IR
        ir = active_returns.mean() / tracking_error

        if annualized:
            ir *= np.sqrt(self.annualization_factor)

        return ir

    def calculate_tracking_error(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        benchmark_returns: Union[pd.Series, np.ndarray, List[float]],
        annualized: bool = True,
    ) -> float:
        """
        Calculate tracking error only.

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            annualized: Whether to annualize

        Returns:
            Tracking error
        """
        returns = self._to_series(returns).dropna()
        benchmark = self._to_series(benchmark_returns).dropna()

        aligned = pd.concat([returns, benchmark], axis=1).dropna()
        if len(aligned) < 2:
            return 0.0

        active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        te = active_returns.std()

        if annualized:
            te *= np.sqrt(self.annualization_factor)

        return te


def calculate_information_ratio(
    returns: Union[pd.Series, np.ndarray, List[float]],
    benchmark_returns: Union[pd.Series, np.ndarray, List[float]],
    annualization_factor: float = 252.0,
) -> float:
    """
    Convenience function to calculate Information Ratio.

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        annualization_factor: Annualization factor

    Returns:
        Annualized Information Ratio
    """
    metric = InformationRatio(annualization_factor=annualization_factor)
    return metric.calculate(returns, benchmark_returns)
