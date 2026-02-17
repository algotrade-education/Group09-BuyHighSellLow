"""
Sharpe Ratio metric.
"""

from typing import List, Union

import numpy as np
import pandas as pd

from .base import RiskAdjustedMetric


class SharpeRatio(RiskAdjustedMetric):
    """
    Sharpe Ratio - Risk-adjusted return metric.

    Measures excess return per unit of risk (standard deviation).

    Formula:
        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns

    Annualized:
        Sharpe = (Mean Return - Rf/252) / Std * sqrt(252)
    """

    def __init__(
        self,
        annualization_factor: float = 252.0,
        risk_free_rate: float = 0.0,
    ):
        """
        Initialize Sharpe Ratio metric.

        Args:
            annualization_factor: Factor for annualizing (252 for daily data)
            risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
        """
        super().__init__(
            name="Sharpe Ratio",
            annualization_factor=annualization_factor,
            risk_free_rate=risk_free_rate,
        )

    def calculate(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        annualized: bool = True,
    ) -> float:
        """
        Calculate Sharpe Ratio.

        Args:
            returns: Series of returns (daily, not cumulative)
            annualized: Whether to annualize the ratio

        Returns:
            Sharpe Ratio value
        """
        returns = self._to_series(returns).dropna()

        if not self._validate_returns(returns):
            return 0.0

        if returns.std() == 0:
            return 0.0

        # Daily risk-free rate
        daily_rf = self.risk_free_rate / self.annualization_factor

        # Excess returns
        excess_returns = returns - daily_rf

        # Calculate Sharpe
        sharpe = excess_returns.mean() / excess_returns.std()

        if annualized:
            sharpe *= np.sqrt(self.annualization_factor)

        return sharpe


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray, List[float]],
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
) -> float:
    """
    Convenience function to calculate Sharpe Ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        annualization_factor: Annualization factor

    Returns:
        Annualized Sharpe Ratio
    """
    metric = SharpeRatio(
        annualization_factor=annualization_factor,
        risk_free_rate=risk_free_rate,
    )
    return metric.calculate(returns)
