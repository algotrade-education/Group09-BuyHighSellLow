"""
Base classes for performance metrics.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Union

import numpy as np
import pandas as pd


class Metric(ABC):
    """
    Abstract base class for performance metrics.

    Subclasses must implement the calculate method.
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize metric.

        Args:
            name: Metric name
            description: Metric description
        """
        self.name = name
        self.description = description

    @abstractmethod
    def calculate(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        **kwargs: Any,
    ) -> float:
        """
        Calculate the metric.

        Args:
            returns: Series of returns
            **kwargs: Additional arguments

        Returns:
            Metric value
        """
        pass

    def _to_series(
        self,
        data: Union[pd.Series, np.ndarray, List[float]],
    ) -> pd.Series:
        """Convert input to pandas Series."""
        if isinstance(data, pd.Series):
            return data
        return pd.Series(data)

    def _validate_returns(self, returns: pd.Series) -> bool:
        """Validate returns series."""
        if returns.empty:
            return False
        if returns.isna().all():
            return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class RiskAdjustedMetric(Metric):
    """Base class for risk-adjusted return metrics."""

    def __init__(
        self,
        name: str,
        annualization_factor: float = 252.0,
        risk_free_rate: float = 0.0,
    ):
        """
        Initialize risk-adjusted metric.

        Args:
            name: Metric name
            annualization_factor: Factor for annualizing returns (252 for daily)
            risk_free_rate: Risk-free rate (annualized)
        """
        super().__init__(name)
        self.annualization_factor = annualization_factor
        self.risk_free_rate = risk_free_rate

    def _annualize_return(self, mean_return: float) -> float:
        """Annualize a mean return."""
        return mean_return * self.annualization_factor

    def _annualize_volatility(self, std_return: float) -> float:
        """Annualize volatility (standard deviation)."""
        return std_return * np.sqrt(self.annualization_factor)


class DrawdownMetric(Metric):
    """Base class for drawdown-based metrics."""

    def _calculate_drawdown_series(
        self,
        equity: Union[pd.Series, np.ndarray, List[float]],
    ) -> pd.Series:
        """
        Calculate drawdown series from equity curve.

        Args:
            equity: Equity curve values

        Returns:
            Series of drawdown values (negative percentages)
        """
        equity = self._to_series(equity)
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        return drawdown

    def _calculate_underwater_periods(
        self,
        drawdown: pd.Series,
    ) -> List[int]:
        """
        Calculate lengths of underwater periods.

        Args:
            drawdown: Drawdown series

        Returns:
            List of underwater period lengths
        """
        underwater = drawdown < 0

        periods = []
        current_period = 0

        for is_underwater in underwater:
            if is_underwater:
                current_period += 1
            elif current_period > 0:
                periods.append(current_period)
                current_period = 0

        if current_period > 0:
            periods.append(current_period)

        return periods
