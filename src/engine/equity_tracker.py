"""
Equity tracking for backtesting.
"""

from abc import ABC, abstractmethod
from datetime import datetime
import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class EquityTracker(ABC):
    """
    Abstract base class for tracking equity during backtesting.
    """

    @abstractmethod
    def record(
        self,
        timestamp: datetime,
        position: str,
        cash: float,
        equity: float,
        unrealized_pnl: float,
        close_price: float,
    ) -> None:
        """
        Record equity information at a given timestamp.

        Args:
            timestamp: The time of the record
            position: The current position (e.g., 'long', 'short', 'flat')
            cash: Current cash balance
            equity: The total equity (Cash + Unrealized P&L)
            unrealized_pnl: The unrealized profit and loss
            close_price: The closing price of the asset
        """
        pass

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the tracked equity records into a pandas DataFrame.

        Returns:
            DataFrame with columns like ['datetime', 'equity', 'cash', etc.]
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the equity tracker to its initial state.
        """
        pass


class SimpleEquityTracker(EquityTracker):
    """
    Simple implementation of the EquityTracker that stores equity information in memory.
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def record(
        self,
        timestamp: datetime,
        position: str,
        cash: float,
        equity: float,
        unrealized_pnl: float,
        close_price: float,
    ) -> None:
        """
        Record equity information at a given timestamp.

        Args:
            timestamp: The time of the record
            position: The current position (e.g., 'long', 'short', 'flat')
            cash: Current cash balance
            equity: The total equity (Cash + Unrealized P&L)
            unrealized_pnl: The unrealized profit and loss
            close_price: The closing price of the asset
        """
        self._records.append(
            {
                "datetime": timestamp,
                "position": position,
                "equity": equity,
                "cash": cash,
                "unrealized_pnl": unrealized_pnl,
                "close_price": close_price,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert internal records list to a pandas DataFrame.

        Returns:
            DataFrame containing the equity curve history.
        """
        return pd.DataFrame(self._records)

    def reset(self) -> None:
        """
        Reset the equity tracker to its initial state.
        """
        self._records = []
