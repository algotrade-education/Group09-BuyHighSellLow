"""
EquityTracker module defines the EquityTracker interface
With a simple implementation, SimpleEquityTracker, for tracking equity over time during backtesting.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class EquityTracker(ABC):
    """Abstract interface for tracking equity during backtesting."""

    @abstractmethod
    def record(
        self,
        timestamp: datetime,
        position: str,
        cash: float,
        equity: float,
        unrealized_pnl: float,
        close_price: float,
        realized_pnl: float = 0.0,
    ) -> None:
        """
        Record equity snapshot at given timestamp.

        Args:
            timestamp: Time of record
            position: Current position side (LONG/SHORT/FLAT)
            cash: Current cash balance
            equity: Total equity (cash + unrealized PnL)
            unrealized_pnl: Unrealized profit/loss
            close_price: Current market price
            realized_pnl: Cumulative realized profit/loss
        """
        ...

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Convert tracked records to DataFrame."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker to initial state."""
        ...

    @abstractmethod
    def get_current_equity(self) -> float | None:
        """Get most recent equity value."""
        ...

    @abstractmethod
    def get_peak_equity(self) -> float | None:
        """Get highest equity value recorded."""
        ...


class SimpleEquityTracker(EquityTracker):
    """
    Simple implementation of EquityTracker to store equity records in memory.

    Using columnar storage (dict of lists) for efficient appends and DataFrame conversion.
    """

    def __init__(self) -> None:
        self._data: dict[str, list] = self._empty_store()
        self._peak_equity: float = 0.0

    def record(
        self,
        timestamp: datetime,
        position: str,
        cash: float,
        equity: float,
        unrealized_pnl: float,
        close_price: float,
        realized_pnl: float = 0.0,
    ) -> None:
        # Validation
        if equity < 0:
            logger.warning("Negative equity %.2f at %s", equity, timestamp)
        if cash < 0:
            logger.warning("Negative cash %.2f at %s", cash, timestamp)

        self._data["datetime"].append(timestamp)
        self._data["position"].append(position)
        self._data["cash"].append(cash)
        self._data["equity"].append(equity)
        self._data["unrealized_pnl"].append(unrealized_pnl)
        self._data["realized_pnl"].append(realized_pnl)
        self._data["close_price"].append(close_price)

        # Track peak
        if equity > self._peak_equity:
            self._peak_equity = equity

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame without setting index (datetime remains a column)."""
        df = pd.DataFrame(self._data)
        return df

    def reset(self) -> None:
        self._data = self._empty_store()
        self._peak_equity = 0.0

    def get_current_equity(self) -> float | None:
        """Get most recent equity value."""
        if len(self._data["equity"]) == 0:
            return None

        return float(self._data["equity"][-1])

    def get_peak_equity(self) -> float | None:
        """Get highest equity value recorded."""
        if self._peak_equity == 0.0:
            return None
        return self._peak_equity

    def get_current_drawdown(self) -> float | None:
        """
        Get current drawdown from peak.

        Returns:
            Drawdown as percentage (e.g., -15.5 for 15.5% drawdown), or None if no data
        """
        current = self.get_current_equity()
        peak = self.get_peak_equity()

        if current is None or peak is None or peak == 0:
            return None

        return ((current - peak) / peak) * 100

    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as pandas Series."""
        df = self.to_dataframe()
        if df.empty:
            return pd.Series(dtype=float)
        return df["equity"]

    def get_drawdown_series(self) -> pd.Series:
        """
        Calculate drawdown series (% from peak).

        Returns:
            Series of drawdown percentages
        """
        equity = self.get_equity_curve()
        if equity.empty:
            return pd.Series(dtype=float)

        peak = equity.expanding().max()
        drawdown = ((equity - peak) / peak) * 100
        return drawdown

    @staticmethod
    def _empty_store() -> dict[str, list]:
        return {
            "datetime": [],
            "position": [],
            "cash": [],
            "equity": [],
            "unrealized_pnl": [],
            "realized_pnl": [],
            "close_price": [],
        }

    def __len__(self) -> int:
        return len(self._data["datetime"])

    def __repr__(self) -> str:
        current = self.get_current_equity()
        peak = self.get_peak_equity()
        dd = self.get_current_drawdown()

        if current is None:
            return "SimpleEquityTracker(records=0)"

        dd_str = f", dd={dd:.1f}%" if dd is not None else ""
        return (
            f"SimpleEquityTracker(records={len(self)}, "
            f"equity={current:.0f}, peak={peak:.0f}{dd_str})"
        )
