"""
Backtest results class
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from src.engine.position import Trade


@dataclass
class BacktestResult:
    """
    Container for backtest results

    Attributes:
        trades (List[Trade]): list of executed trades
        equity_curve (pd.DataFrame): equity curve over time
        signals (List[Dict[str, any]]): list of generated trading signals
        parameters (Dict[str, any]): parameters used for the backtest
    """

    trades: List[Trade]
    equity_curve: pd.DataFrame
    signals: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_trades(self) -> int:
        """Total number of trades executed"""
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        """Number of winning trades."""
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def losing_trades(self) -> int:
        """Number of losing trades."""
        return sum(1 for t in self.trades if t.pnl <= 0)

    @property
    def total_pnl(self) -> float:
        """Total P&L from all trades."""
        return sum(t.pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "trade_id": t.trade_id,
                    "side": t.side.value,
                    "entry_time": t.entry_time,
                    "entry_price": t.entry_price,
                    "exit_time": t.exit_time,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason,
                }
                for t in self.trades
            ]
        )
