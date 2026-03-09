"""
Backtest result models and convenience accessors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from src.engine.position import Trade


@dataclass
class BacktestResult:
    """
    Container for one backtest run result.

    Attributes:
        trades (List[Trade]): List of executed trades.
        equity_curve (pd.DataFrame): Equity curve over time.
        metrics (Dict[str, float]): Performance metrics.
        signals (List[Dict[str, Any]]): List of generated trading signals.
        parameters (Dict[str, Any]): Parameters used for the backtest.
    """

    trades: List[Trade]
    equity_curve: pd.DataFrame
    metrics: Dict[str, float]
    signals: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_trades(self) -> int:
        """Return total number of executed trades."""
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        """Number of winning trades."""
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def losing_trades(self) -> int:
        """Number of losing trades."""
        return sum(1 for t in self.trades if t.pnl < 0)

    @property
    def breakeven_trades(self) -> int:
        """Number of breakeven trades (P&L == 0)."""
        return sum(1 for t in self.trades if t.pnl == 0)

    @property
    def total_pnl(self) -> float:
        """Total P&L from all trades."""
        return sum(t.pnl for t in self.trades)

    @property
    def total_commission(self) -> float:
        """Total commission paid across all trades."""
        return sum(t.commission for t in self.trades)

    @property
    def avg_win(self) -> float:
        """Average profit of winning trades."""
        winners = [t.pnl for t in self.trades if t.pnl > 0]
        return sum(winners) / len(winners) if winners else 0.0

    @property
    def avg_loss(self) -> float:
        """Average loss of losing trades (returned as negative)."""
        losers = [t.pnl for t in self.trades if t.pnl < 0]
        return sum(losers) / len(losers) if losers else 0.0

    @property
    def profit_factor(self) -> float:
        """Ratio of gross profits to gross losses. Returns inf if no losses."""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trade records to a tabular DataFrame."""
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
                    "commission": t.commission,
                    "exit_reason": t.exit_reason,
                }
                for t in self.trades
            ]
        )
