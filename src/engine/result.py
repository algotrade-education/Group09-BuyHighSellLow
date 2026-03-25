"""
Backtest result container and serialization.

This module provides BacktestResult dataclass for storing and analyzing
backtest results including trades, equity curve, and performance metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics.trade_metrics import Trade


@dataclass
class BacktestResult:
    """
    Container for complete backtest run results.

    Stores trades, equity curve, metrics, signals, and parameters
    from a single backtest execution.

    Attributes:
        trades: List of closed Trade records
        equity_curve: DataFrame with equity history (datetime, equity, cash, etc.)
        metrics: Dict of performance metrics (Sharpe, drawdown, etc.)
        signals: List of signal records (optional)
        parameters: Strategy parameters used (optional)
    """

    trades: list[Trade]
    equity_curve: pd.DataFrame
    metrics: dict[str, Any]
    signals: list[dict[str, Any]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    # --- Quick access properties ---

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl < 0)

    @property
    def breakeven_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl == 0)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def total_commission(self) -> float:
        return sum(t.commission for t in self.trades)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def avg_win(self) -> float:
        winners = [t.pnl for t in self.trades if t.pnl > 0]
        return sum(winners) / len(winners) if winners else 0.0

    @property
    def avg_loss(self) -> float:
        losers = [t.pnl for t in self.trades if t.pnl < 0]
        return sum(losers) / len(losers) if losers else 0.0

    @property
    def net_profit_factor(self) -> float:
        """Net PnL based profit factor."""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def gross_profit_factor(self) -> float:
        """Gross PnL (before commission) based profit factor."""
        g_profit = sum(t.gross_pnl for t in self.trades if t.gross_pnl > 0)
        g_loss = abs(sum(t.gross_pnl for t in self.trades if t.gross_pnl <= 0))
        if g_loss == 0:
            return float("inf") if g_profit > 0 else 0.0
        return g_profit / g_loss

    # V1 compatibility
    @property
    def profit_factor(self) -> float:
        return self.net_profit_factor

    # --- Conversion ---

    def trades_to_dataframe(self) -> pd.DataFrame:
        """
        Convert trades list to DataFrame.

        Returns:
            DataFrame with trade details, or empty DataFrame if no trades
        """
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
                    "gross_pnl": t.gross_pnl,
                    "commission": t.commission,
                    "pnl": t.pnl,
                    "exit_reason": t.exit_reason,
                    "mae": t.mae,
                    "mfe": t.mfe,
                    "duration_min": t.duration_minutes,
                }
                for t in self.trades
            ]
        )

    def signals_to_dataframe(self) -> pd.DataFrame:
        """
        Convert signals list to DataFrame.

        Returns:
            DataFrame with signal details, or empty DataFrame if no signals
        """
        if not self.signals:
            return pd.DataFrame()
        return pd.DataFrame(self.signals)

    # --- Serialization ---

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        """
        Save result summary to JSON file.

        Note: Equity curve is saved separately via save() method.

        Args:
            path: Output file path
            indent: JSON indentation level (default: 2)
        """

        def _serialize(obj):  # type: ignore[no-untyped-def]
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, float) and (obj != obj or obj == float("inf")):
                return None  # NaN / inf -> null
            return str(obj)

        data = {
            "parameters": self.parameters,
            "metrics": self.metrics,
            "summary": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": self.win_rate,
                "total_pnl": self.total_pnl,
                "total_commission": self.total_commission,
                "net_profit_factor": self.net_profit_factor
                if self.net_profit_factor != float("inf")
                else None,
            },
        }
        Path(path).write_text(
            json.dumps(data, default=_serialize, indent=indent),
            encoding="utf-8",
        )

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        """
        Save all result components to directory.

        Saves:
        - result.json: Metrics and summary
        - trades.parquet: Trade history
        - equity_curve.parquet: Equity curve
        - signals.parquet: Signal history (if available)

        Args:
            output_dir: Output directory path (created if doesn't exist)

        Returns:
            Dict mapping component name to saved file path
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}

        # Metrics + summary
        json_path = out / "result.json"
        self.to_json(json_path)
        paths["json"] = json_path

        # Trades
        trades_df = self.trades_to_dataframe()
        if not trades_df.empty:
            trades_path = out / "trades.csv"
            trades_df.to_csv(trades_path, index=False)
            paths["trades"] = trades_path

        # Equity curve
        if not self.equity_curve.empty:
            equity_path = out / "equity_curve.csv"
            self.equity_curve.to_csv(equity_path, index=False)
            paths["equity_curve"] = equity_path

        # Signals
        signals_df = self.signals_to_dataframe()
        if not signals_df.empty:
            signals_path = out / "signals.csv"
            signals_df.to_csv(signals_path, index=False)
            paths["signals"] = signals_path

        return paths
