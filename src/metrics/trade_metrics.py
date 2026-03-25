"""
Trade metrics module.
Includes Trade dataclass and trade-level metrics calculation.

Includes:
    - Trade dataclass with MAE/MFE fields
    - TradeMetrics: payoff ratio, expectancy, edge ratio
    - Consecutive wins/losses
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class TradeSide(StrEnum):
    LONG = "long"
    SHORT = "short"


@dataclass(slots=True)
class Trade:
    """
    Closed trade record.

    MAE/MFE:
        - mae: Maximum Adverse Excursion - how far price moved against the position (absolute points)
        - mfe: Maximum Favorable Excursion - maximum profit potential reached

        Both are tracked by AccountState while holding a position.
        None if intrabar tracking is unavailable (e.g. daily data).
    """

    # --- Identity ---
    trade_id: str
    symbol: str = "VN30F1M"
    side: TradeSide = TradeSide.LONG

    # --- Entry ---
    entry_time: datetime | None = None
    entry_price: float = 0.0
    quantity: int = 0

    # --- Exit ---
    exit_time: datetime | None = None
    exit_price: float = 0.0
    exit_reason: str = ""

    # --- PnL ---
    gross_pnl: float = 0.0  # Before commission
    commission: float = 0.0
    pnl: float = 0.0  # Net: gross_pnl - commission

    # --- Risk params at entry time ---
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # --- MAE/MFE (optional - requires intrabar tracking) ---
    mae: float | None = None  # Maximum Adverse Excursion (points)
    mfe: float | None = None  # Maximum Favorable Excursion (points)

    # --- Metadata ---
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    # --- Computed properties ---

    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def is_loser(self) -> bool:
        return self.pnl < 0

    @property
    def duration_seconds(self) -> float:
        """Trade duration in seconds. Returns 0.0 if not closed yet."""
        if self.entry_time and self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds()
        return 0.0

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0

    @property
    def r_multiple(self) -> float | None:
        """
        R-multiple = pnl / initial_risk.

        Measures profit/loss in terms of initial risk units.
        For example, R=2 means profit was 2x the initial risk.

        Note: Assumes pnl and risk are in same units (currency).
        For futures, both should include contract multiplier.

        Returns None if stop_loss is not available.
        """
        if self.stop_loss <= 0 or self.entry_price <= 0:
            return None
        initial_risk = abs(self.entry_price - self.stop_loss) * self.quantity
        if initial_risk == 0:
            return None
        return self.pnl / initial_risk

    @property
    def edge_ratio(self) -> float | None:
        """Edge Ratio = MFE / MAE. Returns None if MAE/MFE data is unavailable."""
        if self.mae is None or self.mfe is None:
            return None
        if self.mae == 0:
            return None
        return self.mfe / self.mae

    def __repr__(self) -> str:
        status = "open" if not self.is_closed else (f"pnl={self.pnl:+.0f}")
        return f"Trade({self.trade_id}, {self.side}, entry={self.entry_price}, {status})"


# --- Trade-level metrics ---


def calculate_trade_metrics(trades: Sequence[Trade]) -> dict[str, Any]:
    """
    Calculate trade-level statistics from a list of closed trades.

    Returns a dict with all trade metrics - the caller decides
    which fields to use (backtest report, optimization scoring, etc.).
    """
    closed = [t for t in trades if t.is_closed]
    if not closed:
        return _empty_trade_metrics()

    winners = [t for t in closed if t.pnl > 0]
    losers = [t for t in closed if t.pnl < 0]
    breakeven = [t for t in closed if t.pnl == 0]

    total = len(closed)
    n_win = len(winners)
    n_loss = len(losers)

    win_rate = n_win / total if total > 0 else 0.0

    # PnL stats - net (after commission)
    total_net_profit = sum(t.pnl for t in winners)
    total_net_loss = abs(sum(t.pnl for t in losers))

    # Profit factor - net version (from pnl after commission)
    net_profit_factor = total_net_profit / total_net_loss if total_net_loss > 0 else 0.0

    # Gross profit factor - from gross_pnl before commission
    total_gross_profit = sum(t.gross_pnl for t in winners)
    total_gross_loss = abs(sum(t.gross_pnl for t in losers))
    gross_profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 0.0

    # Avg win/loss use net PnL (after commission)
    avg_win = total_net_profit / n_win if n_win > 0 else 0.0
    avg_loss = total_net_loss / n_loss if n_loss > 0 else 0.0

    # Payoff ratio
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    # Expectancy = win_rate * avg_win - loss_rate * avg_loss
    loss_rate = n_loss / total if total > 0 else 0.0
    expectancy = win_rate * avg_win - loss_rate * avg_loss

    # Consecutive wins/losses
    max_consec_wins, max_consec_losses = _consecutive_stats(closed)

    # Duration
    durations = [t.duration_minutes for t in closed if t.duration_minutes > 0]
    avg_duration_min = sum(durations) / len(durations) if durations else 0.0

    # MAE/MFE
    mae_values = [t.mae for t in closed if t.mae is not None]
    mfe_values = [t.mfe for t in closed if t.mfe is not None]
    avg_mae = sum(mae_values) / len(mae_values) if mae_values else None
    avg_mfe = sum(mfe_values) / len(mfe_values) if mfe_values else None

    edge_ratios = [t.edge_ratio for t in closed if t.edge_ratio is not None]
    avg_edge_ratio = sum(edge_ratios) / len(edge_ratios) if edge_ratios else None

    # Total commission
    total_commission = sum(t.commission for t in closed)
    total_pnl = sum(t.pnl for t in closed)

    return {
        # Counts
        "total_trades": total,
        "winning_trades": n_win,
        "losing_trades": n_loss,
        "breakeven_trades": len(breakeven),
        # Rates
        "win_rate": win_rate * 100,  # percentage
        # PnL
        "total_pnl": total_pnl,
        "total_commission": total_commission,
        "total_net_profit": total_net_profit,  # Sum of winning trades (after commission)
        "total_net_loss": total_net_loss,  # Sum of losing trades (after commission)
        "net_profit_factor": net_profit_factor,
        "gross_profit_factor": gross_profit_factor,
        # Per-trade
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "expectancy": expectancy,
        # Streaks
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        # Duration
        "avg_duration_minutes": avg_duration_min,
        # MAE/MFE
        "avg_mae": avg_mae,
        "avg_mfe": avg_mfe,
        "avg_edge_ratio": avg_edge_ratio,
    }


def _consecutive_stats(trades: list[Trade]) -> tuple[int, int]:
    """
    Calculate max consecutive wins and max consecutive losses.
    Returns (max_wins, max_losses).
    """
    max_wins = max_losses = 0
    cur_wins = cur_losses = 0

    for t in trades:
        if t.pnl > 0:
            cur_wins += 1
            cur_losses = 0
        elif t.pnl < 0:
            cur_losses += 1
            cur_wins = 0
        else:
            cur_wins = cur_losses = 0

        max_wins = max(max_wins, cur_wins)
        max_losses = max(max_losses, cur_losses)

    return max_wins, max_losses


def _empty_trade_metrics() -> dict:
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "breakeven_trades": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "total_commission": 0.0,
        "total_net_profit": 0.0,
        "total_net_loss": 0.0,
        "net_profit_factor": 0.0,
        "gross_profit_factor": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "payoff_ratio": 0.0,
        "expectancy": 0.0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "avg_duration_minutes": 0.0,
        "avg_mae": None,
        "avg_mfe": None,
        "avg_edge_ratio": None,
    }
