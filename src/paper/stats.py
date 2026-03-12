"""
Session Statistics and Performance Reporting.

The `SessionStats` module is responsible for analyzing the results of a live
paper trading session. It aggregates data from the `PositionTracker` and
`Trade` history to produce a comprehensive performance audit.

Reporting Features:
- **Real-time Metrics**: Calculates Sharpe Ratio, Sortino Ratio, and Max
  Drawdown using the `MetricsCalculator`.
- **Trade Analysis**: Computes win rates, average win/loss, and best/worst
  trade outcomes.
- **Capital Tracking**: Displays initial capital, final equity, net P&L,
  and total commissions spent.
- **Artifact Generation**: Saves detailed trade logs and equity curves
  to CSV for post-session analysis.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from src.engine.position import Trade
from src.metrics.metrics import MetricsCalculator

if TYPE_CHECKING:
    from src.paper.position_tracker import PositionTracker

logger = logging.getLogger(__name__)


class SessionStats:
    """
    Performance Analyzer for the PaperTrader Engine.

    This class serves as the 'end-of-session' reporter. It translates the
    raw trade sequence into human-readable performance indicators.

    Attributes:
        _tracker: The PositionTracker instance containing session data.
        _metrics_calc: Internal calculator for risk-adjusted returns.
    """

    def __init__(self, tracker: "PositionTracker"):
        self._tracker = tracker
        self._metrics_calc = MetricsCalculator()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_equity_df(self) -> pd.DataFrame:
        """Build an equity-curve DataFrame from the tracker's snapshots."""
        snapshots = self._tracker.equity_snapshots
        if not snapshots:
            return pd.DataFrame(columns=["datetime", "equity"])
        df = pd.DataFrame(snapshots, columns=["datetime", "equity"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def compute(self) -> dict:
        """
        Compute performance metrics.

        Returns:
            Dictionary of metric name → value (same schema as BacktestResult.metrics).
        """
        equity_df = self._build_equity_df()
        trades: List[Trade] = self._tracker.trades

        if equity_df.empty or len(equity_df) < 2:
            return {}

        try:
            metrics = self._metrics_calc.calculate(equity=equity_df, trades=trades)
            return metrics.to_dict()
        except Exception as exc:
            logger.error("MetricsCalculator failed: %s", exc, exc_info=True)
            return {}

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Pretty-print a session summary to stdout."""
        trades = self._tracker.trades
        closed = [t for t in trades if t.is_closed]

        initial = self._tracker.initial_capital
        final_equity = self._tracker.equity
        total_pnl = sum(t.pnl for t in closed)
        total_commission = sum(t.commission for t in closed)

        print()
        print("=" * 60)
        print("  PAPER TRADING SESSION SUMMARY")
        print("=" * 60)

        # Capital
        pnl_pct = (total_pnl / initial * 100) if initial > 0 else 0.0
        print(f"\n  Initial Capital : {initial:>18,.0f} VND")
        print(f"  Final Equity    : {final_equity:>18,.0f} VND")
        direction = "+" if total_pnl >= 0 else ""
        print(
            f"  Net P&L         : {direction}{total_pnl:>17,.0f} VND  ({direction}{pnl_pct:.2f}%)"
        )
        print(f"  Total Commission: {total_commission:>18,.0f} VND")

        # Trade stats
        winning = [t for t in closed if t.pnl > 0]
        losing = [t for t in closed if t.pnl <= 0]
        win_rate = len(winning) / len(closed) * 100 if closed else 0.0

        print(f"\n  Total Trades    : {len(closed):>10}")
        print(f"  Winning Trades  : {len(winning):>10}  ({win_rate:.1f}%)")
        print(f"  Losing Trades   : {len(losing):>10}")

        if winning:
            avg_win = sum(t.pnl for t in winning) / len(winning)
            best = max(winning, key=lambda t: t.pnl)
            print(f"  Avg Win         : {avg_win:>18,.0f} VND")
            print(f"  Best Trade      : {best.pnl:>18,.0f} VND")
        if losing:
            avg_loss = sum(t.pnl for t in losing) / len(losing)
            worst = min(losing, key=lambda t: t.pnl)
            print(f"  Avg Loss        : {avg_loss:>18,.0f} VND")
            print(f"  Worst Trade     : {worst.pnl:>18,.0f} VND")

        # Rich metrics (Sharpe, drawdown, etc.) if enough data
        metrics = self.compute()
        if metrics:
            print()
            print("  --- Performance Metrics ---")
            metric_labels = {
                "sharpe_ratio": "Sharpe Ratio",
                "sortino_ratio": "Sortino Ratio",
                "max_drawdown_pct": "Max Drawdown (%)",
                "profit_factor": "Profit Factor",
                "total_return_pct": "Total Return (%)",
            }
            for key, label in metric_labels.items():
                val = metrics.get(key)
                if val is not None:
                    print(f"  {label:<22}: {val:>10.4f}")

        print()
        print("=" * 60)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, output_dir: Optional[str] = None) -> Path:
        """
        Save trade log and equity curve CSVs.

        Args:
            output_dir: Directory path. If None, auto-generates under 'results/'.

        Returns:
            Path to the output directory.
        """
        if output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/paper_{ts}"

        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Trades CSV
        trades = self._tracker.trades
        if trades:
            rows = [
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
                    "duration_s": t.duration,
                }
                for t in trades
            ]
            trades_path = run_dir / "trades.csv"
            pd.DataFrame(rows).to_csv(trades_path, index=False)
            logger.info("Trades saved to: %s", trades_path)

        # Equity curve CSV
        equity_df = self._build_equity_df()
        if not equity_df.empty:
            equity_path = run_dir / "equity_curve.csv"
            equity_df.to_csv(equity_path, index=False)
            logger.info("Equity curve saved to: %s", equity_path)

        return run_dir
