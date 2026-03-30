"""Session statistics computation and reporting for paper trading.

This module provides the SessionStats class for computing performance metrics
from a paper trading session, with optional benchmark comparison.

V2 improvements:
- Defensive equity deduplication (fixes partial fill bug)
- Parquet output format (consistent with backtester)
- Rich trade data (gross_pnl, duration, MAE/MFE)
- Timestamped output directories
- Better formatted summary output
- Error handling for metrics calculation
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.metrics.metrics import MetricsCalculator

if TYPE_CHECKING:
    from src.paper.account.tracker import Tracker

logger = logging.getLogger(__name__)


class SessionStats:
    """Compute and report session statistics with optional benchmark.

    Handles equity snapshot deduplication and conditional information_ratio
    calculation based on benchmark availability.
    """

    def __init__(
        self,
        tracker: Tracker,
        benchmark_equity: pd.Series | None = None,
        freq_minutes: int = 5,
    ) -> None:
        """Initialize SessionStats.

        Args:
            tracker: Account tracker with equity snapshots and trades.
            benchmark_equity: Optional benchmark equity series for IR/alpha/beta.
            freq_minutes: Bar frequency for annualization (default: 5).
        """
        self._tracker = tracker
        self._benchmark_equity = benchmark_equity
        self._metrics_calc = MetricsCalculator(freq_minutes=freq_minutes)

    def _build_equity_df(self) -> pd.DataFrame:
        """Build equity curve from tracker snapshots with defensive deduplication.

        V2 fix: deduplicate timestamps (V1 bug: partial fills caused
        multiple snapshots per bar → duplicate datetime index → Sharpe error).

        Returns:
            DataFrame with columns: datetime, equity
        """
        snapshots = self._tracker.equity_snapshots

        if not snapshots:
            return pd.DataFrame(columns=["datetime", "equity"])

        # Convert to DataFrame
        df = pd.DataFrame(snapshots, columns=["datetime", "equity"])
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Defensive deduplication: keep last snapshot per timestamp
        df = df.drop_duplicates(subset=["datetime"], keep="last")
        df = df.sort_values("datetime").reset_index(drop=True)

        return df

    def compute(self) -> dict:
        """Compute session performance metrics.

        Deduplicates equity snapshots by timestamp before passing to
        MetricsCalculator. Includes information_ratio only when benchmark
        is provided.

        Returns:
            Dictionary of performance metrics. Keys include:
            - total_return_pct, annualized_return_pct, cagr_pct, volatility_pct
            - sharpe_ratio, sortino_ratio
            - max_drawdown_pct, longest_drawdown_bars
            - total_trades, win_rate_pct, etc.
            - information_ratio (only if benchmark provided)
            - alpha, beta (only if benchmark provided)
        """
        # Build equity DataFrame with defensive deduplication
        equity_df = self._build_equity_df()

        if equity_df.empty or len(equity_df) < 2:
            logger.warning("Insufficient equity data for metrics calculation")
            return {}

        # Get closed trades only (Requirement 10.4)
        closed_trades = [t for t in self._tracker.trades if t.is_closed]

        # Calculate metrics with error handling
        try:
            metrics = self._metrics_calc.calculate(
                equity=equity_df,
                trades=closed_trades if closed_trades else None,
                benchmark=self._benchmark_equity,
            )
            result = metrics.to_dict()
        except Exception as e:
            logger.error("MetricsCalculator failed: %s", e, exc_info=True)
            return {}

        # Requirement 10.3: Remove benchmark metrics if no benchmark provided
        if self._benchmark_equity is None:
            result.pop("information_ratio", None)
            result.pop("alpha", None)
            result.pop("beta", None)

        return result

    def print_summary(self) -> None:
        """Print formatted summary of session statistics."""
        trades = self._tracker.trades
        closed = [t for t in trades if t.is_closed]
        initial = self._tracker.initial_capital
        final = self._tracker.equity

        total_pnl = sum(t.pnl for t in closed)
        total_commission = sum(t.commission for t in closed)
        pnl_pct = total_pnl / initial * 100 if initial > 0 else 0.0

        winners = [t for t in closed if t.pnl > 0]
        losers = [t for t in closed if t.pnl <= 0]
        win_rate = len(winners) / len(closed) * 100 if closed else 0.0

        print()
        print("=" * 60)
        print("  PAPER TRADING SESSION SUMMARY")
        print("=" * 60)

        sign = "+" if total_pnl >= 0 else ""
        print(f"\n  Initial Capital : {initial:>18,.0f} VND")
        print(f"  Final Equity    : {final:>18,.0f} VND")
        print(f"  Net P&L         : {sign}{total_pnl:>17,.0f} VND  ({sign}{pnl_pct:.2f}%)")
        print(f"  Total Commission: {total_commission:>18,.0f} VND")

        print(f"\n  Total Trades    : {len(closed):>10}")
        print(f"  Winning Trades  : {len(winners):>10}  ({win_rate:.1f}%)")
        print(f"  Losing Trades   : {len(losers):>10}")

        if winners:
            avg_win = sum(t.pnl for t in winners) / len(winners)
            best = max(winners, key=lambda t: t.pnl)
            print(f"  Avg Win         : {avg_win:>18,.0f} VND")
            print(f"  Best Trade      : {best.pnl:>18,.0f} VND")

        if losers:
            avg_loss = sum(t.pnl for t in losers) / len(losers)
            worst = min(losers, key=lambda t: t.pnl)
            print(f"  Avg Loss        : {avg_loss:>18,.0f} VND")
            print(f"  Worst Trade     : {worst.pnl:>18,.0f} VND")

        # Performance metrics
        metrics = self.compute()
        if metrics:
            print()
            print("  --- Performance Metrics ---")
            labels = {
                "total_return_pct": "Total Return (%)",
                "sharpe_ratio": "Sharpe Ratio",
                "sortino_ratio": "Sortino Ratio",
                "max_drawdown_pct": "Max Drawdown (%)",
                "net_profit_factor": "Net Profit Factor",
            }
            for key, label in labels.items():
                val = metrics.get(key)
                if val is not None:
                    print(f"  {label:<24}: {val:>10.4f}")

            # Benchmark metrics (if available)
            if "information_ratio" in metrics and metrics["information_ratio"] is not None:
                print(f"  {'Information Ratio':<24}: {metrics['information_ratio']:>10.4f}")
            if "alpha" in metrics and metrics["alpha"] is not None:
                print(f"  {'Alpha (%)':<24}: {metrics['alpha']:>10.4f}")
            if "beta" in metrics and metrics["beta"] is not None:
                print(f"  {'Beta':<24}: {metrics['beta']:>10.4f}")

        print()
        print("=" * 60)

    def save(self, output_dir: str | None = None) -> Path:
        """Save session statistics to Parquet files.

        Writes three files:
        - trades.parquet: Closed trades with rich data (gross_pnl, MAE/MFE, duration)
        - equity_curve.parquet: Equity snapshots
        - session_metrics.json: Performance metrics

        Args:
            output_dir: Output directory path. If None, creates timestamped directory.

        Returns:
            Path to the output directory.
        """
        # Create timestamped directory if not specified
        if output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/paper_{ts}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save closed trades only (Requirement 10.4)
        closed_trades = [t for t in self._tracker.trades if t.is_closed]
        if closed_trades:
            trades_data = []
            for trade in closed_trades:
                trades_data.append(
                    {
                        "trade_id": getattr(trade, "trade_id", None),
                        "side": trade.side.value,
                        "entry_time": trade.entry_time,
                        "entry_price": trade.entry_price,
                        "exit_time": trade.exit_time,
                        "exit_price": trade.exit_price,
                        "quantity": trade.quantity,
                        "gross_pnl": getattr(trade, "gross_pnl", trade.pnl + trade.commission),
                        "commission": trade.commission,
                        "pnl": trade.pnl,
                        "exit_reason": trade.exit_reason,
                        "duration_min": getattr(trade, "duration_minutes", None),
                        "mae": getattr(trade, "mae", None),
                        "mfe": getattr(trade, "mfe", None),
                    }
                )

            trades_df = pd.DataFrame(trades_data)
            trades_path = output_path / "trades.parquet"
            trades_df.to_parquet(trades_path, index=False)
            logger.info(f"Saved {len(closed_trades)} closed trades to {trades_path}")

        # Save equity curve
        equity_df = self._build_equity_df()
        if not equity_df.empty:
            equity_path = output_path / "equity_curve.parquet"
            equity_df.to_parquet(equity_path, index=False)
            logger.info(f"Saved equity curve to {equity_path}")

        # Save metrics JSON
        metrics = self.compute()
        if metrics:
            metrics_path = output_path / "session_metrics.json"
            metrics_path.write_text(
                json.dumps(metrics, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info(f"Saved metrics to {metrics_path}")

        return output_path
