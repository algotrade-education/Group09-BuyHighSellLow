"""
Monte Carlo simulation on a completed backtest's trade list.

What it does:
    Takes the list of trades from a backtest, shuffles their order N times,
    and rebuilds the equity curve each time. This gives a distribution of
    possible outcomes - answering "what if these trades happened in a
    different order?"

Why it's useful:
    A single backtest gives you ONE sequence of trades. But in reality,
    the order trades occur is somewhat random (depends on which day you
    started, market conditions, etc.). Monte Carlo shows you the range
    of outcomes that are plausible given the same set of trades.

    Key outputs:
        - 5th percentile max drawdown  → realistic worst-case risk
        - 95th percentile return       → realistic best-case upside
        - Probability of ruin          → % of simulations that hit a loss threshold

Usage:
    from src.optimization.monte_carlo import MonteCarlo

    # After running a backtest:
    mc = MonteCarlo(trades=result.trades, initial_capital=500_000_000)
    mc_result = mc.run(n_simulations=1000)
    mc_result.print_summary()
    mc_result.save("results/orb_20240101/monte_carlo")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics.trade_metrics import Trade

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""

    n_simulations: int
    initial_capital: float

    # Distribution of final equity across all simulations
    final_equities: np.ndarray  # shape: (n_simulations,)

    # Distribution of max drawdown across all simulations
    max_drawdowns: np.ndarray  # shape: (n_simulations,) - positive %

    # Distribution of total return across all simulations
    total_returns: np.ndarray  # shape: (n_simulations,) - %

    # Percentile equity curves (5th, 25th, 50th, 75th, 95th)
    percentile_curves: dict[int, np.ndarray] = field(default_factory=dict)

    # Ruin threshold used
    ruin_threshold_pct: float = 20.0

    @property
    def ruin_probability(self) -> float:
        """% of simulations where max drawdown exceeded ruin_threshold_pct."""
        return float(np.mean(self.max_drawdowns > self.ruin_threshold_pct) * 100)

    @property
    def median_return(self) -> float:
        return float(np.median(self.total_returns))

    @property
    def return_p5(self) -> float:
        """5th percentile return - pessimistic scenario."""
        return float(np.percentile(self.total_returns, 5))

    @property
    def return_p95(self) -> float:
        """95th percentile return - optimistic scenario."""
        return float(np.percentile(self.total_returns, 95))

    @property
    def drawdown_p50(self) -> float:
        return float(np.percentile(self.max_drawdowns, 50))

    @property
    def drawdown_p95(self) -> float:
        """95th percentile max drawdown - near worst-case."""
        return float(np.percentile(self.max_drawdowns, 95))

    def print_summary(self) -> None:
        print(f"\n{'─' * 60}")
        print("  MONTE CARLO SUMMARY")
        print(f"{'─' * 60}")
        print(f"  Simulations:        {self.n_simulations:,}")
        print(f"  Initial Capital:    {self.initial_capital:,.0f}")
        print()
        print("  Return Distribution:")
        print(f"    5th  percentile:  {self.return_p5:>8.2f}%  ← pessimistic")
        print(f"    50th percentile:  {self.median_return:>8.2f}%  ← median")
        print(f"    95th percentile:  {self.return_p95:>8.2f}%  ← optimistic")
        print()
        print("  Max Drawdown Distribution:")
        print(f"    50th percentile:  {self.drawdown_p50:>8.2f}%")
        print(f"    95th percentile:  {self.drawdown_p95:>8.2f}%  ← near worst-case")
        print()
        print(f"  Ruin Probability:   {self.ruin_probability:>8.2f}%")
        print(f"  (drawdown > {self.ruin_threshold_pct:.0f}%)")
        print(f"{'─' * 60}\n")

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        """Save simulation results to CSV files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        # Summary stats
        summary = {
            "n_simulations": self.n_simulations,
            "initial_capital": self.initial_capital,
            "return_p5": self.return_p5,
            "return_p50": self.median_return,
            "return_p95": self.return_p95,
            "drawdown_p50": self.drawdown_p50,
            "drawdown_p95": self.drawdown_p95,
            "ruin_probability_pct": self.ruin_probability,
            "ruin_threshold_pct": self.ruin_threshold_pct,
        }
        import json

        summary_path = out / "mc_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        paths["summary"] = summary_path

        # Distribution data
        dist_df = pd.DataFrame(
            {
                "final_equity": self.final_equities,
                "total_return_pct": self.total_returns,
                "max_drawdown_pct": self.max_drawdowns,
            }
        )
        dist_path = out / "mc_distribution.csv"
        dist_df.to_csv(dist_path, index=False)
        paths["distribution"] = dist_path

        # Percentile curves
        if self.percentile_curves:
            curves_df = pd.DataFrame(self.percentile_curves)
            curves_df.columns = [f"p{p}" for p in curves_df.columns]
            curves_path = out / "mc_percentile_curves.csv"
            curves_df.to_csv(curves_path, index=False)
            paths["curves"] = curves_path

        logger.info("Monte Carlo results saved to %s", out)
        return paths


class MonteCarlo:
    """
    Monte Carlo simulation by shuffling trade order.

    Each simulation:
        1. Shuffle the trade list randomly
        2. Rebuild equity curve: equity[i] = equity[i-1] + trade[i].pnl
        3. Calculate max drawdown and total return

    This is NOT price simulation - it uses actual historical trade PnLs.
    The assumption is that trade PnLs are roughly independent (no serial
    correlation), which holds reasonably well for intraday strategies.
    """

    def __init__(
        self,
        trades: list[Trade],
        initial_capital: float,
        ruin_threshold_pct: float = 20.0,
        seed: int | None = 42,
    ) -> None:
        """
        Args:
            trades:              Closed trades from a BacktestResult.
            initial_capital:     Starting capital (for drawdown % calculation).
            ruin_threshold_pct:  Drawdown % considered "ruin" for probability calc.
                                 Default 20% = lose 20% of capital.
            seed:                Random seed for reproducibility. None = random.
        """
        if not trades:
            raise ValueError("trades list is empty - run a backtest first.")

        self._pnls = np.array([t.pnl for t in trades], dtype=np.float64)
        self._initial_capital = initial_capital
        self._ruin_threshold = ruin_threshold_pct
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        n_simulations: int = 1000,
        save_percentile_curves: bool = True,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Args:
            n_simulations:         Number of shuffles to run. 1000 is usually enough.
                                   Use 5000+ for stable tail percentiles.
            save_percentile_curves: Compute and save equity curves at 5/25/50/75/95th
                                   percentiles. Useful for plotting.

        Returns:
            MonteCarloResult with distribution statistics.
        """
        _ = len(self._pnls)
        final_equities = np.empty(n_simulations)
        max_drawdowns = np.empty(n_simulations)

        # Store all equity curves only if we need percentile curves
        all_curves: list[np.ndarray] | None = [] if save_percentile_curves else None

        for i in range(n_simulations):
            shuffled = self._rng.permutation(self._pnls)
            equity = self._build_equity_curve(shuffled)

            final_equities[i] = equity[-1]
            max_drawdowns[i] = self._max_drawdown_pct(equity)

            if all_curves is not None:
                all_curves.append(equity)

        total_returns = (final_equities - self._initial_capital) / self._initial_capital * 100

        # Compute percentile curves
        percentile_curves: dict[int, np.ndarray] = {}
        if all_curves is not None and all_curves:
            curves_matrix = np.stack(all_curves)  # (n_simulations, n_trades+1)
            for p in (5, 25, 50, 75, 95):
                percentile_curves[p] = np.percentile(curves_matrix, p, axis=0)

        logger.info(
            "Monte Carlo: %d simulations, median return=%.2f%%, p95 drawdown=%.2f%%",
            n_simulations,
            float(np.median(total_returns)),
            float(np.percentile(max_drawdowns, 95)),
        )

        return MonteCarloResult(
            n_simulations=n_simulations,
            initial_capital=self._initial_capital,
            final_equities=final_equities,
            max_drawdowns=max_drawdowns,
            total_returns=total_returns,
            percentile_curves=percentile_curves,
            ruin_threshold_pct=self._ruin_threshold,
        )

    def _build_equity_curve(self, pnls: np.ndarray) -> np.ndarray:
        """Build equity curve from PnL array. Returns array of length n+1."""
        curve = np.empty(len(pnls) + 1)
        curve[0] = self._initial_capital
        np.cumsum(pnls, out=curve[1:])
        curve[1:] += self._initial_capital
        return curve

    def _max_drawdown_pct(self, equity: np.ndarray) -> float:
        """Calculate max drawdown as % of peak equity."""
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100  # negative values
        return float(-np.min(drawdown))  # return as positive %
