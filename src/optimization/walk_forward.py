"""
Walk-Forward Optimization - anchored and rolling modes.

Supports both GridSearch and OptunaSearch as the inner optimizer,
so you can use exhaustive grid search for small param spaces or
Bayesian TPE for larger ones.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.optimization.grid_search import GridResult
from src.optimization.optuna_search import OptunaResult
from src.optimization.scoring import ScorerConfig

logger = logging.getLogger(__name__)

# trial_fn signature: (params, data, capital) -> BacktestResult
# - params:  best params found by inner optimizer
# - data:    the slice of data for this window (train or test)
# - capital: starting capital for this window
TrialFn = Callable[[dict[str, Any], pd.DataFrame, float], Any]


@dataclass
class WalkForwardWindow:
    """Results for a single walk-forward window."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: dict[str, Any]
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    train_capital_end: float = 0.0
    test_capital_end: float = 0.0

    @property
    def train_sharpe(self) -> float:
        return float(self.train_metrics.get("sharpe_ratio", 0))

    @property
    def test_sharpe(self) -> float:
        return float(self.test_metrics.get("sharpe_ratio", 0))

    @property
    def sharpe_degradation(self) -> float:
        """Percentage change from train Sharpe to test Sharpe."""
        if self.train_sharpe == 0:
            return 0.0
        return (self.test_sharpe - self.train_sharpe) / abs(self.train_sharpe) * 100


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward optimization results."""

    windows: list[WalkForwardWindow]
    combined_test_trades: list[Any]
    combined_test_equity: pd.DataFrame
    aggregate_metrics: dict[str, Any]
    mode: str = "anchored"
    optimizer: str = "grid"

    @property
    def avg_train_sharpe(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.train_sharpe for w in self.windows) / len(self.windows)

    @property
    def avg_test_sharpe(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.test_sharpe for w in self.windows) / len(self.windows)

    @property
    def avg_degradation(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.sharpe_degradation for w in self.windows) / len(self.windows)

    @property
    def robustness_ratio(self) -> float:
        """Ratio of avg test Sharpe to avg train Sharpe. > 0.5 is generally acceptable."""
        if self.avg_train_sharpe == 0:
            return 0.0
        return self.avg_test_sharpe / self.avg_train_sharpe

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "window": w.window_id,
                    "train_start": w.train_start,
                    "train_end": w.train_end,
                    "test_start": w.test_start,
                    "test_end": w.test_end,
                    "train_sharpe": w.train_sharpe,
                    "test_sharpe": w.test_sharpe,
                    "degradation_pct": w.sharpe_degradation,
                    "test_capital_end": w.test_capital_end,
                    **{f"param_{k}": v for k, v in w.best_params.items()},
                }
                for w in self.windows
            ]
        )

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        """Save all result components. Returns dict of name → path."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        df_path = out / "walk_forward_windows.csv"
        self.to_dataframe().to_csv(df_path, index=False)
        paths["windows"] = df_path

        if not self.combined_test_equity.empty:
            eq_path = out / "walk_forward_equity.csv"
            self.combined_test_equity.to_csv(eq_path, index=False)
            paths["equity"] = eq_path

        agg = {
            **self.aggregate_metrics,
            "robustness_ratio": self.robustness_ratio,
            "avg_train_sharpe": self.avg_train_sharpe,
            "avg_test_sharpe": self.avg_test_sharpe,
            "avg_degradation_pct": self.avg_degradation,
        }
        agg_path = out / "walk_forward_aggregate.json"
        agg_path.write_text(json.dumps(agg, indent=2, default=str))
        paths["aggregate"] = agg_path

        logger.info("Walk-forward results saved to %s", out)
        return paths

    def print_summary(self) -> None:
        print(f"\n{'-' * 60}")
        print("  WALK-FORWARD SUMMARY")
        print(f"{'-' * 60}")
        print(f"  Mode:              {self.mode}")
        print(f"  Optimizer:         {self.optimizer}")
        print(f"  Windows:           {len(self.windows)}")
        print(f"\n  Avg Train Sharpe:  {self.avg_train_sharpe:.3f}")
        print(f"  Avg Test Sharpe:   {self.avg_test_sharpe:.3f}")
        print(f"  Avg Degradation:   {self.avg_degradation:.1f}%")
        print(f"  Robustness Ratio:  {self.robustness_ratio:.2f}")
        print("\n  Per-Window:")
        for w in self.windows:
            print(
                f"    Window {w.window_id}: "
                f"train {w.train_start.date()} → {w.train_end.date()} "
                f"| test {w.test_start.date()} → {w.test_end.date()}"
            )
            print(
                f"      Train Sharpe: {w.train_sharpe:.3f}  "
                f"Test Sharpe: {w.test_sharpe:.3f}  "
                f"Degradation: {w.sharpe_degradation:.1f}%  "
                f"Params: {w.best_params}"
            )
        print(f"{'-' * 60}\n")


class WalkForwardOptimizer:
    """
    Walk-forward optimizer supporting both GridSearch and OptunaSearch.

    Process:
        1. Split data into N train/test windows (anchored or rolling).
        2. Run inner optimizer (Grid or Optuna) on train window.
        3. Test best params on out-of-sample test window.
        4. Capital at end of test window chains to next window.
        5. Aggregate results into robustness ratio.

    The trial_fn signature is:
        def trial_fn(params: dict, data: pd.DataFrame, capital: float) -> BacktestResult

    This is different from standalone GridSearch/OptunaSearch where trial_fn
    only takes params. The extra data and capital args let walk-forward pass
    the correct window slice and chained capital to each backtest.

    Embargo period:
        The first `embargo_bars` bars of each test window are excluded.
        Set to max indicator lookback (e.g. 14 for ATR-14) to prevent
        indicator state leakage from train into test.

    Usage:
        def trial_fn(params, data, capital):
            config = ORBConfig.from_dict({..., "strategy": params})
            strategy = ORBStrategy(config)
            registry = ORBStrategy.build_registry(**params)
            pipeline = DataPipeline(registry)
            df = pipeline.run(data)
            bt = Backtester(strategy, initial_capital=capital, ...)
            return bt.run(df)

        wfo = WalkForwardOptimizer(
            trial_fn=trial_fn,
            param_space={"orb_minutes": [15, 20, 30], "atr_period": [10, 14]},
            optimizer="optuna",
            n_trials=100,          # only used when optimizer="optuna"
            n_windows=5,
            anchored=True,
        )
        result = wfo.optimize(full_data, initial_capital=500_000_000)
        result.print_summary()
    """

    def __init__(
        self,
        trial_fn: TrialFn,
        param_space: dict[str, Any],
        optimizer: Literal["grid", "optuna"] = "optuna",
        n_windows: int = 5,
        train_pct: float = 0.7,
        anchored: bool = True,
        scorer: ScorerConfig | None = None,
        embargo_bars: int = 0,
        chain_capital: bool = True,
        # Grid-specific
        grid_n_jobs: int = 1,
        # Optuna-specific
        n_trials: int = 100,
        optuna_seed: int = 42,
        optuna_storage: str | None = None,
    ) -> None:
        """
        Args:
            trial_fn:         (params, data, capital) → BacktestResult.
            param_space:      For grid: {"param": [v1, v2, ...]}.
                              For optuna: {"param": {"type": "int", "low": 10, "high": 50}}.
            optimizer:        "grid" for exhaustive search, "optuna" for Bayesian TPE.
                              Use "grid" for < ~200 combinations, "optuna" for larger spaces.
            n_windows:        Number of walk-forward windows.
            train_pct:        Fraction of each window for training (rolling mode only).
                              Must be in (0, 1). E.g. 0.7 = 70% train, 30% test.
            anchored:         True = anchored (train grows), False = rolling (fixed size).
            scorer:           ScorerConfig for composite scoring.
            embargo_bars:     Bars to skip at start of test window (indicator leakage guard).
            chain_capital:    Chain ending equity from test window to next window's capital.
            grid_n_jobs:      Parallel workers for grid search (ThreadPoolExecutor).
            n_trials:         Optuna trials per window.
            optuna_seed:      Optuna random seed.
            optuna_storage:   SQLite path for Optuna persistence per window.
        """
        if n_windows < 2:
            raise ValueError(f"n_windows must be >= 2, got {n_windows}")
        if not 0 < train_pct < 1:
            raise ValueError(f"train_pct must be in (0, 1), got {train_pct}")
        if embargo_bars < 0:
            raise ValueError(f"embargo_bars must be >= 0, got {embargo_bars}")

        self._trial_fn = trial_fn
        self._param_space = param_space
        self._optimizer = optimizer
        self._n_windows = n_windows
        self._train_pct = train_pct
        self._anchored = anchored
        self._scorer = scorer or ScorerConfig()
        self._embargo_bars = embargo_bars
        self._chain_capital = chain_capital
        self._grid_n_jobs = grid_n_jobs
        self._n_trials = n_trials
        self._optuna_seed = optuna_seed
        self._optuna_storage = optuna_storage

        self.result: WalkForwardResult | None = None

    # --- Optimize -------------------------------------------------

    def optimize(
        self,
        data: pd.DataFrame,
        initial_capital: float = 500_000_000.0,
        show_progress: bool = True,
    ) -> WalkForwardResult:
        """
        Execute the full walk-forward cycle.

        Args:
            data:            Full dataset (OHLCV + indicators pre-computed,
                             or raw OHLCV if trial_fn handles indicators).
            initial_capital: Starting capital for the first window.
            show_progress:   Print per-window progress.

        Returns:
            WalkForwardResult with per-window details and aggregate metrics.
        """
        windows = self._create_windows(data)
        if not windows:
            raise ValueError("No valid walk-forward windows could be created.")

        wf_windows: list[WalkForwardWindow] = []
        all_test_trades: list[Any] = []
        all_test_equity: list[pd.DataFrame] = []
        current_capital = initial_capital

        for i, (train_df, test_df, ts, te, os_s, os_e) in enumerate(windows, 1):
            if show_progress:
                print(
                    f"\n  Window {i}/{len(windows)}: "
                    f"train {ts.date()} → {te.date()} | "
                    f"test {os_s.date()} → {os_e.date()}"
                )

            # Apply embargo
            if self._embargo_bars > 0 and len(test_df) > self._embargo_bars:
                test_df = test_df.iloc[self._embargo_bars :].reset_index(drop=True)

            if test_df.empty:
                logger.warning("Window %d: test data empty after embargo - skipping.", i)
                continue

            # Optimize on train window
            if show_progress:
                print(f"    Optimizing on {len(train_df):,} train bars ({self._optimizer})...")

            best_params, train_metrics = self._run_inner_optimizer(
                train_df, current_capital, window_id=i
            )

            if best_params is None:
                logger.warning("Window %d: optimizer produced no valid results - skipping.", i)
                continue

            if show_progress:
                train_sharpe = train_metrics.get("sharpe_ratio", 0)
                print(f"    Best train Sharpe: {train_sharpe:.3f} | Params: {best_params}")

            # Test on out-of-sample window
            if show_progress:
                print(
                    f"    Testing on {len(test_df):,} test bars "
                    f"(capital: {current_capital:,.0f})..."
                )

            try:
                test_result = self._trial_fn(best_params, test_df, current_capital)
                test_metrics = (
                    test_result.metrics if hasattr(test_result, "metrics") else test_result
                )
            except Exception as e:
                logger.error("Window %d: test backtest failed: %s", i, e)
                continue

            test_equity_curve = getattr(test_result, "equity_curve", pd.DataFrame())
            test_capital_end = (
                float(test_equity_curve["equity"].iloc[-1])
                if not test_equity_curve.empty and "equity" in test_equity_curve.columns
                else current_capital
            )

            wf_win = WalkForwardWindow(
                window_id=i,
                train_start=ts,
                train_end=te,
                test_start=os_s,
                test_end=os_e,
                best_params=best_params,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                train_capital_end=current_capital,
                test_capital_end=test_capital_end,
            )
            wf_windows.append(wf_win)
            all_test_trades.extend(getattr(test_result, "trades", []))
            if not test_equity_curve.empty:
                all_test_equity.append(test_equity_curve)

            if show_progress:
                test_sharpe = test_metrics.get("sharpe_ratio", 0)
                print(
                    f"    Test Sharpe: {test_sharpe:.3f} | "
                    f"Degradation: {wf_win.sharpe_degradation:.1f}%"
                )

            if self._chain_capital:
                current_capital = test_capital_end

        combined_equity = pd.DataFrame()
        if all_test_equity:
            combined_equity = pd.concat(all_test_equity, ignore_index=True)
            if "datetime" in combined_equity.columns:
                combined_equity = combined_equity.sort_values("datetime", ignore_index=True)

        self.result = WalkForwardResult(
            windows=wf_windows,
            combined_test_trades=all_test_trades,
            combined_test_equity=combined_equity,
            aggregate_metrics=self._build_aggregate(wf_windows, all_test_trades),
            mode="anchored" if self._anchored else "rolling",
            optimizer=self._optimizer,
        )
        return self.result

    # --- Inner optimizer ------------------------------------------

    def _run_inner_optimizer(
        self,
        train_df: pd.DataFrame,
        capital: float,
        window_id: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """
        Run grid or optuna search on the train window.

        Returns (best_params, best_metrics). Both None/{} if no valid results.
        """

        # Wrap trial_fn to inject data and capital
        def wrapped_trial(params: dict[str, Any]) -> Any:
            return self._trial_fn(params, train_df, capital)

        results: list[GridResult] | list[OptunaResult] = []
        if self._optimizer == "grid":
            from src.optimization.grid_search import GridSearch

            grid = GridSearch(wrapped_trial, self._param_space, scorer=self._scorer)
            results = grid.optimize(n_jobs=self._grid_n_jobs, show_progress=False)
            if not results:
                return None, {}
            return results[0].params, results[0].metrics

        elif self._optimizer == "optuna":
            from src.optimization.optuna_search import OptunaSearch

            storage = None
            if self._optuna_storage:
                # Per-window storage to avoid trial conflicts
                p = Path(self._optuna_storage)
                storage = str(p.parent / f"{p.stem}_w{window_id}{p.suffix}")

            search = OptunaSearch(
                trial_fn=wrapped_trial,
                param_space=self._param_space,
                scorer=self._scorer,
                n_trials=self._n_trials,
                seed=self._optuna_seed,
                study_name=f"wfo_window_{window_id}",
                storage_path=storage,
            )

            results = search.optimize(show_progress=False)
            if not results:
                return None, {}
            return results[0].params, results[0].metrics

        else:
            raise ValueError(f"Unknown optimizer: {self._optimizer!r}. Use 'grid' or 'optuna'.")

    # --- Window creation ------------------------------------------

    def _create_windows(
        self,
        data: pd.DataFrame,
        datetime_col: str = "datetime",
    ) -> list[tuple]:
        data = data.sort_values(datetime_col).reset_index(drop=True)
        n_rows = len(data)

        if self._anchored:
            return self._anchored_windows(data, n_rows, datetime_col)
        return self._rolling_windows(data, n_rows, datetime_col)

    def _anchored_windows(self, data: pd.DataFrame, n_rows: int, datetime_col: str) -> list[tuple]:
        """
        Train always starts from row 0, grows with each window.

        Splits data into (n_windows + 1) equal segments.
        Window i: train = rows[0 : (i+1)*seg], test = rows[(i+1)*seg : (i+2)*seg].
        Last window's test extends to end of data to avoid wasting bars.
        """
        seg = n_rows // (self._n_windows + 1)
        if seg == 0:
            raise ValueError(
                f"Not enough data ({n_rows} bars) for {self._n_windows} windows. "
                f"Reduce n_windows or provide more data."
            )

        windows = []
        for i in range(self._n_windows):
            train_end_idx = (i + 1) * seg
            test_start_idx = train_end_idx
            # Last window: extend test to end of data to avoid wasting bars
            test_end_idx = n_rows if i == self._n_windows - 1 else (i + 2) * seg

            train_df = data.iloc[:train_end_idx]
            test_df = data.iloc[test_start_idx:test_end_idx]

            if train_df.empty or test_df.empty:
                logger.warning("Anchored window %d: empty slice - skipping.", i + 1)
                continue

            windows.append(
                (
                    train_df,
                    test_df,
                    train_df[datetime_col].iloc[0],
                    train_df[datetime_col].iloc[-1],
                    test_df[datetime_col].iloc[0],
                    test_df[datetime_col].iloc[-1],
                )
            )

        return windows

    def _rolling_windows(self, data: pd.DataFrame, n_rows: int, datetime_col: str) -> list[tuple]:
        """Fixed-size train+test block slides forward."""
        window_size = n_rows // self._n_windows
        train_size = int(window_size * self._train_pct)
        windows = []

        for i in range(self._n_windows):
            start_idx = i * window_size
            train_end_idx = start_idx + train_size
            test_end_idx = min(start_idx + window_size, n_rows)

            train_df = data.iloc[start_idx:train_end_idx]
            test_df = data.iloc[train_end_idx:test_end_idx]

            if train_df.empty or test_df.empty:
                logger.warning("Rolling window %d: empty slice - skipping.", i + 1)
                continue

            windows.append(
                (
                    train_df,
                    test_df,
                    train_df[datetime_col].iloc[0],
                    train_df[datetime_col].iloc[-1],
                    test_df[datetime_col].iloc[0],
                    test_df[datetime_col].iloc[-1],
                )
            )

        return windows

    # --- Aggregate ------------------------------------------------

    def _build_aggregate(
        self,
        windows: list[WalkForwardWindow],
        all_trades: list[Any],
    ) -> dict[str, Any]:
        if not windows:
            return {}

        agg: dict[str, Any] = {
            "windows_completed": len(windows),
            "total_test_trades": len(all_trades),
        }

        if all_trades:
            agg["total_test_pnl"] = sum(t.pnl for t in all_trades)
            n_wins = sum(1 for t in all_trades if t.pnl > 0)
            agg["test_win_rate"] = n_wins / len(all_trades) * 100

        return agg
