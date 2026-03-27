"""
Exhaustive grid search over strategy parameter combinations.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.optimization.scoring import ScorerConfig, calculate_score

logger = logging.getLogger(__name__)


@dataclass
class GridResult:
    """Result for one parameter combination."""

    params: dict[str, Any]
    metrics: dict[str, Any]
    score: float
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    total_trades: int

    def to_dict(self) -> dict[str, Any]:
        result = {"score": self.score}
        result.update(self.params)
        result.update(self.metrics)
        return result


# Type alias for the trial function
# Signature: (params: dict) -> BacktestResult
TrialFn = Callable[[dict[str, Any]], Any]


class GridSearch:
    """
    Exhaustive grid search over all Cartesian product combinations.

    Unlike V1 which used strategy_class(**params) directly, V2 accepts
    a trial_fn callable - the caller decides how to build strategy +
    run backtest. This decouples GridSearch from any specific strategy class.

    Usage:
        def trial_fn(params: dict) -> BacktestResult:
            config   = ORBConfig.from_dict({..., "strategy": params})
            strategy = ORBStrategy(config)
            registry = ORBStrategy.build_registry(**params)
            pipeline = DataPipeline(registry)
            data     = pipeline.run(preprocessed_df)
            bt       = Backtester(strategy, ...)
            return bt.run(data)

        grid = GridSearch(
            trial_fn=trial_fn,
            param_grid={"orb_minutes": [15, 20, 30], "atr_period": [10, 14]},
        )
        results = grid.optimize()
    """

    def __init__(
        self,
        trial_fn: TrialFn,
        param_grid: dict[str, list[Any]],
        scorer: ScorerConfig | None = None,
        objective: str = "score",
        minimize: bool = False,
    ) -> None:
        """
        Args:
            trial_fn:   Callable that accepts a params dict and returns
                        a BacktestResult (or any object with .metrics dict).
            param_grid: Dict mapping param name → list of values to try.
            scorer:     ScorerConfig for composite scoring. Uses defaults if None.
            objective:  Metric key to rank results by (default "score").
            minimize:   If True, rank ascending (lower is better).
        """
        self._trial_fn = trial_fn
        self._param_grid = param_grid
        self._scorer = scorer or ScorerConfig()
        self._objective = objective
        self._minimize = minimize

        self.results: list[GridResult] = []
        self.failed_params: list[dict[str, Any]] = []

    # ── Properties ────────────────────────────────────────────────

    @property
    def total_combinations(self) -> int:
        return math.prod(len(v) for v in self._param_grid.values())

    @property
    def best_params(self) -> dict[str, Any] | None:
        return self.results[0].params if self.results else None

    @property
    def best_result(self) -> GridResult | None:
        return self.results[0] if self.results else None

    # ── Optimize ──────────────────────────────────────────────────

    def optimize(
        self,
        n_jobs: int = 1,
        chunk_size: int = 50,
        show_progress: bool = True,
    ) -> list[GridResult]:
        """
        Run grid search over all parameter combinations.

        Args:
            n_jobs:        Number of parallel workers. 1 = serial (recommended
                           on Windows or when trial_fn has side effects).
            chunk_size:    Max futures submitted at once to avoid OOM.
                           (V1 bug: submitted all combinations simultaneously)
            show_progress: Print progress to stdout.

        Returns:
            Results sorted best → worst by objective.
        """
        self.results = []
        self.failed_params = []

        combinations = self._dedup_combinations()
        total = len(combinations)

        if show_progress:
            print(
                f"Grid search: {total} combinations ({n_jobs} worker{'s' if n_jobs != 1 else ''})"
            )

        if n_jobs == 1:
            self._run_serial(combinations, show_progress)
        else:
            self._run_parallel(combinations, n_jobs, chunk_size, show_progress)

        self._sort_results()

        if self.failed_params and show_progress:
            print(f"⚠️  {len(self.failed_params)} combinations failed - check logs.")

        return self.results

    # ── Serial ────────────────────────────────────────────────────

    def _run_serial(
        self,
        combinations: list[dict[str, Any]],
        show_progress: bool,
    ) -> None:
        total = len(combinations)
        for i, params in enumerate(combinations, 1):
            if show_progress and i % max(1, total // 20) == 0:
                print(f"\r  {i}/{total} ({i / total * 100:.0f}%)", end="", flush=True)
            result = self._run_trial(params)
            if result is not None:
                self.results.append(result)
            else:
                self.failed_params.append(params)
        if show_progress:
            print()

    # ── Parallel ──────────────────────────────────────────────────

    def _run_parallel(
        self,
        combinations: list[dict[str, Any]],
        n_jobs: int,
        chunk_size: int,
        show_progress: bool,
    ) -> None:
        """
        Submit futures in chunks to avoid OOM.
        V1 bug: submitted all combinations at once → peak RAM = n_combos × data_size.

        Uses ThreadPoolExecutor (not ProcessPoolExecutor) because trial_fn is
        typically a closure that captures data/config and is not picklable.
        For CPU-bound workloads, use n_jobs=1 with external parallelism instead.
        """
        total = len(combinations)
        completed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            for chunk_start in range(0, total, chunk_size):
                chunk = combinations[chunk_start : chunk_start + chunk_size]
                futures = {executor.submit(self._trial_fn, params): params for params in chunk}

                for future in concurrent.futures.as_completed(futures):
                    params = futures[future]
                    try:
                        bt_result = future.result()
                        result = self._make_grid_result(params, bt_result)
                        if result is not None:
                            self.results.append(result)
                        else:
                            self.failed_params.append(params)
                    except Exception as e:
                        logger.error("Trial failed for params=%s: %s", params, e)
                        self.failed_params.append(params)

                    completed += 1
                    if show_progress and completed % max(1, total // 20) == 0:
                        print(
                            f"\r  {completed}/{total} ({completed / total * 100:.0f}%)",
                            end="",
                            flush=True,
                        )

        if show_progress:
            print()

    # ── Single trial ──────────────────────────────────────────────

    def _run_trial(self, params: dict[str, Any]) -> GridResult | None:
        """Run one trial in the current process."""
        try:
            bt_result = self._trial_fn(params)
            return self._make_grid_result(params, bt_result)
        except Exception as e:
            logger.error("Trial failed for params=%s: %s", params, e, exc_info=True)
            return None

    def _make_grid_result(self, params: dict[str, Any], bt_result: Any) -> GridResult | None:
        """Convert BacktestResult → GridResult with composite score."""
        try:
            metrics = bt_result.metrics if hasattr(bt_result, "metrics") else bt_result
            score = calculate_score(metrics, self._scorer)
            return GridResult(
                params=params,
                metrics=metrics,
                score=score,
                sharpe_ratio=float(metrics.get("sharpe_ratio", 0)),
                total_return=float(metrics.get("total_return_pct", 0)),
                max_drawdown=float(metrics.get("max_drawdown_pct", 0)),
                total_trades=int(metrics.get("total_trades", 0)),
            )
        except Exception as e:
            logger.error("Failed to build GridResult: %s", e)
            return None

    # ── Helpers ───────────────────────────────────────────────────

    def _dedup_combinations(self) -> list[dict[str, Any]]:
        """
        Generate Cartesian product, deduplicate, and return list.
        Dedup catches accidental duplicates like [14, 14, 20] in param lists.
        """
        keys = list(self._param_grid.keys())
        values = list(self._param_grid.values())
        seen = set()
        result = []

        for combo in itertools.product(*values):
            key = tuple(combo)
            if key not in seen:
                seen.add(key)
                result.append(dict(zip(keys, combo, strict=False)))

        n_dupes = self.total_combinations - len(result)
        if n_dupes > 0:
            logger.warning("Removed %d duplicate param combinations.", n_dupes)

        return result

    def _sort_results(self) -> None:
        key = self._objective if self._objective != "score" else "score"
        self.results.sort(
            key=lambda r: r.metrics.get(key, r.score) if key != "score" else r.score,
            reverse=not self._minimize,
        )

    # ── Export ────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.results])

    def save(self, output_dir: str | Path, filename: str | None = None) -> Path:
        if not self.results:
            raise ValueError("No results to save.")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = filename or f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = out / name
        self.to_dataframe().to_csv(path, index=False)
        logger.info("Grid search results saved to %s", path)
        return path

    def print_top(self, n: int = 10) -> None:
        if not self.results:
            print("No results.")
            return
        print(f"\n{'─' * 60}")
        print(f"  TOP {n} GRID SEARCH RESULTS")
        print(f"{'─' * 60}")
        for i, r in enumerate(self.results[:n], 1):
            print(f"\n  Rank {i}:")
            print(f"    Score:   {r.score:.4f}")
            print(f"    Params:  {r.params}")
            print(
                f"    Sharpe:  {r.sharpe_ratio:.3f}  |  "
                f"Return: {r.total_return:.2f}%  |  "
                f"MaxDD: {r.max_drawdown:.2f}%  |  "
                f"Trades: {r.total_trades}"
            )
        print()
