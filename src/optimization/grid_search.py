"""
Grid Search optimization for strategy parameters.
"""

import concurrent.futures
import itertools
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import pandas as pd
from tqdm import tqdm

from config.config import CONTRACT_MULTIPLIER, DEFAULT_INITIAL_CAPITAL, RESULTS_DIR
from src.engine.backtester import Backtester

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Serializable result for one parameter combination."""

    params: Dict[str, Any]
    metrics: Dict[str, float]
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    total_trades: int

    def to_dict(self) -> Dict[str, Any]:
        """Flatten params + metrics for tabular export."""
        result = self.params.copy()
        result.update(self.metrics)
        return result


class GridSearch:
    """
    Grid search optimizer for strategy parameters.

    Tests all combinations of parameters in the search grid
    and ranks by objective function (default: Sharpe ratio).
    """

    def __init__(
        self,
        strategy_class: type | None,
        param_grid: Dict[str, List[Any]],
        objective: str = "sharpe_ratio",
        minimize: bool = False,
        indicator_fn: Optional[
            Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]
        ] = None,
    ):
        """
        Initialize grid search.

        Args:
            strategy_class: Strategy class to instantiate
            param_grid: Dictionary of parameter names to lists of values
            objective: Metric to optimize
            minimize: If True, minimize objective; if False, maximize
            indicator_fn: Function to recalculate indicators for each param set.
                          Signature: (data_copy, params) -> DataFrame with indicators.
        """
        if strategy_class is None:
            raise ValueError("A valid strategy class must be provided.")

        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.objective = objective
        self.minimize = minimize
        self.indicator_fn = indicator_fn

        self.results: List[OptimizationResult] = []

    def _generate_combinations(self) -> Iterator[Dict[str, Any]]:
        """Yield Cartesian-product parameter dictionaries."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    @property
    def total_combinations(self) -> int:
        """Return total search-space size."""
        total = 1
        for values in self.param_grid.values():
            total *= len(values)
        return total

    @staticmethod
    def _run_backtest(
        strategy_class: type,
        params: Dict[str, Any],
        data: pd.DataFrame,
        initial_capital: float,
        contract_multiplier: float,
        indicator_fn: Optional[
            Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]
        ] = None,
    ) -> Optional[OptimizationResult]:
        """
        Run a single backtest (helper for parallel execution).
        """
        try:
            # Create strategy with parameters
            strategy = strategy_class(**params)

            # Recalculate indicators if an indicator function is provided
            test_data = data
            if indicator_fn:
                test_data = indicator_fn(data.copy(), params)

            # Run backtest
            backtester = Backtester(
                strategy=strategy,
                initial_capital=initial_capital,
                contract_multiplier=contract_multiplier,
            )
            result = backtester.run(test_data)

            # Create optimization result
            return OptimizationResult(
                params=params,
                metrics=result.metrics,
                sharpe_ratio=result.metrics.get("sharpe_ratio", 0),
                total_return=result.metrics.get("total_return_pct", 0),
                max_drawdown=result.metrics.get("max_drawdown_pct", 0),
                total_trades=int(result.metrics.get("total_trades", 0)),
            )
        except Exception:
            logger.exception("Backtest failed for params=%s", params)
            return None

    def optimize(
        self,
        data: pd.DataFrame,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        contract_multiplier: float = CONTRACT_MULTIPLIER,
        show_progress: bool = True,
        n_jobs: int = -1,
    ) -> List[OptimizationResult]:
        """
        Run grid search optimization (optionally parallel).

        Args:
            data: Preprocessed backtest data (without indicators or with base indicators)
            initial_capital: Starting capital for each backtest
            show_progress: Show progress bar
            n_jobs: Number of parallel jobs (-1 for all cores)

        Returns:
            List of OptimizationResult sorted by objective
        """
        self.results = []

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        combinations = list(self._generate_combinations())

        if show_progress:
            print(
                f"Running grid search on {len(combinations)} combinations using {n_jobs} cores..."
            )

        # Remove indicators from data to avoid using stale values
        # Keep only OHLCV columns
        base_columns = ["datetime", "open", "high", "low", "close", "volume"]

        # Only keep base columns that exist in the dataframe
        available_base_cols = [col for col in base_columns if col in data.columns]
        clean_data = data[available_base_cols].copy()

        if n_jobs == 1:
            # --- Serial execution (avoids Windows ProcessPoolExecutor spawn issues) ---
            iterator = combinations
            if show_progress:
                iterator = tqdm(
                    combinations, total=len(combinations), desc="Optimizing"
                )

            for params in iterator:
                result = self._run_backtest(
                    self.strategy_class,
                    params,
                    clean_data,
                    initial_capital,
                    contract_multiplier,
                    self.indicator_fn,
                )
                if result:
                    self.results.append(result)
        else:
            # --- Parallel execution ---
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all jobs
                future_to_params = {
                    executor.submit(
                        self._run_backtest,
                        self.strategy_class,
                        params,
                        clean_data,
                        initial_capital,
                        contract_multiplier,
                        self.indicator_fn,
                    ): params
                    for params in combinations
                }

                # Process results as they complete
                iterator = concurrent.futures.as_completed(future_to_params)
                if show_progress:
                    iterator = tqdm(
                        iterator, total=len(combinations), desc="Optimizing"
                    )

                for future in iterator:
                    try:
                        result = future.result()
                        if result:
                            self.results.append(result)
                    except Exception:
                        logger.exception("Optimization worker job failed")

        # Sort by objective
        self.results.sort(
            key=lambda x: x.metrics.get(self.objective, 0),
            reverse=not self.minimize,
        )

        return self.results

    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        """Return params of the top-ranked result (if available)."""
        if not self.results:
            return None
        return self.results[0].params

    @property
    def best_result(self) -> Optional[OptimizationResult]:
        """Return the top-ranked optimization result (if available)."""
        if not self.results:
            return None
        return self.results[0]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all optimization results to a DataFrame."""
        if not self.results:
            return pd.DataFrame()

        return pd.DataFrame([r.to_dict() for r in self.results])

    def save_results(
        self,
        filename: Optional[str] = None,
        directory: str = RESULTS_DIR,
    ) -> Path:
        """
        Save optimization results to CSV.

        Args:
            filename: Output filename (auto-generated if None)
            directory: Output directory

        Returns:
            Path to saved file
        """
        if not self.results:
            raise ValueError("No results to save")

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_grid_{timestamp}.csv"

        output_path = output_dir / filename

        df = self.to_dataframe()
        df.to_csv(output_path, index=False)

        return output_path

    def print_top_results(self, n: int = 10) -> None:
        """Print a console summary for the top N parameter sets."""
        if not self.results:
            print("No results available")
            return

        print(f"\n{'=' * 60}")
        print(f"TOP {n} PARAMETER COMBINATIONS")
        print(f"Objective: {self.objective} ({'min' if self.minimize else 'max'})")
        print(f"{'=' * 60}\n")

        for i, result in enumerate(self.results[:n], 1):
            print(f"Rank {i}:")
            print(f"  Parameters: {result.params}")
            print(f"  Sharpe:     {result.sharpe_ratio:.3f}")
            print(f"  Return:     {result.total_return:.2f}%")
            print(f"  Max DD:     {result.max_drawdown:.2f}%")
            print(f"  Trades:     {result.total_trades}")
            print()
