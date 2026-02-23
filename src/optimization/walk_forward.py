"""
Walk-Forward Optimization.

Implements anchored and rolling walk-forward analysis to avoid
overfitting and ensure parameter robustness.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from config.config import CONTRACT_MULTIPLIER, DEFAULT_INITIAL_CAPITAL, RESULTS_DIR
from src.engine.backtester import Backtester

from .grid_search import GridSearch


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: Dict[str, Any]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]

    @property
    def train_sharpe(self) -> float:
        return self.train_metrics.get("sharpe_ratio", 0)

    @property
    def test_sharpe(self) -> float:
        return self.test_metrics.get("sharpe_ratio", 0)

    @property
    def sharpe_degradation(self) -> float:
        """How much Sharpe degraded from train to test."""
        if self.train_sharpe == 0:
            return 0
        return (self.test_sharpe - self.train_sharpe) / abs(self.train_sharpe) * 100


@dataclass
class WalkForwardResult:
    """Container for walk-forward optimization results."""

    windows: List[WalkForwardWindow]
    combined_test_trades: List
    combined_test_equity: pd.DataFrame
    aggregate_metrics: Dict[str, float]

    @property
    def avg_train_sharpe(self) -> float:
        return sum(w.train_sharpe for w in self.windows) / len(self.windows)

    @property
    def avg_test_sharpe(self) -> float:
        return sum(w.test_sharpe for w in self.windows) / len(self.windows)

    @property
    def avg_degradation(self) -> float:
        return sum(w.sharpe_degradation for w in self.windows) / len(self.windows)

    @property
    def robustness_ratio(self) -> float:
        """Ratio of test performance to train performance."""
        if self.avg_train_sharpe == 0:
            return 0
        return self.avg_test_sharpe / self.avg_train_sharpe

    def to_dataframe(self) -> pd.DataFrame:
        """Convert windows to DataFrame."""
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
                    "degradation": w.sharpe_degradation,
                    **{f"param_{k}": v for k, v in w.best_params.items()},
                }
                for w in self.windows
            ]
        )


class WalkForwardOptimizer:
    """
    Walk-Forward Optimizer with anchored or rolling windows.

    Process:
    1. Split data into train/test windows
    2. Optimize parameters on train window
    3. Test best parameters on test window
    4. Repeat for all windows
    5. Aggregate results
    """

    def __init__(
        self,
        strategy_class: type | None,
        param_grid: Dict[str, List[Any]],
        n_windows: int = 5,
        train_pct: float = 0.7,
        anchored: bool = True,
        objective: str = "sharpe_ratio",
        indicator_fn: Optional[
            Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]
        ] = None,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            strategy_class: Strategy class to instantiate
            param_grid: Parameter search grid
            n_windows: Number of walk-forward windows
            train_pct: Percentage of window for training
            anchored: If True, train window always starts from beginning
            objective: Optimization objective metric
        """
        if strategy_class is None:
            raise ValueError("A valid strategy class must be provided.")

        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.n_windows = n_windows
        self.train_pct = train_pct
        self.anchored = anchored
        self.objective = objective
        self.indicator_fn = indicator_fn

        self.result: Optional[WalkForwardResult] = None

    def _create_windows(
        self,
        data: pd.DataFrame,
        datetime_column: str = "datetime",
    ) -> List[
        Tuple[pd.DataFrame, pd.DataFrame, datetime, datetime, datetime, datetime]
    ]:
        """
        Create train/test window splits.

        Returns:
            List of (train_data, test_data, train_start, train_end, test_start, test_end)
        """
        data = data.sort_values(datetime_column).reset_index(drop=True)
        n_rows = len(data)

        # Calculate window sizes
        if self.anchored:
            # Anchored: train grows, test slides
            test_size = n_rows // (self.n_windows + 1)
            windows = []

            for i in range(self.n_windows):
                train_end_idx = int(n_rows * (i + 1) / (self.n_windows + 1))
                test_start_idx = train_end_idx
                test_end_idx = min(test_start_idx + test_size, n_rows)

                train_data = data.iloc[:train_end_idx]
                test_data = data.iloc[test_start_idx:test_end_idx]

                windows.append(
                    (
                        train_data,
                        test_data,
                        train_data[datetime_column].iloc[0],
                        train_data[datetime_column].iloc[-1],
                        test_data[datetime_column].iloc[0],
                        test_data[datetime_column].iloc[-1],
                    )
                )
        else:
            # Rolling: fixed window size that slides
            window_size = n_rows // self.n_windows
            train_size = int(window_size * self.train_pct)
            test_size = window_size - train_size
            windows = []

            for i in range(self.n_windows):
                start_idx = i * window_size
                train_end_idx = start_idx + train_size
                test_end_idx = min(start_idx + window_size, n_rows)

                train_data = data.iloc[start_idx:train_end_idx]
                test_data = data.iloc[train_end_idx:test_end_idx]

                if len(test_data) == 0:
                    continue

                windows.append(
                    (
                        train_data,
                        test_data,
                        train_data[datetime_column].iloc[0],
                        train_data[datetime_column].iloc[-1],
                        test_data[datetime_column].iloc[0],
                        test_data[datetime_column].iloc[-1],
                    )
                )

        return windows

    def optimize(
        self,
        data: pd.DataFrame,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        contract_multiplier: float = CONTRACT_MULTIPLIER,
        show_progress: bool = True,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            data: Preprocessed backtest data (OHLC only, indicators will be calculated per window)
            initial_capital: Starting capital
            show_progress: Show progress bar

        Returns:
            WalkForwardResult with all window results
        """
        # Remove indicators from data to avoid using stale values
        base_columns = ["datetime", "open", "high", "low", "close"]

        # Only keep base columns that exist in the dataframe
        available_base_cols = [col for col in base_columns if col in data.columns]
        clean_data = data[available_base_cols].copy()

        windows = self._create_windows(clean_data)
        wf_windows: List[WalkForwardWindow] = []
        all_test_trades = []
        all_test_equity = []

        iterator = (
            tqdm(enumerate(windows), total=len(windows), desc="Walk-Forward")
            if show_progress
            else enumerate(windows)
        )

        for i, (
            train_data,
            test_data,
            train_start,
            train_end,
            test_start,
            test_end,
        ) in iterator:
            # Optimize on train data
            grid_search = GridSearch(
                strategy_class=self.strategy_class,
                param_grid=self.param_grid,
                objective=self.objective,
                indicator_fn=self.indicator_fn,
            )

            grid_search.optimize(train_data, initial_capital, show_progress=False)

            if not grid_search.best_params or not grid_search.best_result:
                continue

            best_params = grid_search.best_params
            train_metrics = grid_search.best_result.metrics

            # Recalculate indicators for test data using train+test context
            # to avoid cold-start NaNs at the beginning of each test window.
            if self.indicator_fn:
                original_test_len = len(test_data)
                contextual_data = pd.concat([train_data, test_data], ignore_index=True)
                contextual_data = self.indicator_fn(contextual_data.copy(), best_params)

                # Keep only the out-of-sample segment after indicator calculation.
                if len(contextual_data) >= original_test_len:
                    test_data = contextual_data.tail(original_test_len).reset_index(
                        drop=True
                    )
                else:
                    test_data = contextual_data.reset_index(drop=True)

            # Test on out-of-sample data
            test_strategy = self.strategy_class(**best_params)
            test_backtester = Backtester(
                strategy=test_strategy,
                initial_capital=initial_capital,
                contract_multiplier=contract_multiplier,
            )
            test_result = test_backtester.run(test_data)

            # Record window results
            wf_window = WalkForwardWindow(
                window_id=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                train_metrics=train_metrics,
                test_metrics=test_result.metrics,
            )
            wf_windows.append(wf_window)

            # Collect test trades and equity
            all_test_trades.extend(test_result.trades)
            if not test_result.equity_curve.empty:
                all_test_equity.append(test_result.equity_curve)

        # Combine test equity curves
        if all_test_equity:
            combined_equity = pd.concat(all_test_equity, ignore_index=True)
        else:
            combined_equity = pd.DataFrame()

        # Calculate aggregate metrics
        if wf_windows:
            aggregate_metrics = {
                "avg_train_sharpe": sum(w.train_sharpe for w in wf_windows)
                / len(wf_windows),
                "avg_test_sharpe": sum(w.test_sharpe for w in wf_windows)
                / len(wf_windows),
                "avg_degradation": sum(w.sharpe_degradation for w in wf_windows)
                / len(wf_windows),
                "total_test_trades": len(all_test_trades),
                "windows_tested": len(wf_windows),
            }

            if all_test_trades:
                aggregate_metrics["total_test_pnl"] = sum(
                    t.pnl for t in all_test_trades
                )
                aggregate_metrics["test_win_rate"] = (
                    sum(1 for t in all_test_trades if t.pnl > 0)
                    / len(all_test_trades)
                    * 100
                )
        else:
            aggregate_metrics = {}

        self.result = WalkForwardResult(
            windows=wf_windows,
            combined_test_trades=all_test_trades,
            combined_test_equity=combined_equity,
            aggregate_metrics=aggregate_metrics,
        )

        return self.result

    def save_results(
        self,
        filename: Optional[str] = None,
        directory: str = RESULTS_DIR,
    ) -> Path:
        """Save walk-forward results to CSV."""
        if not self.result:
            raise ValueError("No results to save")

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"walk_forward_{timestamp}.csv"

        output_path = output_dir / filename

        df = self.result.to_dataframe()
        df.to_csv(output_path, index=False)

        return output_path

    def print_summary(self) -> None:
        """Print walk-forward optimization summary."""
        if not self.result:
            print("No results available")
            return

        print(f"\n{'=' * 60}")
        print("WALK-FORWARD OPTIMIZATION SUMMARY")
        print(f"{'=' * 60}\n")

        print(f"Windows:        {len(self.result.windows)}")
        print(f"Mode:           {'Anchored' if self.anchored else 'Rolling'}")
        print(f"Objective:      {self.objective}")
        print()

        print("Aggregate Metrics:")
        print(f"  Avg Train Sharpe:  {self.result.avg_train_sharpe:.3f}")
        print(f"  Avg Test Sharpe:   {self.result.avg_test_sharpe:.3f}")
        print(f"  Avg Degradation:   {self.result.avg_degradation:.1f}%")
        print(f"  Robustness Ratio:  {self.result.robustness_ratio:.2f}")
        print()

        print("Per-Window Results:")
        for w in self.result.windows:
            print(f"  Window {w.window_id}:")
            print(f"    Train: {w.train_start} to {w.train_end}")
            print(f"    Test:  {w.test_start} to {w.test_end}")
            print(f"    Train Sharpe: {w.train_sharpe:.3f}")
            print(f"    Test Sharpe:  {w.test_sharpe:.3f}")
            print(f"    Params: {w.best_params}")
            print()
