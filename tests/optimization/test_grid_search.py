"""
Tests for src/optimization/grid_search.py

Covers:
- total_combinations (math.prod)
- Cartesian product generation + deduplication
- Serial and parallel execution
- Failed trial handling
- Sorting by objective
- to_dataframe / save
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.optimization.grid_search import GridSearch
from src.optimization.scoring import INVALID_SCORE, ScorerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bt_result(
    sharpe: float = 1.0,
    total_return: float = 20.0,
    max_dd: float = 10.0,
    trades: int = 50,
    win_rate: float = 55.0,
    pf: float = 1.5,
) -> Any:
    """Minimal BacktestResult-like object."""
    result = MagicMock()
    result.metrics = {
        "sharpe_ratio": sharpe,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd,
        "total_trades": trades,
        "win_rate_pct": win_rate,
        "net_profit_factor": pf,
        "sortino_ratio": sharpe * 1.2,
    }
    return result


def _good_trial_fn(params: dict[str, Any]) -> Any:
    """Always returns a valid backtest result."""
    return _make_bt_result()


def _failing_trial_fn(params: dict[str, Any]) -> Any:
    raise RuntimeError("Simulated trial failure")


# ---------------------------------------------------------------------------
# total_combinations
# ---------------------------------------------------------------------------


def test_total_combinations_single_param():
    grid = GridSearch(_good_trial_fn, {"a": [1, 2, 3]})
    assert grid.total_combinations == 3


def test_total_combinations_cartesian():
    grid = GridSearch(_good_trial_fn, {"a": [1, 2], "b": [10, 20, 30]})
    assert grid.total_combinations == 6


def test_total_combinations_empty():
    grid = GridSearch(_good_trial_fn, {"a": []})
    assert grid.total_combinations == 0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_dedup_removes_duplicates():
    grid = GridSearch(_good_trial_fn, {"a": [1, 1, 2], "b": [10]})
    results = grid.optimize(show_progress=False)
    # [1,10] and [2,10] only - duplicate [1,10] removed
    assert len(results) == 2


def test_no_duplicates_unchanged():
    grid = GridSearch(_good_trial_fn, {"a": [1, 2], "b": [10, 20]})
    results = grid.optimize(show_progress=False)
    assert len(results) == 4


# ---------------------------------------------------------------------------
# Serial execution
# ---------------------------------------------------------------------------


def test_serial_returns_all_results():
    grid = GridSearch(_good_trial_fn, {"a": [1, 2, 3], "b": [10, 20]})
    results = grid.optimize(n_jobs=1, show_progress=False)
    assert len(results) == 6


def test_serial_results_sorted_best_first():
    call_count = [0]

    def varying_trial_fn(params: dict) -> Any:
        call_count[0] += 1
        # Return different sharpe based on param value
        return _make_bt_result(sharpe=float(params["a"]))

    grid = GridSearch(varying_trial_fn, {"a": [1, 2, 3]})
    results = grid.optimize(show_progress=False)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_failed_trials_tracked():
    grid = GridSearch(_failing_trial_fn, {"a": [1, 2, 3]})
    results = grid.optimize(show_progress=False)
    assert len(results) == 0
    assert len(grid.failed_params) == 3


def test_partial_failures():
    def sometimes_fails(params: dict) -> Any:
        if params["a"] == 2:
            raise ValueError("bad param")
        return _make_bt_result()

    grid = GridSearch(sometimes_fails, {"a": [1, 2, 3]})
    results = grid.optimize(show_progress=False)
    assert len(results) == 2
    assert len(grid.failed_params) == 1


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


def test_parallel_same_count_as_serial():
    grid_serial = GridSearch(_good_trial_fn, {"a": [1, 2, 3], "b": [10, 20]})
    grid_parallel = GridSearch(_good_trial_fn, {"a": [1, 2, 3], "b": [10, 20]})

    serial = grid_serial.optimize(n_jobs=1, show_progress=False)
    parallel = grid_parallel.optimize(n_jobs=2, show_progress=False)

    assert len(serial) == len(parallel)


# ---------------------------------------------------------------------------
# Scorer integration
# ---------------------------------------------------------------------------


def test_scorer_filters_invalid_results():
    """Trials with 0 trades should score INVALID_SCORE."""

    def zero_trades_fn(params: dict) -> Any:
        return _make_bt_result(trades=0)

    cfg = ScorerConfig(min_trades=30)
    grid = GridSearch(zero_trades_fn, {"a": [1, 2]}, scorer=cfg)
    results = grid.optimize(show_progress=False)

    for r in results:
        assert r.score == INVALID_SCORE


# ---------------------------------------------------------------------------
# Minimize mode
# ---------------------------------------------------------------------------


def test_minimize_sorts_ascending():
    """minimize=True should sort results ascending by score."""
    call_order = []

    def varying_fn(params: dict) -> Any:
        call_order.append(params["a"])
        # Higher 'a' → higher sharpe → higher score
        return _make_bt_result(sharpe=float(params["a"]))

    grid = GridSearch(varying_fn, {"a": [1, 2, 3]}, minimize=True)
    results = grid.optimize(show_progress=False)
    scores = [r.score for r in results]
    assert scores == sorted(scores)  # ascending when minimize=True


# ---------------------------------------------------------------------------
# best_params / best_result
# ---------------------------------------------------------------------------


def test_best_params_none_before_optimize():
    grid = GridSearch(_good_trial_fn, {"a": [1]})
    assert grid.best_params is None


def test_best_result_after_optimize():
    grid = GridSearch(_good_trial_fn, {"a": [1, 2]})
    grid.optimize(show_progress=False)
    assert grid.best_result is not None
    assert grid.best_params is not None


# ---------------------------------------------------------------------------
# to_dataframe / save
# ---------------------------------------------------------------------------


def test_to_dataframe_shape():
    grid = GridSearch(_good_trial_fn, {"a": [1, 2, 3]})
    grid.optimize(show_progress=False)
    df = grid.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "score" in df.columns
    assert "a" in df.columns


def test_to_dataframe_empty_before_optimize():
    grid = GridSearch(_good_trial_fn, {"a": [1]})
    df = grid.to_dataframe()
    assert df.empty


def test_save_creates_csv(tmp_path):
    grid = GridSearch(_good_trial_fn, {"a": [1, 2]})
    grid.optimize(show_progress=False)
    path = grid.save(tmp_path)
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == 2


def test_save_raises_when_no_results(tmp_path):
    grid = GridSearch(_failing_trial_fn, {"a": [1]})
    grid.optimize(show_progress=False)
    with pytest.raises(ValueError, match="No results"):
        grid.save(tmp_path)
