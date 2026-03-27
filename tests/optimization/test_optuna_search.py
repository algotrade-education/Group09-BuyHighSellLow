"""
Tests for src/optimization/optuna_search.py

Covers:
- OptunaResult.to_dict
- OptunaSearch init + sampler building
- optimize() - basic run, resume, all sampler types
- _sample_params for int / float / categorical / log-scale
- _collect_results
- best_params / best_result properties
- to_dataframe / save
- print_top / print_study_summary (smoke tests)
- ERROR_SCORE returned on trial exception
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.optimization.optuna_search import OptunaResult, OptunaSearch
from src.optimization.scoring import ERROR_SCORE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_SPACE = {
    "x": {"type": "int", "low": 1, "high": 5},
    "y": {"type": "float", "low": 0.1, "high": 1.0},
}


def _make_bt_result(
    sharpe: float = 1.0,
    trades: int = 50,
    total_return: float = 20.0,
    max_dd: float = 10.0,
) -> Any:
    r = MagicMock()
    r.metrics = {
        "sharpe_ratio": sharpe,
        "total_trades": trades,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd,
        "sortino_ratio": sharpe * 1.2,
        "net_profit_factor": 1.5,
        "win_rate_pct": 55.0,
    }
    return r


def _good_trial_fn(params: dict) -> Any:
    return _make_bt_result()


def _failing_trial_fn(params: dict) -> Any:
    raise RuntimeError("Simulated failure")


# ---------------------------------------------------------------------------
# OptunaResult
# ---------------------------------------------------------------------------


def test_optuna_result_to_dict():
    r = OptunaResult(
        trial_number=3,
        params={"x": 2, "y": 0.5},
        score=1.23,
        metrics={"sharpe_ratio": 1.1},
    )
    d = r.to_dict()
    assert d["trial"] == 3
    assert d["score"] == 1.23
    assert d["x"] == 2
    assert d["sharpe_ratio"] == 1.1


# ---------------------------------------------------------------------------
# OptunaSearch init
# ---------------------------------------------------------------------------


def test_raises_without_optuna_mock(monkeypatch):
    """If optuna not available, __init__ should raise ImportError."""
    import src.optimization.optuna_search as mod

    original = mod.OPTUNA_AVAILABLE
    mod.OPTUNA_AVAILABLE = False
    try:
        with pytest.raises(ImportError, match="Optuna"):
            OptunaSearch(_good_trial_fn, _SIMPLE_SPACE)
    finally:
        mod.OPTUNA_AVAILABLE = original


def test_default_startup_trials():
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=200)
    assert search._n_startup_trials == max(20, 200 // 10)


def test_custom_startup_trials():
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=100, n_startup_trials=5)
    assert search._n_startup_trials == 5


def test_default_patience():
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=100)
    assert search._patience == max(100 // 4, 20)


# ---------------------------------------------------------------------------
# optimize() - basic
# ---------------------------------------------------------------------------


def test_optimize_returns_results():
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=5, seed=0)
    results = search.optimize(show_progress=False)
    assert len(results) == 5


def test_optimize_results_sorted_best_first():
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=10, seed=0)
    results = search.optimize(show_progress=False)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_optimize_failing_trial_returns_error_score():
    search = OptunaSearch(_failing_trial_fn, _SIMPLE_SPACE, n_trials=3, seed=0)
    results = search.optimize(show_progress=False)
    for r in results:
        assert r.score == ERROR_SCORE


def test_optimize_sets_study():
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=3, seed=0)
    assert search.study is None
    search.optimize(show_progress=False)
    assert search.study is not None


# ---------------------------------------------------------------------------
# Resume (load_if_exists)
# ---------------------------------------------------------------------------


def test_optimize_resumes_from_storage(tmp_path):
    db = str(tmp_path / "study.db")
    search1 = OptunaSearch(
        _good_trial_fn,
        _SIMPLE_SPACE,
        n_trials=5,
        seed=0,
        study_name="resume_test",
        storage_path=db,
    )
    search1.optimize(show_progress=False)

    # Second run with same storage - should load existing 5 trials
    search2 = OptunaSearch(
        _good_trial_fn,
        _SIMPLE_SPACE,
        n_trials=5,
        seed=0,
        study_name="resume_test",
        storage_path=db,
    )
    results2 = search2.optimize(show_progress=False)
    assert len(results2) == 5  # already done, no new trials


# ---------------------------------------------------------------------------
# Sampler types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampler", ["tpe", "tpe_multivariate", "cmaes", "qmc"])
def test_all_sampler_types_run(sampler):
    space = {"x": {"type": "float", "low": 0.0, "high": 1.0}}
    search = OptunaSearch(
        _good_trial_fn,
        space,
        n_trials=3,
        sampler=sampler,
        seed=0,
    )
    results = search.optimize(show_progress=False)
    assert len(results) == 3


def test_unknown_sampler_raises():
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=1)
    search._sampler_type = "unknown"
    with pytest.raises(ValueError, match="Unknown sampler"):
        search._build_sampler()


# ---------------------------------------------------------------------------
# Param space types
# ---------------------------------------------------------------------------


def test_int_param_sampled_in_range():
    seen = set()

    def capture_fn(params: dict) -> Any:
        seen.add(params["x"])
        return _make_bt_result()

    space = {"x": {"type": "int", "low": 1, "high": 3}}
    search = OptunaSearch(capture_fn, space, n_trials=20, seed=42)
    search.optimize(show_progress=False)
    assert all(1 <= v <= 3 for v in seen)


def test_float_param_sampled_in_range():
    seen = []

    def capture_fn(params: dict) -> Any:
        seen.append(params["y"])
        return _make_bt_result()

    space = {"y": {"type": "float", "low": 0.0, "high": 1.0}}
    search = OptunaSearch(capture_fn, space, n_trials=10, seed=42)
    search.optimize(show_progress=False)
    assert all(0.0 <= v <= 1.0 for v in seen)


def test_categorical_param():
    seen = set()

    def capture_fn(params: dict) -> Any:
        seen.add(params["flag"])
        return _make_bt_result()

    space = {"flag": {"type": "categorical", "choices": [True, False]}}
    search = OptunaSearch(capture_fn, space, n_trials=20, seed=42)
    search.optimize(show_progress=False)
    assert True in seen and False in seen


def test_unknown_param_type_returns_error_score():
    """Unknown param type is caught inside _objective and returns ERROR_SCORE."""
    space = {"z": {"type": "unknown_type", "low": 0, "high": 1}}
    search = OptunaSearch(_good_trial_fn, space, n_trials=1, seed=0)
    results = search.optimize(show_progress=False)
    assert all(r.score == ERROR_SCORE for r in results)


# ---------------------------------------------------------------------------
# best_params / best_result
# ---------------------------------------------------------------------------


def test_best_params_none_before_optimize():
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=3)
    assert search.best_params is None
    assert search.best_result is None


def test_best_result_after_optimize():
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=5, seed=0)
    search.optimize(show_progress=False)
    assert search.best_result is not None
    assert search.best_params is not None
    assert "x" in search.best_params
    assert "y" in search.best_params


# ---------------------------------------------------------------------------
# to_dataframe / save
# ---------------------------------------------------------------------------


def test_to_dataframe_shape():
    import pandas as pd

    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=5, seed=0)
    search.optimize(show_progress=False)
    df = search.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "score" in df.columns


def test_to_dataframe_empty_before_optimize():
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=3)
    df = search.to_dataframe()
    assert df.empty


def test_save_creates_csv(tmp_path):
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=3, seed=0)
    search.optimize(show_progress=False)
    path = search.save(tmp_path)
    assert path.exists()


def test_save_raises_when_no_results(tmp_path):
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=3)
    with pytest.raises(ValueError, match="No results"):
        search.save(tmp_path)


# ---------------------------------------------------------------------------
# print_top / print_study_summary (smoke)
# ---------------------------------------------------------------------------


def test_print_top_no_crash(capsys):
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=3, seed=0)
    search.optimize(show_progress=False)
    search.print_top(n=2)
    out = capsys.readouterr().out
    assert "Rank" in out


def test_print_study_summary_no_crash(capsys):
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=3, seed=0)
    search.optimize(show_progress=False)
    search.print_study_summary()
    out = capsys.readouterr().out
    assert "OPTUNA STUDY SUMMARY" in out


def test_print_top_no_results(capsys):
    search = OptunaSearch(_good_trial_fn, _SIMPLE_SPACE, n_trials=3)
    search.print_top()
    out = capsys.readouterr().out
    assert "No results" in out
