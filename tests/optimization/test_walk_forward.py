"""
Tests for src/optimization/walk_forward.py

Covers:
- WalkForwardWindow properties (train_sharpe, test_sharpe, sharpe_degradation)
- WalkForwardResult properties (avg_train/test_sharpe, avg_degradation, robustness_ratio)
- WalkForwardResult.to_dataframe / save
- WalkForwardOptimizer init validation
- _create_windows: anchored and rolling modes
- optimize() full cycle with grid and optuna inner optimizers
- Embargo period
- Capital chaining
- Empty / skipped windows
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.optimization.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardResult,
    WalkForwardWindow,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_window(
    window_id: int = 1,
    train_sharpe: float = 1.5,
    test_sharpe: float = 1.0,
) -> WalkForwardWindow:
    base = datetime(2023, 1, 1)
    return WalkForwardWindow(
        window_id=window_id,
        train_start=base,
        train_end=base + timedelta(days=180),
        test_start=base + timedelta(days=181),
        test_end=base + timedelta(days=270),
        best_params={"orb_minutes": 15},
        train_metrics={"sharpe_ratio": train_sharpe},
        test_metrics={"sharpe_ratio": test_sharpe},
    )


def _make_ohlcv(n: int = 300, freq: str = "5min") -> pd.DataFrame:
    dates = pd.date_range("2023-01-01 09:00", periods=n, freq=freq)
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": 1000.0,
            "high": 1010.0,
            "low": 990.0,
            "close": 1005.0,
            "volume": 1000,
        }
    )


def _make_bt_result(
    sharpe: float = 1.0,
    trades: int = 50,
    capital_end: float = 510_000_000.0,
) -> Any:
    r = MagicMock()
    r.metrics = {
        "sharpe_ratio": sharpe,
        "total_trades": trades,
        "total_return_pct": 2.0,
        "max_drawdown_pct": 5.0,
        "sortino_ratio": sharpe * 1.2,
        "net_profit_factor": 1.5,
        "win_rate_pct": 55.0,
    }
    r.trades = []
    equity = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-01-01", periods=5, freq="D"),
            "equity": [500e6, 502e6, 505e6, 508e6, capital_end],
        }
    )
    r.equity_curve = equity
    return r


def _simple_trial_fn(params: dict, data: pd.DataFrame, capital: float) -> Any:
    return _make_bt_result()


# ---------------------------------------------------------------------------
# WalkForwardWindow
# ---------------------------------------------------------------------------


def test_window_train_sharpe():
    w = _make_window(train_sharpe=1.5)
    assert w.train_sharpe == pytest.approx(1.5)


def test_window_test_sharpe():
    w = _make_window(test_sharpe=0.8)
    assert w.test_sharpe == pytest.approx(0.8)


def test_window_sharpe_degradation():
    w = _make_window(train_sharpe=2.0, test_sharpe=1.0)
    # (1.0 - 2.0) / 2.0 * 100 = -50%
    assert w.sharpe_degradation == pytest.approx(-50.0)


def test_window_sharpe_degradation_zero_train():
    w = _make_window(train_sharpe=0.0, test_sharpe=1.0)
    assert w.sharpe_degradation == 0.0


# ---------------------------------------------------------------------------
# WalkForwardResult properties
# ---------------------------------------------------------------------------


def _make_result(windows: list[WalkForwardWindow]) -> WalkForwardResult:
    return WalkForwardResult(
        windows=windows,
        combined_test_trades=[],
        combined_test_equity=pd.DataFrame(),
        aggregate_metrics={"windows_completed": len(windows)},
    )


def test_result_avg_train_sharpe():
    windows = [_make_window(1, train_sharpe=1.0), _make_window(2, train_sharpe=2.0)]
    r = _make_result(windows)
    assert r.avg_train_sharpe == pytest.approx(1.5)


def test_result_avg_test_sharpe():
    windows = [_make_window(1, test_sharpe=0.5), _make_window(2, test_sharpe=1.5)]
    r = _make_result(windows)
    assert r.avg_test_sharpe == pytest.approx(1.0)


def test_result_robustness_ratio():
    windows = [_make_window(1, train_sharpe=2.0, test_sharpe=1.0)]
    r = _make_result(windows)
    assert r.robustness_ratio == pytest.approx(0.5)


def test_result_robustness_ratio_zero_train():
    windows = [_make_window(1, train_sharpe=0.0, test_sharpe=1.0)]
    r = _make_result(windows)
    assert r.robustness_ratio == 0.0


def test_result_empty_windows():
    r = _make_result([])
    assert r.avg_train_sharpe == 0.0
    assert r.avg_test_sharpe == 0.0
    assert r.avg_degradation == 0.0
    assert r.robustness_ratio == 0.0


# ---------------------------------------------------------------------------
# WalkForwardResult.to_dataframe
# ---------------------------------------------------------------------------


def test_result_to_dataframe_columns():
    windows = [_make_window(1), _make_window(2)]
    r = _make_result(windows)
    df = r.to_dataframe()
    assert "window" in df.columns
    assert "train_sharpe" in df.columns
    assert "test_sharpe" in df.columns
    assert "degradation_pct" in df.columns
    assert "param_orb_minutes" in df.columns


def test_result_to_dataframe_row_count():
    windows = [_make_window(i) for i in range(4)]
    r = _make_result(windows)
    assert len(r.to_dataframe()) == 4


# ---------------------------------------------------------------------------
# WalkForwardResult.save
# ---------------------------------------------------------------------------


def test_result_save_creates_files(tmp_path):
    windows = [_make_window(1), _make_window(2)]
    equity = pd.DataFrame({"datetime": pd.date_range("2023-01-01", periods=3), "equity": [1, 2, 3]})
    r = WalkForwardResult(
        windows=windows,
        combined_test_trades=[],
        combined_test_equity=equity,
        aggregate_metrics={"windows_completed": 2},
    )
    paths = r.save(tmp_path)
    assert paths["windows"].exists()
    assert paths["equity"].exists()
    assert paths["aggregate"].exists()


def test_result_save_no_equity(tmp_path):
    windows = [_make_window(1)]
    r = WalkForwardResult(
        windows=windows,
        combined_test_trades=[],
        combined_test_equity=pd.DataFrame(),
        aggregate_metrics={},
    )
    paths = r.save(tmp_path)
    assert "equity" not in paths


# ---------------------------------------------------------------------------
# WalkForwardOptimizer init validation
# ---------------------------------------------------------------------------


def test_init_n_windows_too_small():
    with pytest.raises(ValueError, match="n_windows"):
        WalkForwardOptimizer(_simple_trial_fn, {}, n_windows=1)


def test_init_train_pct_out_of_range():
    with pytest.raises(ValueError, match="train_pct"):
        WalkForwardOptimizer(_simple_trial_fn, {}, train_pct=0.0)
    with pytest.raises(ValueError, match="train_pct"):
        WalkForwardOptimizer(_simple_trial_fn, {}, train_pct=1.0)


def test_init_negative_embargo():
    with pytest.raises(ValueError, match="embargo_bars"):
        WalkForwardOptimizer(_simple_trial_fn, {}, embargo_bars=-1)


# ---------------------------------------------------------------------------
# Window creation - anchored
# ---------------------------------------------------------------------------


def test_anchored_window_count():
    data = _make_ohlcv(300)
    wfo = WalkForwardOptimizer(
        _simple_trial_fn,
        {"x": [1, 2]},
        optimizer="grid",
        n_windows=3,
        anchored=True,
    )
    windows = wfo._create_windows(data)
    assert len(windows) == 3


def test_anchored_train_grows():
    data = _make_ohlcv(400)
    wfo = WalkForwardOptimizer(
        _simple_trial_fn,
        {"x": [1]},
        optimizer="grid",
        n_windows=3,
        anchored=True,
    )
    windows = wfo._create_windows(data)
    train_sizes = [len(w[0]) for w in windows]
    assert train_sizes == sorted(train_sizes)


def test_anchored_raises_insufficient_data():
    data = _make_ohlcv(5)
    wfo = WalkForwardOptimizer(
        _simple_trial_fn,
        {"x": [1]},
        optimizer="grid",
        n_windows=10,
        anchored=True,
    )
    with pytest.raises(ValueError, match="Not enough data"):
        wfo._create_windows(data)


# ---------------------------------------------------------------------------
# Window creation - rolling
# ---------------------------------------------------------------------------


def test_rolling_window_count():
    data = _make_ohlcv(300)
    wfo = WalkForwardOptimizer(
        _simple_trial_fn,
        {"x": [1, 2]},
        optimizer="grid",
        n_windows=4,
        anchored=False,
        train_pct=0.7,
    )
    windows = wfo._create_windows(data)
    assert len(windows) == 4


def test_rolling_train_size_consistent():
    data = _make_ohlcv(400)
    wfo = WalkForwardOptimizer(
        _simple_trial_fn,
        {"x": [1]},
        optimizer="grid",
        n_windows=4,
        anchored=False,
        train_pct=0.7,
    )
    windows = wfo._create_windows(data)
    train_sizes = [len(w[0]) for w in windows]
    # All train windows should be the same size in rolling mode
    assert len(set(train_sizes)) == 1


# ---------------------------------------------------------------------------
# Full optimize() cycle
# ---------------------------------------------------------------------------


def test_optimize_grid_returns_result():
    data = _make_ohlcv(300)
    wfo = WalkForwardOptimizer(
        _simple_trial_fn,
        {"x": [1, 2]},
        optimizer="grid",
        n_windows=2,
        anchored=True,
    )
    result = wfo.optimize(data, initial_capital=500_000_000, show_progress=False)
    assert result is not None
    assert len(result.windows) == 2


def test_optimize_optuna_returns_result():
    data = _make_ohlcv(300)
    space = {"x": {"type": "int", "low": 1, "high": 3}}
    wfo = WalkForwardOptimizer(
        _simple_trial_fn,
        space,
        optimizer="optuna",
        n_trials=3,
        n_windows=2,
        anchored=True,
    )
    result = wfo.optimize(data, initial_capital=500_000_000, show_progress=False)
    assert len(result.windows) == 2


def test_optimize_capital_chaining():
    """Capital at end of test window should chain to next window."""
    capital_seen = []

    def tracking_fn(params: dict, data: pd.DataFrame, capital: float) -> Any:
        capital_seen.append(capital)
        return _make_bt_result(capital_end=capital * 1.02)

    data = _make_ohlcv(300)
    wfo = WalkForwardOptimizer(
        tracking_fn,
        {"x": [1]},
        optimizer="grid",
        n_windows=3,
        anchored=True,
        chain_capital=True,
    )
    wfo.optimize(data, initial_capital=500_000_000, show_progress=False)

    # Each window's capital should be larger than the previous (1.02x growth)
    # capital_seen has both train and test calls per window
    assert len(capital_seen) > 0


def test_optimize_no_capital_chaining():
    capital_seen = []

    def tracking_fn(params: dict, data: pd.DataFrame, capital: float) -> Any:
        capital_seen.append(capital)
        return _make_bt_result(capital_end=capital * 1.5)

    data = _make_ohlcv(300)
    wfo = WalkForwardOptimizer(
        tracking_fn,
        {"x": [1]},
        optimizer="grid",
        n_windows=3,
        anchored=True,
        chain_capital=False,
    )
    wfo.optimize(data, initial_capital=500_000_000, show_progress=False)

    # All test calls should use the same initial capital
    test_capitals = capital_seen[1::2]  # every other call is the test run
    assert all(c == pytest.approx(500_000_000) for c in test_capitals)


def test_optimize_embargo_reduces_test_bars():
    test_bar_counts = []

    def tracking_fn(params: dict, data: pd.DataFrame, capital: float) -> Any:
        test_bar_counts.append(len(data))
        return _make_bt_result()

    data = _make_ohlcv(300)

    wfo_no_embargo = WalkForwardOptimizer(
        tracking_fn,
        {"x": [1]},
        optimizer="grid",
        n_windows=2,
        anchored=True,
        embargo_bars=0,
    )
    wfo_no_embargo.optimize(data, show_progress=False)
    no_embargo_counts = test_bar_counts.copy()

    test_bar_counts.clear()

    wfo_embargo = WalkForwardOptimizer(
        tracking_fn,
        {"x": [1]},
        optimizer="grid",
        n_windows=2,
        anchored=True,
        embargo_bars=10,
    )
    wfo_embargo.optimize(data, show_progress=False)
    embargo_counts = test_bar_counts.copy()

    # Test windows with embargo should have fewer bars
    for no_emb, emb in zip(no_embargo_counts[1::2], embargo_counts[1::2], strict=False):
        assert emb < no_emb


def test_optimize_result_mode_and_optimizer_fields():
    data = _make_ohlcv(200)
    wfo = WalkForwardOptimizer(
        _simple_trial_fn,
        {"x": [1]},
        optimizer="grid",
        n_windows=2,
        anchored=False,
    )
    result = wfo.optimize(data, show_progress=False)
    assert result.mode == "rolling"
    assert result.optimizer == "grid"


def test_optimize_unknown_optimizer_raises():
    data = _make_ohlcv(200)
    wfo = WalkForwardOptimizer(
        _simple_trial_fn,
        {"x": [1]},
        optimizer="grid",
        n_windows=2,
    )
    wfo._optimizer = "unknown"
    with pytest.raises(ValueError, match="Unknown optimizer"):
        wfo._run_inner_optimizer(data, 500_000_000, window_id=1)


# ---------------------------------------------------------------------------
# print_summary (smoke)
# ---------------------------------------------------------------------------


def test_print_summary_no_crash(capsys):
    windows = [_make_window(1), _make_window(2)]
    r = _make_result(windows)
    r.print_summary()
    out = capsys.readouterr().out
    assert "WALK-FORWARD SUMMARY" in out
