from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.metrics.metrics import MetricsCalculator


def _make_intraday_equity(bars_per_day: int = 10, days: int = 3) -> pd.DataFrame:
    """Create deterministic intraday equity curve with datetime stamps."""
    start = datetime(2024, 1, 2, 9, 0, 0)
    datetimes = []
    equity = [100_000.0]

    # Alternating small returns to keep variance non-zero
    returns_pattern = [0.0010, -0.0005, 0.0012, -0.0007, 0.0008]

    index = 0
    for day in range(days):
        day_start = start + timedelta(days=day)
        for bar in range(bars_per_day):
            datetimes.append(day_start + timedelta(minutes=bar))
            if index > 0:
                r = returns_pattern[(index - 1) % len(returns_pattern)]
                equity.append(equity[-1] * (1 + r))
            index += 1

    return pd.DataFrame({"datetime": datetimes, "equity": equity})


def test_intraday_annualization_inferred_from_datetime():
    """Sharpe should use inferred periods-per-year for intraday data."""
    equity_df = _make_intraday_equity(bars_per_day=10, days=3)
    returns = equity_df["equity"].pct_change().dropna()

    expected_ppy = 10 * 252
    expected_sharpe = (returns.mean() / returns.std()) * np.sqrt(expected_ppy)

    metrics = MetricsCalculator().calculate(equity_df)

    assert metrics.sharpe_ratio == pytest.approx(expected_sharpe, rel=1e-10)


def test_max_drawdown_and_longest_drawdown_are_consistent():
    """Drawdown metrics should match known equity path behavior."""
    equity = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=7, freq="D"),
            "equity": [100, 110, 120, 90, 95, 100, 130],
        }
    )

    metrics = MetricsCalculator().calculate(equity)

    # Peak 120 -> trough 90 = -25%
    assert metrics.max_drawdown == pytest.approx(-25.0)
    # Underwater from 90,95,100 until recovery at 130 => 3 periods underwater
    assert metrics.longest_drawdown == 3
