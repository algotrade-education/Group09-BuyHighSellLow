"""
src/metrics/rolling_metrics.py

Rolling / time-series metrics.
Static Sharpe does not capture regime changes - rolling metrics
are important for detecting when a strategy is deteriorating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_sharpe(
    equity: pd.Series,
    window: int = 30,
    periods_per_year: float = 12_852.0,  # VN30 5min default
    risk_free_rate: float = 0.0,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Rolling Sharpe Ratio.

    Args:
        equity:           Equity curve Series.
        window:           Rolling window in bars.
        periods_per_year: Annualization factor.
                          VN30 5min = 51 bars/day × 252 = 12,852.
        risk_free_rate:   Annual risk-free rate.
        min_periods:      Minimum bars required. Default = window // 2.

    Returns:
        Series with the same index as equity, NaN when data is insufficient.
    """
    if min_periods is None:
        min_periods = window // 2

    returns = equity.pct_change().dropna()
    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf

    roll_mean = excess.rolling(window=window, min_periods=min_periods).mean()
    roll_std = excess.rolling(window=window, min_periods=min_periods).std()

    # Avoid division by zero
    sharpe = roll_mean / roll_std.replace(0, np.nan) * np.sqrt(periods_per_year)
    return sharpe.rename("rolling_sharpe")


def rolling_drawdown(equity: pd.Series) -> pd.Series:
    """
    Rolling drawdown from running peak.

    Returns:
        Series with negative values (0 = at peak, -0.1 = 10% below peak).
    """
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    return dd.rename("rolling_drawdown")


def rolling_sortino(
    equity: pd.Series,
    window: int = 30,
    periods_per_year: float = 12_852.0,
    min_acceptable_return: float = 0.0,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Rolling Sortino Ratio - penalizes only downside volatility.

    Args:
        min_acceptable_return: MAR per period (default 0).
    """
    if min_periods is None:
        min_periods = window // 2

    returns = equity.pct_change().dropna()
    excess = returns - min_acceptable_return

    # Calculate downside deviation
    # x is excess returns (returns - MAR), so x < 0 means returns < MAR
    def downside_std(x: pd.Series) -> float:
        downside_returns = x[x < 0]
        if len(downside_returns) == 0:
            return float(np.nan)

        return float(downside_returns.std())

    roll_mean = excess.rolling(window=window, min_periods=min_periods).mean()
    roll_downstd = excess.rolling(window=window, min_periods=min_periods).apply(
        downside_std, raw=False
    )

    sortino = roll_mean / roll_downstd.replace(0, np.nan) * np.sqrt(periods_per_year)
    return sortino.rename("rolling_sortino")


def rolling_win_rate(
    pnl_series: pd.Series,
    window: int = 20,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Rolling win rate from a per-trade PnL series.

    Args:
        pnl_series: Series of trade PnL values (one per trade).
        window:     Number of trades in rolling window.
    """
    if min_periods is None:
        min_periods = window // 2

    wins = (pnl_series > 0).astype(float)
    return wins.rolling(window=window, min_periods=min_periods).mean().rename("rolling_win_rate")


def calculate_rolling_metrics(
    equity: pd.Series,
    sharpe_window: int = 30,
    periods_per_year: float = 12_852.0,
) -> pd.DataFrame:
    """
    Convenience function - calculates all rolling metrics at once.

    Returns:
        DataFrame with columns: rolling_sharpe, rolling_drawdown, rolling_sortino.
    """
    result = pd.DataFrame(index=equity.index)
    result["rolling_sharpe"] = rolling_sharpe(equity, sharpe_window, periods_per_year)
    result["rolling_drawdown"] = rolling_drawdown(equity)
    result["rolling_sortino"] = rolling_sortino(equity, sharpe_window, periods_per_year)
    return result
