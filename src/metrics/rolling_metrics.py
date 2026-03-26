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
    Rolling Sharpe Ratio - computed on daily returns to avoid noise.

    Bar-level returns (5min) are too noisy for rolling Sharpe: a 30-bar
    window covers only ~30 minutes of trading, and multiplying by
    sqrt(12852) amplifies noise to ±20. Resampling to daily returns
    first gives a stable, interpretable rolling Sharpe.

    Args:
        equity:           Equity curve Series. If it has a DatetimeIndex,
                          returns are resampled to daily before computing.
        window:           Rolling window in *trading days* (default 30 ≈ 1.5 months).
        periods_per_year: Annualization factor. When resampling to daily,
                          pass 252 (default). The bar-level value (12852) is
                          only used as fallback when no DatetimeIndex is present.
        risk_free_rate:   Annual risk-free rate.
        min_periods:      Minimum days required. Default = window // 2.

    Returns:
        Series aligned to the original equity index, NaN when insufficient data.
    """
    if min_periods is None:
        min_periods = window // 2

    if isinstance(equity.index, pd.DatetimeIndex):
        # Resample to daily - much more stable than bar-level
        daily = equity.resample("B").last().dropna()
        ann = 252.0
    else:
        daily = equity
        ann = periods_per_year  # bar-level fallback

    daily_returns = daily.pct_change().dropna()
    daily_rf = risk_free_rate / ann
    excess = daily_returns - daily_rf

    roll_mean = excess.rolling(window=window, min_periods=min_periods).mean()
    roll_std = excess.rolling(window=window, min_periods=min_periods).std()
    sharpe_daily = roll_mean / roll_std.replace(0, np.nan) * np.sqrt(ann)

    if not isinstance(equity.index, pd.DatetimeIndex):
        return sharpe_daily.reindex(equity.index).rename("rolling_sharpe")

    # Map daily Sharpe back to bar-level index using merge_asof
    sharpe_df = sharpe_daily.reset_index()
    sharpe_df.columns = ["date", "sharpe"]
    equity_df = pd.DataFrame({"date": equity.index})
    merged = pd.merge_asof(
        equity_df.sort_values("date"),
        sharpe_df.sort_values("date"),
        on="date",
        direction="backward",
    )
    result = pd.Series(merged["sharpe"].values, index=equity.index)
    return result.rename("rolling_sharpe")


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
    periods_per_year: float = 252.0,  # daily after resampling
) -> pd.DataFrame:
    """
    Convenience function - calculates all rolling metrics at once.

    rolling_sharpe is computed on daily-resampled returns (window = trading days).
    rolling_drawdown and rolling_sortino remain bar-level.

    Returns:
        DataFrame with columns: rolling_sharpe, rolling_drawdown, rolling_sortino.
    """
    result = pd.DataFrame(index=equity.index)
    result["rolling_sharpe"] = rolling_sharpe(equity, sharpe_window, periods_per_year)
    result["rolling_drawdown"] = rolling_drawdown(equity)
    result["rolling_sortino"] = rolling_sortino(equity, sharpe_window, periods_per_year)
    return result
