"""
Metrics aggregator - combines all metrics into one interface.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .information_ratio import calculate_information_ratio
from .longest_drawdown import LongestDrawdown
from .maximum_drawdown import MaximumDrawdown
from .returns import (
    calculate_annualized_return,
    calculate_cagr,
    calculate_returns,
    calculate_total_return,
    calculate_volatility,
)
from .sharpe_ratio import SharpeRatio
from .sortino_ratio import SortinoRatio


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""

    # Returns
    total_return: float
    annualized_return: float
    cagr: float
    volatility: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float

    # Drawdown
    max_drawdown: float
    longest_drawdown: int

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Optional
    information_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return_pct": self.total_return,
            "annualized_return_pct": self.annualized_return,
            "cagr_pct": self.cagr,
            "volatility_pct": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown_pct": self.max_drawdown,
            "longest_drawdown": self.longest_drawdown,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "information_ratio": self.information_ratio,
        }

    def __str__(self) -> str:
        """Pretty print metrics."""
        lines = [
            "=" * 50,
            "PERFORMANCE METRICS",
            "=" * 50,
            "",
            "Returns:",
            f"  Total Return:      {self.total_return:>10.2f}%",
            f"  Annualized Return: {self.annualized_return:>10.2f}%",
            f"  CAGR:              {self.cagr:>10.2f}%",
            f"  Volatility:        {self.volatility:>10.2f}%",
            "",
            "Risk-Adjusted:",
            f"  Sharpe Ratio:      {self.sharpe_ratio:>10.2f}",
            f"  Sortino Ratio:     {self.sortino_ratio:>10.2f}",
            "",
            "Drawdown:",
            f"  Max Drawdown:      {self.max_drawdown:>10.2f}%",
            f"  Longest Drawdown:  {self.longest_drawdown:>10d} periods",
            "",
            "Trade Statistics:",
            f"  Total Trades:      {self.total_trades:>10d}",
            f"  Winning Trades:    {self.winning_trades:>10d}",
            f"  Losing Trades:     {self.losing_trades:>10d}",
            f"  Win Rate:          {self.win_rate:>10.2f}%",
            f"  Profit Factor:     {self.profit_factor:>10.2f}",
            f"  Avg Win:           {self.avg_win:>10.2f}",
            f"  Avg Loss:          {self.avg_loss:>10.2f}",
            "=" * 50,
        ]

        if self.information_ratio is not None:
            lines.insert(-1, f"  Information Ratio: {self.information_ratio:>10.2f}")

        return "\n".join(lines)


class MetricsCalculator:
    """
    Calculates all performance metrics from equity curve and trades.
    """

    def __init__(
        self,
        periods_per_year: float = 252.0,
        risk_free_rate: float = 0.0,
    ):
        """
        Initialize calculator.

        Args:
            periods_per_year: Trading periods per year
            risk_free_rate: Annual risk-free rate
        """
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

        # Initialize metrics
        self.sharpe = SharpeRatio(
            annualization_factor=periods_per_year,
            risk_free_rate=risk_free_rate,
        )
        self.sortino = SortinoRatio(
            annualization_factor=periods_per_year,
        )
        self.max_dd = MaximumDrawdown()
        self.longest_dd = LongestDrawdown()

    def _infer_periods_per_year(self, equity: Union[pd.Series, pd.DataFrame]) -> float:
        """
        Infer annualization factor from equity timestamps.

        For intraday data, this estimates bars-per-day from observed data and
        multiplies by 252 trading days. Falls back to configured default when
        timestamp information is unavailable.
        """
        datetimes = None

        if isinstance(equity, pd.DataFrame):
            if "datetime" in equity.columns:
                datetimes = pd.to_datetime(equity["datetime"], errors="coerce")
            elif isinstance(equity.index, pd.DatetimeIndex):
                datetimes = pd.to_datetime(equity.index, errors="coerce")
        elif isinstance(equity, pd.Series) and isinstance(
            equity.index, pd.DatetimeIndex
        ):
            datetimes = pd.to_datetime(equity.index, errors="coerce")

        if datetimes is None:
            return self.periods_per_year

        datetimes = pd.Series(datetimes).dropna()
        if datetimes.empty:
            return self.periods_per_year

        bars_per_day = datetimes.dt.date.value_counts()
        if bars_per_day.empty:
            return self.periods_per_year

        median_bars_per_day = float(bars_per_day.median())
        if median_bars_per_day <= 0:
            return self.periods_per_year

        # If approximately daily data, keep daily convention.
        if median_bars_per_day <= 1.5:
            return 252.0

        return median_bars_per_day * 252.0

    def calculate(
        self,
        equity: Union[pd.Series, pd.DataFrame],
        trades: Optional[List] = None,
        benchmark: Optional[pd.Series] = None,
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            equity: Equity curve (Series or DataFrame with 'equity' column)
            trades: Optional list of Trade objects
            benchmark: Optional benchmark returns for IR calculation

        Returns:
            PerformanceMetrics object
        """
        # Handle DataFrame input
        if isinstance(equity, pd.DataFrame):
            if "equity" in equity.columns:
                equity_series = equity["equity"]
            else:
                equity_series = equity.iloc[:, 0]
        else:
            equity_series = equity

        periods_per_year = self._infer_periods_per_year(equity)

        sharpe_metric = SharpeRatio(
            annualization_factor=periods_per_year,
            risk_free_rate=self.risk_free_rate,
        )
        sortino_metric = SortinoRatio(
            annualization_factor=periods_per_year,
            minimum_acceptable_return=self.risk_free_rate,
        )

        # Calculate returns
        returns = calculate_returns(equity_series)

        # Return metrics
        total_return = calculate_total_return(equity_series)
        annualized_return = calculate_annualized_return(returns, periods_per_year)
        cagr = calculate_cagr(equity_series, periods_per_year)
        volatility = calculate_volatility(returns, True, periods_per_year)

        # Risk-adjusted
        sharpe_ratio = sharpe_metric.calculate(returns)
        sortino_ratio = sortino_metric.calculate(returns)

        # Drawdown
        max_drawdown = self.max_dd.calculate(equity_series)
        longest_drawdown = self.longest_dd.calculate(equity_series)

        # Trade statistics
        if trades:
            closed_trades = [
                t for t in trades if hasattr(t, "is_closed") and t.is_closed
            ]
            total_trades = len(closed_trades)
            winning = [t for t in closed_trades if t.pnl > 0]
            losing = [t for t in closed_trades if t.pnl <= 0]
            winning_trades = len(winning)
            losing_trades = len(losing)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            gross_profit = sum(t.pnl for t in winning)
            gross_loss = abs(sum(t.pnl for t in losing))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        else:
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0

        # Information ratio
        ir = None
        if benchmark is not None:
            ir = calculate_information_ratio(returns, benchmark, periods_per_year)

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=cagr,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            longest_drawdown=longest_drawdown,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            information_ratio=ir,
        )
