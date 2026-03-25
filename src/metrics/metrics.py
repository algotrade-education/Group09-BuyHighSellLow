"""
Calculate performance metrics from equity curve and trades.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config.schemas.session import VN30SessionConfig
from src.metrics.longest_drawdown import LongestDrawdown
from src.metrics.maximum_drawdown import MaximumDrawdown
from src.metrics.returns import (
    calculate_annualized_return,
    calculate_cagr,
    calculate_returns,
    calculate_total_return,
    calculate_volatility,
)
from src.metrics.sharpe_ratio import SharpeRatio
from src.metrics.sortino_ratio import SortinoRatio
from src.metrics.trade_metrics import Trade, calculate_trade_metrics

logger = logging.getLogger(__name__)

# --- VN30 annualization constants ---
# Hardcoded instead of inferred from data - avoids bias from gaps/holidays.
# Based on VN30SessionConfig: 255 trading minutes/day (excluding ATC).
_VN30_BARS_PER_YEAR = {
    1: VN30SessionConfig().bars_per_year(1),  # 255 bars/day * 252 days
    5: VN30SessionConfig().bars_per_year(5),  # 51
    15: VN30SessionConfig().bars_per_year(15),  # 17
    30: VN30SessionConfig().bars_per_year(
        30
    ),  # 9 (8.5 if we counted the 15min bar that includes the 14:30-14:45 ATC period)
    60: VN30SessionConfig().bars_per_year(
        60
    ),  # 4 (3.5 if we counted the 30min bar that includes the 14:30-14:45 ATC period)
}
_VN30_DEFAULT_BARS_PER_YEAR = _VN30_BARS_PER_YEAR[5]  # 5min is the default


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""

    # --- Returns ---
    total_return: float = 0.0  # %
    annualized_return: float = 0.0  # %
    cagr: float = 0.0  # %
    volatility: float = 0.0  # % annualized

    # --- Risk-adjusted ---
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # --- Drawdown ---
    max_drawdown: float = 0.0  # %
    longest_drawdown: int = 0  # bars

    # --- Trade statistics ---
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate: float = 0.0  # %

    # Profit factor - both versions
    gross_profit_factor: float = 0.0  # gross_pnl before commission
    net_profit_factor: float = 0.0  # net_pnl after commission

    # V1 compatibility alias
    @property
    def profit_factor(self) -> float:
        return self.net_profit_factor

    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 0.0
    expectancy: float = 0.0

    # --- Streaks ---
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # --- Duration ---
    avg_duration_minutes: float = 0.0
    total_commission: float = 0.0

    # --- MAE/MFE ---
    avg_mae: float | None = None
    avg_mfe: float | None = None
    avg_edge_ratio: float | None = None

    # --- Benchmark ---
    information_ratio: float | None = None
    alpha: float | None = None
    beta: float | None = None

    # --- Extra ---
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_return_pct": self.total_return,
            "annualized_return_pct": self.annualized_return,
            "cagr_pct": self.cagr,
            "volatility_pct": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown_pct": self.max_drawdown,
            "longest_drawdown_bars": self.longest_drawdown,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "win_rate_pct": self.win_rate,
            "gross_profit_factor": self.gross_profit_factor,
            "net_profit_factor": self.net_profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "payoff_ratio": self.payoff_ratio,
            "expectancy": self.expectancy,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_duration_minutes": self.avg_duration_minutes,
            "total_commission": self.total_commission,
            "avg_mae": self.avg_mae,
            "avg_mfe": self.avg_mfe,
            "avg_edge_ratio": self.avg_edge_ratio,
            "information_ratio": self.information_ratio,
            "alpha": self.alpha,
            "beta": self.beta,
            **self.extra,
        }

    def __str__(self) -> str:
        def fmt_opt(v: float | None, fmt: str = ".2f") -> str:
            return f"{v:{fmt}}" if v is not None else "N/A"

        lines = [
            "=" * 55,
            "  PERFORMANCE METRICS",
            "=" * 55,
            "",
            "Returns:",
            f"  Total Return:         {self.total_return:>10.2f}%",
            f"  Annualized Return:    {self.annualized_return:>10.2f}%",
            f"  CAGR:                 {self.cagr:>10.2f}%",
            f"  Volatility:           {self.volatility:>10.2f}%",
            "",
            "Risk-Adjusted:",
            f"  Sharpe Ratio:         {self.sharpe_ratio:>10.2f}",
            f"  Sortino Ratio:        {self.sortino_ratio:>10.2f}",
            "",
            "Drawdown:",
            f"  Max Drawdown:         {self.max_drawdown:>10.2f}%",
            f"  Longest Drawdown:     {self.longest_drawdown:>10d} bars",
            "",
            "Trade Statistics:",
            f"  Total Trades:         {self.total_trades:>10d}",
            f"  Win / Loss / Even:    {self.winning_trades:>4d} / {self.losing_trades:>4d} / {self.breakeven_trades:>4d}",
            f"  Win Rate:             {self.win_rate:>10.2f}%",
            f"  Gross Profit Factor:  {self.gross_profit_factor:>10.2f}",
            f"  Net Profit Factor:    {self.net_profit_factor:>10.2f}",
            f"  Avg Win:              {self.avg_win:>10.2f}",
            f"  Avg Loss:             {self.avg_loss:>10.2f}",
            f"  Payoff Ratio:         {self.payoff_ratio:>10.2f}",
            f"  Expectancy:           {self.expectancy:>10.2f}",
            "",
            "Streaks:",
            f"  Max Consec. Wins:     {self.max_consecutive_wins:>10d}",
            f"  Max Consec. Losses:   {self.max_consecutive_losses:>10d}",
            "",
            "Duration & Cost:",
            f"  Avg Duration:         {self.avg_duration_minutes:>10.1f} min",
            f"  Total Commission:     {self.total_commission:>10.0f}",
        ]

        if self.avg_mae is not None:
            lines += [
                "",
                "MAE/MFE:",
                f"  Avg MAE:              {fmt_opt(self.avg_mae):>10}",
                f"  Avg MFE:              {fmt_opt(self.avg_mfe):>10}",
                f"  Avg Edge Ratio:       {fmt_opt(self.avg_edge_ratio):>10}",
            ]

        if self.information_ratio is not None:
            lines += [
                "",
                "Benchmark:",
                f"  Information Ratio:    {fmt_opt(self.information_ratio):>10}",
                f"  Alpha:                {fmt_opt(self.alpha):>10}",
                f"  Beta:                 {fmt_opt(self.beta):>10}",
            ]

        lines.append("=" * 55)
        return "\n".join(lines)


class MetricsCalculator:
    """
    Calculate all performance metrics from equity curve and trades.

    Usage:
        calc = MetricsCalculator(freq_minutes=5)   # VN30 5min
        metrics = calc.calculate(equity_df, trades)
        print(metrics)
    """

    def __init__(
        self,
        freq_minutes: int = 5,
        risk_free_rate: float = 0.0,
        custom_periods_per_year: float | None = None,
    ) -> None:
        """
        Args:
            freq_minutes:             Bar frequency. Used to select annualization factor.
                                      Supported: 1, 5, 15, 30. Default 5.
            risk_free_rate:           Annual risk-free rate.
            custom_periods_per_year:  Override annualization factor for non-VN30 markets.
        """
        if custom_periods_per_year is not None:
            self.periods_per_year = custom_periods_per_year
        else:
            if freq_minutes not in _VN30_BARS_PER_YEAR:
                logger.warning(
                    f"freq_minutes={freq_minutes} not in supported values {list(_VN30_BARS_PER_YEAR.keys())}. "
                    f"Using default {_VN30_DEFAULT_BARS_PER_YEAR} periods/year."
                )
            self.periods_per_year = _VN30_BARS_PER_YEAR.get(
                freq_minutes, _VN30_DEFAULT_BARS_PER_YEAR
            )
        self.risk_free_rate = risk_free_rate

    def calculate(
        self,
        equity: pd.Series | pd.DataFrame,
        trades: Sequence[Trade] | None = None,
        benchmark: pd.Series | None = None,
    ) -> PerformanceMetrics:
        """
        Calculate all metrics.

        Args:
            equity:    Equity curve. Series or DataFrame with an 'equity' column.
            trades:    List of closed Trade objects. None = skip trade metrics.
            benchmark: Benchmark equity curve for IR/alpha/beta. None = skip.

        Returns:
            PerformanceMetrics with all fields populated.
        """
        equity_series = self._extract_equity_series(equity)

        if equity_series.empty or len(equity_series) < 2:
            return PerformanceMetrics()

        returns = calculate_returns(equity_series).dropna()

        # --- Return metrics ---
        total_return = calculate_total_return(equity_series)
        annualized_return = calculate_annualized_return(returns, self.periods_per_year)
        cagr = calculate_cagr(equity_series, self.periods_per_year)
        volatility = calculate_volatility(returns, True, self.periods_per_year)

        # --- Risk-adjusted ---
        sharpe_metric = SharpeRatio(
            annualization_factor=self.periods_per_year,
            risk_free_rate=self.risk_free_rate,
        )
        sortino_metric = SortinoRatio(
            annualization_factor=self.periods_per_year,
            minimum_acceptable_return=self.risk_free_rate,
        )
        sharpe_ratio = sharpe_metric.calculate(returns)
        sortino_ratio = sortino_metric.calculate(returns)

        # --- Drawdown ---
        max_drawdown = MaximumDrawdown().calculate(equity_series)
        longest_drawdown = LongestDrawdown().calculate(equity_series)

        # --- Trade metrics ---
        trade_stats: dict[str, Any] = {}
        if trades:
            trade_stats = calculate_trade_metrics(trades)

        # --- Benchmark ---
        ir = alpha = beta = None
        if benchmark is not None:
            ir, alpha, beta = self._calculate_benchmark_metrics(returns, benchmark)

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=cagr,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            longest_drawdown=longest_drawdown,
            # Trade stats - use get() with defaults
            total_trades=trade_stats.get("total_trades", 0),
            winning_trades=trade_stats.get("winning_trades", 0),
            losing_trades=trade_stats.get("losing_trades", 0),
            breakeven_trades=trade_stats.get("breakeven_trades", 0),
            win_rate=trade_stats.get("win_rate", 0.0),
            gross_profit_factor=trade_stats.get("gross_profit_factor", 0.0),
            net_profit_factor=trade_stats.get("net_profit_factor", 0.0),
            avg_win=trade_stats.get("avg_win", 0.0),
            avg_loss=trade_stats.get("avg_loss", 0.0),
            payoff_ratio=trade_stats.get("payoff_ratio", 0.0),
            expectancy=trade_stats.get("expectancy", 0.0),
            max_consecutive_wins=trade_stats.get("max_consecutive_wins", 0),
            max_consecutive_losses=trade_stats.get("max_consecutive_losses", 0),
            avg_duration_minutes=trade_stats.get("avg_duration_minutes", 0.0),
            total_commission=trade_stats.get("total_commission", 0.0),
            avg_mae=trade_stats.get("avg_mae"),
            avg_mfe=trade_stats.get("avg_mfe"),
            avg_edge_ratio=trade_stats.get("avg_edge_ratio"),
            # Benchmark
            information_ratio=ir,
            alpha=alpha,
            beta=beta,
        )

    # --- Private helpers ---

    @staticmethod
    def _extract_equity_series(equity: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(equity, pd.DataFrame):
            if "equity" in equity.columns:
                return equity["equity"]
            return equity.iloc[:, 0]
        return equity

    def _calculate_benchmark_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_equity: pd.Series,
    ) -> tuple[float | None, float | None, float | None]:
        """Calculate Information Ratio, Alpha, Beta vs benchmark."""
        try:
            bench_returns = calculate_returns(benchmark_equity).dropna()

            # Align
            aligned = pd.concat([strategy_returns, bench_returns], axis=1, join="inner")
            if len(aligned) < 10:
                return None, None, None

            s_ret = aligned.iloc[:, 0]
            b_ret = aligned.iloc[:, 1]

            # IR = (mean excess return) / std(excess return)
            excess = s_ret - b_ret
            ir = (
                excess.mean() / excess.std() * np.sqrt(self.periods_per_year)
                if excess.std() > 0
                else None
            )

            # Beta = cov(s, b) / var(b)
            cov_matrix = np.cov(s_ret, b_ret)
            var_bench = cov_matrix[1, 1]
            beta = cov_matrix[0, 1] / var_bench if var_bench > 0 else None

            # Alpha = annualized excess return over CAPM expected return
            # Alpha = (Rs - Rf) - Beta * (Rb - Rf)
            # Simplified when Rf=0: Alpha = Rs - Beta * Rb
            alpha = None
            if beta is not None:
                # Annualize the alpha (excess return per period * periods per year)
                alpha_per_period = s_ret.mean() - beta * b_ret.mean()
                alpha = alpha_per_period * self.periods_per_year

            return ir, alpha, beta

        except (ValueError, KeyError, ZeroDivisionError) as e:
            logger.debug(f"Benchmark metrics calculation failed: {e}")
            return None, None, None
