"""
Performance metrics module.
"""

from .longest_drawdown import LongestDrawdown
from .maximum_drawdown import DrawdownInfo, MaximumDrawdown
from .metrics import MetricsCalculator, PerformanceMetrics
from .plotter import BacktestPlotter, PlotData
from .rolling_metrics import (
    calculate_rolling_metrics,
    rolling_drawdown,
    rolling_sharpe,
    rolling_sortino,
)
from .sharpe_ratio import SharpeRatio, calculate_sharpe_ratio
from .sortino_ratio import SortinoRatio, calculate_sortino_ratio
from .trade_metrics import Trade, TradeSide, calculate_trade_metrics

__all__ = [
    # Main classes
    "MetricsCalculator",
    "PerformanceMetrics",
    "Trade",
    "TradeSide",
    "BacktestPlotter",
    "PlotData",
    # Metrics
    "SharpeRatio",
    "SortinoRatio",
    "MaximumDrawdown",
    "LongestDrawdown",
    "DrawdownInfo",
    # Functions
    "calculate_trade_metrics",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "rolling_sharpe",
    "rolling_drawdown",
    "rolling_sortino",
    "calculate_rolling_metrics",
]
