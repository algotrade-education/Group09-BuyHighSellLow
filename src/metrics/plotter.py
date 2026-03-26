"""
Pluggable chart design.
To add a new chart, just create a ChartBase subclass without modifying BacktestPlotter.

Supports:
    - PNG (matplotlib) - for reports and CI artifacts
    - HTML (plotly)    - interactive and easy to explore

Usage:
    data = PlotData(equity=equity_df, trades=trades, metrics=metrics, benchmark=benchmark)
    plotter = BacktestPlotter(data, output_dir="results/plots")
    plotter.plot_all(fmt="png")                                  # All charts
    plotter.plot(["equity_curve", "drawdown"], fmt="html")        # Selected charts
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.metrics.metrics import PerformanceMetrics
from src.metrics.rolling_metrics import calculate_rolling_metrics
from src.metrics.trade_metrics import Trade

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


# --- Result container ---


class PlotData:
    """
    Data container passed to each chart.
    Separates data preparation from rendering.
    """

    def __init__(
        self,
        equity: pd.DataFrame,
        trades: list[Trade],
        metrics: PerformanceMetrics,
        benchmark: pd.Series | None = None,
        initial_capital: float = 0.0,
    ) -> None:
        self.equity = equity  # DataFrame with 'datetime', 'equity'
        self.trades = trades
        self.metrics = metrics
        self.benchmark = benchmark
        self.initial_capital = initial_capital

        # Pre-compute returns and rolling metrics once
        self._equity_series: pd.Series | None = None
        self._rolling: pd.DataFrame | None = None

    @property
    def equity_series(self) -> pd.Series:
        if self._equity_series is None:
            col = "equity" if "equity" in self.equity.columns else self.equity.columns[0]
            s = self.equity[col]
            # Attach DatetimeIndex so rolling_sharpe can resample to daily
            if "datetime" in self.equity.columns:
                s = s.copy()
                s.index = pd.to_datetime(self.equity["datetime"])
            self._equity_series = s
        return self._equity_series

    @property
    def rolling(self) -> pd.DataFrame:
        if self._rolling is None:
            self._rolling = calculate_rolling_metrics(self.equity_series)
        return self._rolling


# --- Abstract chart ---


class ChartBase(ABC):
    """
    Base class for a chart.
    Each subclass renders a specific chart type.
    """

    name: str = "chart"  # Used as filename
    description: str = ""

    @abstractmethod
    def render_png(self, data: PlotData, output_path: Path) -> bool:
        """Render chart to PNG. True = success."""
        ...

    @abstractmethod
    def render_html(self, data: PlotData, output_path: Path) -> bool:
        """Render chart to interactive HTML. True = success."""
        ...

    def render(self, data: PlotData, output_path: Path, fmt: str) -> bool:
        try:
            if fmt == "html":
                return self.render_html(data, output_path)
            return self.render_png(data, output_path)
        except Exception as e:
            logger.error("Chart '%s' failed: %s", self.name, e, exc_info=True)
            return False

    @staticmethod
    def _get_datetime_index(data: PlotData) -> pd.DatetimeIndex | pd.Series:
        """Extract datetime index from equity data."""
        return pd.to_datetime(data.equity.get("datetime", data.equity.index))


# --- Chart implementations ---


class EquityCurveChart(ChartBase):
    name = "equity_curve"
    description = "Portfolio equity curve vs initial capital"

    def render_png(self, data: PlotData, output_path: Path) -> bool:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(16, 6))
        dt = self._get_datetime_index(data)
        eq = data.equity_series

        ax.plot(dt, eq, color="#2196F3", linewidth=1.5, label="Equity")

        if data.initial_capital > 0:
            ax.axhline(
                data.initial_capital,
                color="#F44336",
                linestyle="--",
                linewidth=1,
                label=f"Initial ({data.initial_capital:,.0f})",
            )

        if data.benchmark is not None:
            # Scale benchmark to same starting value
            scale = eq.iloc[0] / data.benchmark.iloc[0] if data.benchmark.iloc[0] != 0 else 1
            ax.plot(
                dt,
                data.benchmark * scale,
                color="#9E9E9E",
                linewidth=1,
                alpha=0.7,
                linestyle="--",
                label="Benchmark",
            )

        ax.set_title("Equity Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _format_y_axis_millions(ax)

        # Metrics summary in top-left corner
        m = data.metrics
        summary = (
            f"Sharpe: {m.sharpe_ratio:.2f}  |  "
            f"Return: {m.total_return:.2f}%  |  "
            f"MaxDD: {abs(m.max_drawdown):.2f}%  |  "
            f"Trades: {m.total_trades}  |  "
            f"WR: {m.win_rate:.1f}%"
        )
        ax.text(
            0.01,
            0.02,
            summary,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True

    def render_html(self, data: PlotData, output_path: Path) -> bool:
        import plotly.graph_objects as go

        dt = self._get_datetime_index(data)
        eq = data.equity_series
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=eq,
                name="Equity",
                line=dict(color="#2196F3", width=1.5),
            )
        )
        if data.initial_capital > 0:
            fig.add_hline(
                y=data.initial_capital,
                line_dash="dash",
                line_color="#F44336",
                annotation_text=f"Initial ({data.initial_capital:,.0f})",
            )
        if data.benchmark is not None:
            scale = eq.iloc[0] / data.benchmark.iloc[0] if data.benchmark.iloc[0] != 0 else 1
            fig.add_trace(
                go.Scatter(
                    x=dt,
                    y=data.benchmark * scale,
                    name="Benchmark",
                    line=dict(color="#9E9E9E", width=1, dash="dash"),
                    opacity=0.7,
                )
            )
        fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Equity")
        fig.write_html(str(output_path))
        return True


class DrawdownChart(ChartBase):
    name = "drawdown"
    description = "Drawdown from peak"

    def render_png(self, data: PlotData, output_path: Path) -> bool:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(16, 4))
        dt = self._get_datetime_index(data)
        dd = data.rolling["rolling_drawdown"] * 100  # negative %

        ax.fill_between(dt, dd, 0, color="#F44336", alpha=0.4, label="Drawdown")
        ax.plot(dt, dd, color="#F44336", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)

        # max_drawdown is stored as negative — abs() for display
        m = abs(data.metrics.max_drawdown)
        ax.set_title(f"Drawdown (Max: -{m:.2f}%)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True

    def render_html(self, data: PlotData, output_path: Path) -> bool:
        import plotly.graph_objects as go

        dt = self._get_datetime_index(data)
        dd = data.rolling["rolling_drawdown"] * 100
        m = abs(data.metrics.max_drawdown)
        fig = go.Figure(
            go.Scatter(
                x=dt,
                y=dd,
                fill="tozeroy",
                name="Drawdown",
                line=dict(color="#F44336"),
            )
        )
        fig.update_layout(
            title=f"Drawdown (Max: -{m:.2f}%)",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
        )
        fig.write_html(str(output_path))
        return True


class TradeMarkersChart(ChartBase):
    name = "trade_markers"
    description = "Price chart with entry/exit trade markers"

    def render_png(self, data: PlotData, output_path: Path) -> bool:
        import matplotlib.pyplot as plt

        if not data.trades:
            logger.warning("TradeMarkersChart: no trades to plot.")
            return False

        if "close_price" not in data.equity.columns:
            logger.warning("TradeMarkersChart: 'close_price' column missing from equity DataFrame.")
            return False

        fig, ax = plt.subplots(figsize=(20, 8))
        dt = self._get_datetime_index(data)
        ax.plot(
            dt,
            data.equity["close_price"],
            color="black",
            linewidth=0.8,
            alpha=0.6,
            label="Close Price",
        )

        # Markers at actual price level (entry_price / exit_price)
        long_entries = [
            (t.entry_time, t.entry_price)
            for t in data.trades
            if t.side.value == "long" and t.entry_time is not None
        ]
        short_entries = [
            (t.entry_time, t.entry_price)
            for t in data.trades
            if t.side.value == "short" and t.entry_time is not None
        ]
        exits_win = [
            (t.exit_time, t.exit_price)
            for t in data.trades
            if t.is_winner and t.is_closed and t.exit_time is not None
        ]
        exits_loss = [
            (t.exit_time, t.exit_price)
            for t in data.trades
            if t.is_loser and t.is_closed and t.exit_time is not None
        ]

        def scatter(
            points: list[tuple],
            marker: str,
            color: str,
            label: str,
            size: int = 40,
            alpha: float = 0.8,
        ) -> None:
            if points:
                xs, ys = zip(*points, strict=False)
                ax.scatter(
                    xs,
                    ys,
                    marker=marker,
                    color=color,
                    s=size,
                    label=label,
                    zorder=5,
                    alpha=alpha,
                    linewidths=0.5,
                )

        scatter(long_entries, "^", "#4CAF50", "Long Entry")
        scatter(short_entries, "v", "#F44336", "Short Entry")
        scatter(exits_win, "x", "#2196F3", "Exit (Win)", size=30)
        scatter(exits_loss, "x", "#FF5722", "Exit (Loss)", size=30)

        ax.set_title("Price with Trade Markers", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True

    def render_html(self, data: PlotData, output_path: Path) -> bool:
        import plotly.graph_objects as go

        if not data.trades:
            logger.warning("TradeMarkersChart: no trades to plot.")
            return False

        if "close_price" not in data.equity.columns:
            logger.warning("TradeMarkersChart: 'close_price' column missing from equity DataFrame.")
            return False

        dt = self._get_datetime_index(data)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=data.equity["close_price"],
                name="Close Price",
                line=dict(color="black", width=1),
            )
        )

        long_entries = [
            (t.entry_time, t.entry_price)
            for t in data.trades
            if t.side.value == "long" and t.entry_time is not None
        ]
        short_entries = [
            (t.entry_time, t.entry_price)
            for t in data.trades
            if t.side.value == "short" and t.entry_time is not None
        ]
        exits_win = [
            (t.exit_time, t.exit_price)
            for t in data.trades
            if t.is_winner and t.is_closed and t.exit_time is not None
        ]
        exits_loss = [
            (t.exit_time, t.exit_price)
            for t in data.trades
            if t.is_loser and t.is_closed and t.exit_time is not None
        ]

        def add_trace(points: list[tuple], name: str, color: str, symbol: str, size: int) -> None:
            if not points:
                return
            xs, ys = zip(*points, strict=False)
            fig.add_trace(
                go.Scatter(
                    x=list(xs),
                    y=list(ys),
                    mode="markers",
                    name=name,
                    marker=dict(color=color, symbol=symbol, size=size, opacity=0.8),
                )
            )

        add_trace(long_entries, "Long Entry", "#4CAF50", "triangle-up", 9)
        add_trace(short_entries, "Short Entry", "#F44336", "triangle-down", 9)
        add_trace(exits_win, "Exit (Win)", "#2196F3", "x", 8)
        add_trace(exits_loss, "Exit (Loss)", "#FF5722", "x", 8)

        fig.update_layout(
            title="Price with Trade Markers",
            xaxis_title="Date",
            yaxis_title="Price",
        )
        fig.write_html(str(output_path))
        return True


class PnLBarChart(ChartBase):
    name = "pnl_per_trade"
    description = "PnL per trade bar chart"

    def render_png(self, data: PlotData, output_path: Path) -> bool:
        import matplotlib.pyplot as plt

        if not data.trades:
            logger.warning("PnLBarChart: no trades to plot.")
            return False

        closed = [t for t in data.trades if t.is_closed]
        if not closed:
            logger.warning("PnLBarChart: no closed trades to plot.")
            return False
        pnls = [t.pnl for t in closed]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]

        fig, ax = plt.subplots(figsize=(max(12, len(pnls) * 0.15), 5))
        ax.bar(range(1, len(pnls) + 1), pnls, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)

        # Cumulative PnL overlay on secondary axis
        cumulative = pd.Series(pnls).cumsum()
        ax2 = ax.twinx()
        ax2.plot(
            range(1, len(pnls) + 1),
            cumulative,
            color="#2196F3",
            linewidth=1.5,
            label="Cumulative PnL",
            alpha=0.8,
        )
        ax2.set_ylabel("Cumulative PnL", color="#2196F3")
        ax2.tick_params(axis="y", labelcolor="#2196F3")
        _format_y_axis_millions(ax2)

        ax.set_title("PnL per Trade", fontsize=14, fontweight="bold")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("PnL")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True

    def render_html(self, data: PlotData, output_path: Path) -> bool:
        import plotly.graph_objects as go

        if not data.trades:
            logger.warning("PnLBarChart: no trades to plot.")
            return False

        closed = [t for t in data.trades if t.is_closed]
        if not closed:
            logger.warning("PnLBarChart: no closed trades to plot.")
            return False
        pnls = [t.pnl for t in closed]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        fig = go.Figure(
            go.Bar(
                x=list(range(1, len(pnls) + 1)),
                y=pnls,
                marker_color=colors,
                name="PnL",
            )
        )
        fig.update_layout(title="PnL per Trade", xaxis_title="Trade #", yaxis_title="PnL")
        fig.write_html(str(output_path))
        return True


class ExitReasonsChart(ChartBase):
    name = "exit_reasons"
    description = "Distribution of exit reasons"

    def render_png(self, data: PlotData, output_path: Path) -> bool:
        import matplotlib.pyplot as plt

        reasons = _categorize_exit_reasons(data.trades)
        if not reasons:
            logger.warning("ExitReasonsChart: no exit reasons to plot.")
            return False

        counts = pd.Series(reasons).value_counts()
        colors = ["#4CAF50", "#F44336", "#2196F3", "#FF9800", "#9C27B0"]

        fig, ax = plt.subplots(figsize=(7, 7))
        counts.plot.pie(
            ax=ax,
            autopct=lambda p: f"{p:.1f}%\n({int(round(p * sum(counts) / 100))})",
            colors=colors[: len(counts)],
            startangle=90,
            wedgeprops={"edgecolor": "white"},
        )
        ax.set_title("Exit Reasons", fontsize=14, fontweight="bold")
        ax.set_ylabel("")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True

    def render_html(self, data: PlotData, output_path: Path) -> bool:
        import plotly.graph_objects as go

        reasons = _categorize_exit_reasons(data.trades)
        if not reasons:
            logger.warning("ExitReasonsChart: no exit reasons to plot.")
            return False

        counts = pd.Series(reasons).value_counts()
        fig = go.Figure(
            go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                hole=0.3,
            )
        )
        fig.update_layout(title="Exit Reasons")
        fig.write_html(str(output_path))
        return True


class RollingSharpeChart(ChartBase):
    name = "rolling_sharpe"
    description = "Rolling Sharpe ratio over time"

    def render_png(self, data: PlotData, output_path: Path) -> bool:
        import matplotlib.pyplot as plt

        rs = data.rolling.get("rolling_sharpe")
        if rs is None or rs.dropna().empty:
            logger.warning("RollingSharpeChart: no rolling sharpe data to plot.")
            return False

        dt = self._get_datetime_index(data)
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(dt, rs, color="#673AB7", linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axhline(1, color="#4CAF50", linewidth=0.5, linestyle="--", alpha=0.7, label="Sharpe=1")
        ax.fill_between(dt, rs, 0, where=rs >= 0, color="#4CAF50", alpha=0.1)
        ax.fill_between(dt, rs, 0, where=rs < 0, color="#F44336", alpha=0.1)
        ax.set_title("Rolling Sharpe Ratio", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True

    def render_html(self, data: PlotData, output_path: Path) -> bool:
        import plotly.graph_objects as go

        rs = data.rolling.get("rolling_sharpe")
        if rs is None or rs.dropna().empty:
            logger.warning("RollingSharpeChart: no rolling sharpe data to plot.")
            return False
        dt = self._get_datetime_index(data)
        rs_positive = rs.where(rs >= 0, other=0)
        rs_negative = rs.where(rs < 0, other=0)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=rs_positive,
                mode="lines",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor="rgba(76, 175, 80, 0.10)",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=rs_negative,
                mode="lines",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor="rgba(244, 67, 54, 0.10)",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(x=dt, y=rs, name="Rolling Sharpe", line=dict(color="#673AB7", width=1.2))
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.add_hline(y=1, line_dash="dash", line_color="#4CAF50", opacity=0.7)
        fig.update_layout(title="Rolling Sharpe Ratio", xaxis_title="Date", yaxis_title="Sharpe")
        fig.write_html(str(output_path))
        return True


class MonthlyReturnsChart(ChartBase):
    name = "monthly_returns"
    description = "Monthly returns heatmap"

    @staticmethod
    def _build_monthly_returns(data: PlotData) -> pd.DataFrame | None:
        """Build month x year pivot of monthly returns (%)."""
        eq = data.equity_series.copy()
        dt = pd.to_datetime(
            data.equity["datetime"] if "datetime" in data.equity.columns else data.equity.index
        )
        eq.index = dt

        # Resample to month-end equity, then compute monthly return
        monthly = eq.resample("ME").last().dropna()
        if len(monthly) < 2:
            return None

        ret = monthly.pct_change().dropna() * 100
        ret.index = pd.to_datetime(ret.index)

        pivot = ret.groupby([ret.index.year, ret.index.month]).first().unstack(level=1)
        pivot.columns = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ][: len(pivot.columns)]
        return pivot

    def render_png(self, data: PlotData, output_path: Path) -> bool:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        pivot = self._build_monthly_returns(data)
        if pivot is None or pivot.empty:
            logger.warning("MonthlyReturnsChart: not enough data for monthly heatmap.")
            return False

        fig, ax = plt.subplots(
            figsize=(max(10, len(pivot.columns) * 0.9), max(4, len(pivot) * 0.6))
        )

        vmax = max(
            abs(pivot.values[~pd.isna(pivot.values)].max()),
            abs(pivot.values[~pd.isna(pivot.values)].min()),
            1,
        )
        cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#F44336", "#FFFFFF", "#4CAF50"])

        im = ax.imshow(pivot.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="Return (%)", shrink=0.8)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not pd.isna(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if abs(val) < vmax * 0.6 else "white",
                    )

        ax.set_title("Monthly Returns (%)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True

    def render_html(self, data: PlotData, output_path: Path) -> bool:
        import plotly.graph_objects as go

        pivot = self._build_monthly_returns(data)
        if pivot is None or pivot.empty:
            logger.warning("MonthlyReturnsChart: not enough data for monthly heatmap.")
            return False

        text = [[f"{v:.1f}%" if not pd.isna(v) else "" for v in row] for row in pivot.values]
        fig = go.Figure(
            go.Heatmap(
                z=pivot.values.tolist(),
                x=pivot.columns.tolist(),
                y=[str(y) for y in pivot.index],
                text=text,
                texttemplate="%{text}",
                colorscale=[[0, "#F44336"], [0.5, "#FFFFFF"], [1, "#4CAF50"]],
                zmid=0,
                colorbar=dict(title="Return (%)"),
            )
        )
        fig.update_layout(
            title="Monthly Returns (%)",
            xaxis_title="Month",
            yaxis_title="Year",
        )
        fig.write_html(str(output_path))
        return True


# --- Registry ---

_CHARTS: dict[str, ChartBase] = {
    c.name: c
    for c in [
        EquityCurveChart(),
        DrawdownChart(),
        TradeMarkersChart(),
        PnLBarChart(),
        ExitReasonsChart(),
        RollingSharpeChart(),
        MonthlyReturnsChart(),
    ]
}


def register_chart(chart: ChartBase) -> None:
    """Add a custom chart to the registry."""
    _CHARTS[chart.name] = chart


# --- Plotter ---


class BacktestPlotter:
    """
    Orchestrator - combines multiple charts into one report.

    Usage:
        plotter = BacktestPlotter(plot_data, output_dir="results/plots")
        paths = plotter.plot_all(fmt="png")
        paths = plotter.plot(["equity_curve", "drawdown"], fmt="html")
    """

    DEFAULT_CHARTS = [
        "equity_curve",
        "drawdown",
        "trade_markers",
        "pnl_per_trade",
        "exit_reasons",
        "rolling_sharpe",
        "monthly_returns",
    ]

    def __init__(
        self,
        data: PlotData,
        output_dir: str | Path = "results/plots",
    ) -> None:
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Apply consistent style
        try:
            import matplotlib.pyplot as plt

            plt.style.use("ggplot")
        except Exception:
            pass

    def plot_all(self, fmt: str = "png") -> list[Path]:
        """Render all charts in DEFAULT_CHARTS."""
        return self.plot(self.DEFAULT_CHARTS, fmt=fmt)

    def plot(
        self,
        chart_names: Sequence[str] | None = None,
        fmt: str = "png",
    ) -> list[Path]:
        """
        Render a list of charts.

        Args:
            chart_names: List of chart names. None = all registered charts.
            fmt:         "png" or "html".

        Returns:
            List of paths to generated files.
        """
        names = chart_names if chart_names is not None else list(_CHARTS.keys())
        ext = ".html" if fmt == "html" else ".png"
        created = []

        for name in names:
            chart = _CHARTS.get(name)
            if chart is None:
                logger.warning("Chart '%s' not found. Available: %s", name, list(_CHARTS.keys()))
                continue

            output_path = self.output_dir / f"{name}{ext}"
            success = chart.render(self.data, output_path, fmt=fmt)

            if success:
                created.append(output_path)
                logger.info("Chart saved: %s", output_path)

        return created


# --- Helpers ---


def _categorize_exit_reasons(trades: list[Trade]) -> list[str]:
    categories = []
    for t in trades:
        reason = (t.exit_reason or "").lower()
        if "take profit" in reason or " tp" in reason:
            categories.append("Take Profit")
        elif "stop loss" in reason or " sl" in reason:
            categories.append("Stop Loss")
        elif "eod" in reason or "session" in reason or "close" in reason:
            categories.append("EOD / Session Close")
        elif "trail" in reason:
            categories.append("Trailing Stop")
        else:
            categories.append("Other")

    return categories


def _format_y_axis_millions(ax: Axes) -> None:
    """Format y-axis ticks as M/K for readability."""
    from matplotlib.ticker import FuncFormatter

    def formatter(x: float, _: Any) -> str:
        if abs(x) >= 1_000_000:
            return f"{x / 1_000_000:.1f}M"
        if abs(x) >= 1_000:
            return f"{x / 1_000:.0f}K"
        return f"{x:.0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(formatter))
