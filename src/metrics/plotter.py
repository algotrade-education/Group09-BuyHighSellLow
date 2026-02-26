import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from src.engine.position import Trade

logger = logging.getLogger(__name__)


class BacktestPlotter:
    """
    Handles visualization of backtest results.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the plotter.

        Args:
            output_dir: Directory to save plots to.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Set style
        plt.style.use("ggplot")
        plt.rcParams["figure.figsize"] = (20, 12)

    def plot_equity_curve(
        self,
        equity_df: pd.DataFrame,
        initial_capital: float,
        filename: str = "equity_curve.png",
    ) -> None:
        """
        Plot portfolio equity curve vs initial capital.

        Args:
            equity_df: DataFrame containing 'timestamp' and 'equity' columns.
            initial_capital: Initial capital amount.
            filename: Output filename.
        """
        if equity_df.empty:
            logger.warning("No equity data to plot.")
            return

        try:
            plt.figure(figsize=(20, 10))

            # Convert timestamp if needed
            if not pd.api.types.is_datetime64_any_dtype(equity_df["datetime"]):
                equity_df["datetime"] = pd.to_datetime(equity_df["datetime"])

            # Plot Equity
            plt.plot(
                equity_df["datetime"],
                equity_df["equity"],
                label="Portfolio Equity",
                color="blue",
                linewidth=1.5,
            )

            # Plot Initial Capital (Red Line)
            plt.axhline(
                y=initial_capital,
                color="red",
                linestyle="--",
                label=f"Initial Capital ({initial_capital:,.0f})",
            )

            plt.title("Portfolio Equity Curve", fontsize=16)
            plt.xlabel("Date")
            plt.ylabel("Equity")
            plt.legend()
            plt.grid(True, which="both", linestyle="--", alpha=0.7)

            output_path = self.output_dir / filename
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            logger.info("Equity curve plot saved to %s", output_path)

        except Exception as e:
            logger.error("Error plotting equity curve: %s", e)

    def plot_backtest_results(
        self,
        data: pd.DataFrame,
        trades: List[Trade],
        filename: str = "backtest_results.png",
    ) -> None:
        """
        Plot Price, Indicators (SMA, BB), and Buy/Sell signals.

        Args:
            data: DataFrame containing 'datetime', 'close', 'sma', 'upper_band', 'lower_band'.
            trades: List of Trade objects.
            filename: Output filename.
        """
        if data.empty:
            logger.warning("No data to plot backtest results.")
            return

        try:
            plt.figure(figsize=(48, 12))

            # Ensure datetime is datetime object
            if not pd.api.types.is_datetime64_any_dtype(data["datetime"]):
                data["datetime"] = pd.to_datetime(data["datetime"])

            # Plot Price
            plt.plot(
                data["datetime"],
                data["close"],
                label="Close Price",
                color="black",
                alpha=0.6,
                linewidth=1,
            )

            # Plot SMA
            # if "bb_middle" in data.columns:
            #     plt.plot(
            #         data["datetime"],
            #         data["bb_middle"],
            #         label="SMA",
            #         color="blue",
            #         linestyle="-",
            #         alpha=0.8,
            #     )

            # Plot Bollinger Bands
            # if "bb_upper" in data.columns and "bb_lower" in data.columns:
            #     plt.plot(
            #         data["datetime"],
            #         data["bb_upper"],
            #         label="Upper BB",
            #         color="green",
            #         linestyle="-",
            #         alpha=0.6,
            #     )
            #     plt.plot(
            #         data["datetime"],
            #         data["bb_lower"],
            #         label="Lower BB",
            #         color="red",
            #         linestyle="-",
            #         alpha=0.6,
            #     )
            #     plt.fill_between(
            #         data["datetime"],
            #         data["bb_upper"],
            #         data["bb_lower"],
            #         color="blue",
            #         alpha=0.1,
            #     )

            # Plot Buy/Sell Signals from Trades
            buy_times = []
            buy_prices = []
            sell_times = []
            sell_prices = []
            exit_buy_times = []
            exit_buy_prices = []
            exit_sell_times = []
            exit_sell_prices = []

            for trade in trades:
                if trade.side.name == "LONG":
                    # Entry Buy
                    buy_times.append(trade.entry_time)
                    buy_prices.append(trade.entry_price)
                    # Exit Buy (Sell to close)
                    exit_buy_times.append(trade.exit_time)
                    exit_buy_prices.append(trade.exit_price)
                elif trade.side.name == "SHORT":
                    # Entry Sell
                    sell_times.append(trade.entry_time)
                    sell_prices.append(trade.entry_price)
                    # Exit Sell (Buy to close)
                    exit_sell_times.append(trade.exit_time)
                    exit_sell_prices.append(trade.exit_price)

            # Scatter Plots for signals
            if buy_times:
                plt.scatter(
                    buy_times,
                    buy_prices,
                    marker="^",
                    color="green",
                    s=100,
                    label="Buy Entry",
                    zorder=5,
                )
            if sell_times:
                plt.scatter(
                    sell_times,
                    sell_prices,
                    marker="v",
                    color="red",
                    s=100,
                    label="Sell Entry",
                    zorder=5,
                )
            if exit_buy_times:
                plt.scatter(
                    exit_buy_times,
                    exit_buy_prices,
                    marker="x",
                    color="black",
                    s=80,
                    label="Exit Buy",
                    zorder=5,
                )
            if exit_sell_times:
                plt.scatter(
                    exit_sell_times,
                    exit_sell_prices,
                    marker="x",
                    color="purple",
                    s=80,
                    label="Exit Sell",
                    zorder=5,
                )

            plt.title("Backtest Results", fontsize=16)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(loc="best")
            plt.grid(True, which="both", linestyle="--", alpha=0.5)

            output_path = self.output_dir / filename
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            logger.info("Backtest results plot saved to %s", output_path)

        except Exception as e:
            logger.error("Error plotting backtest results: %s", e)

    def plot_trade_analysis(
        self, trades: List[Trade], filename: str = "trade_analysis.png"
    ) -> None:
        """
        Plot Trade Analysis: PnL per trade.

        Args:
            trades: List of Trade objects.
            filename: Output filename.
        """
        if not trades:
            logger.warning("No trades to plot analysis.")
            return

        try:
            plt.figure(figsize=(20, 10))

            # Extract PnL data
            trade_pnls = [t.pnl for t in trades]
            trade_nums = range(1, len(trades) + 1)
            colors = ["green" if pnl > 0 else "red" for pnl in trade_pnls]

            # Bar Plot
            plt.bar(trade_nums, trade_pnls, color=colors, alpha=0.7)

            # Zero line
            plt.axhline(0, color="black", linewidth=1, linestyle="-")

            plt.title("PNL per Trade", fontsize=16)
            plt.xlabel("Trade Number")
            plt.ylabel("Profit/Loss")
            plt.grid(True, axis="y", linestyle="--", alpha=0.7)

            output_path = self.output_dir / filename
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            logger.info("Trade analysis plot saved to %s", output_path)

        except Exception as e:
            logger.error("Error plotting trade analysis: %s", e)

    def plot_exit_reasons(
        self, trades: List[Trade], filename: str = "exit_reasons.png"
    ) -> None:
        """
        Plot distribution of trade exit reasons.

        Args:
            trades: List of Trade objects.
            filename: Output filename.
        """
        if not trades:
            logger.warning("No trades to plot exit reasons.")
            return

        try:
            plt.figure(figsize=(8, 8))

            # Extract and categorize exit reasons
            categorized_reasons = []
            for t in trades:
                reason = t.exit_reason.lower() if t.exit_reason else ""

                if "take profit" in reason or "tp" in reason:
                    categorized_reasons.append("Take Profit")
                elif "stop loss" in reason or "sl" in reason:
                    categorized_reasons.append("Stop Loss")
                elif "eod" in reason or "close" in reason:
                    categorized_reasons.append("EOD Close")
                else:
                    categorized_reasons.append("Other")

            if not categorized_reasons:
                logger.warning("No exit reasons found in trades.")
                return

            counts = pd.Series(categorized_reasons).value_counts()

            # Custom autopct to show value and percentage
            def make_autopct(values):
                def my_autopct(pct):
                    total = sum(values)
                    val = int(round(pct * total / 100.0))
                    return "{p:.1f}%\n({v:d})".format(p=pct, v=val)

                return my_autopct

            # Pie Chart
            counts.plot.pie(
                autopct=make_autopct(counts),
                startangle=90,
                cmap="Pastel1",
                wedgeprops={"edgecolor": "black"},
            ) # type: ignore

            plt.title("Trade Exit Reasons", fontsize=24)
            plt.ylabel("")  # Hide y-label

            output_path = self.output_dir / filename
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            logger.info("Exit reasons plot saved to %s", output_path)

        except Exception as e:
            logger.error("Error plotting exit reasons: %s", e)
