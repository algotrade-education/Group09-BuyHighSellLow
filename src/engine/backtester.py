import logging
from typing import Callable, Dict, Optional

import pandas as pd

from src.engine.equity_tracker import EquityTracker, SimpleEquityTracker
from src.engine.order import Order
from src.engine.position_sizer import PositionSizer
from src.engine.result import BacktestResult
from src.engine.session_manager import SessionManager, VN30Session
from src.engine.trade_manager import TradeManager
from src.metrics.metrics import MetricsCalculator
from src.strategy.base import Strategy

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtester for evaluating trading strategies on historical data.
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.00015,
        slippage_points: float = 0.5,
        contract_multiplier: float = 1.0,
        margin_rate: float = 0.18,
        position_size: int = 1,
        position_sizer: Optional[PositionSizer] = None,
        order_ttl: int = 0,
        equity_tracker: Optional[EquityTracker] = None,
        session_manager: Optional[SessionManager] = None,
        use_trailing_stop: bool = False,
        trailing_atr_multiplier: float = 2.0,
        max_daily_loss_pct: float = 0.0,
    ) -> None:
        """
        Initialize the backtester.

        Args:
            strategy: Trading strategy to backtest
            initial_capital: Starting capital for backtesting
            commission_rate: Commission rate per trade (e.g., 0.00015 for 0.015%)
            slippage_points: Slippage in price points (e.g., 0.5 for half a point)
            contract_multiplier: Multiplier for contract size (e.g., 1 for index futures)
            margin_rate: Margin requirement as a percentage (e.g., 0.18 for 18%)
            position_size: Fixed number of contracts per trade (default: 1)
            position_sizer: Dynamic position sizer (overrides position_size if provided)
            order_ttl: Time-to-live for pending orders (in number of bars; 0=no expiration)
            equity_tracker: Custom equity tracker (optional)
            session_manager: Custom session manager (optional)
            use_trailing_stop: If True, enable trailing stop loss
            trailing_atr_multiplier: ATR multiplier for trailing distance
            max_daily_loss_pct: Max daily loss as fraction of equity (0=disabled)
        """
        self.strategy = strategy

        self.order_ttl = order_ttl  # 0 = no expiration

        # Trade manager
        self.trade_manager = TradeManager(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_points=slippage_points,
            contract_multiplier=contract_multiplier,
            margin_rate=margin_rate,
            position_size=position_size,
            position_sizer=position_sizer,
            use_trailing_stop=use_trailing_stop,
            trailing_atr_multiplier=trailing_atr_multiplier,
            max_daily_loss_pct=max_daily_loss_pct,
        )

        # Session manager
        if session_manager is None:
            self.session_manager = VN30Session()
        else:
            self.session_manager = session_manager

        # Equity tracker
        if equity_tracker is None:
            self.equity_tracker = SimpleEquityTracker()
        else:
            self.equity_tracker = equity_tracker

        # Metrics calculator
        self.metrics_calculator = MetricsCalculator()

        logger.info(
            "Backtester initialized with %s and %s",
            self.equity_tracker.__class__.__name__,
            {
                "initial_capital": initial_capital,
                "commission_rate": commission_rate,
                "slippage_points": slippage_points,
                "contract_multiplier": contract_multiplier,
                "margin_rate": margin_rate,
                "order_ttl": order_ttl,
            },
        )

    def run(
        self,
        data: pd.DataFrame,
        datetime_column: str = "datetime",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BacktestResult:
        """
        Run the backtest on the provided historical data.

        Args:
            data: Historical price data as a DataFrame
            datetime_column: Name of the column containing datetime information
            progress_callback: Optional callback function for progress updates (current, total)

        Returns:
            BacktestResult: The result of the backtest
        """
        logger.info("Starting backtest with %d data points", len(data))

        required_columns = {datetime_column, "open", "high", "low", "close"}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns for backtest: {sorted(missing_columns)}"
            )

        # Reset state
        self.trade_manager.reset()
        self.equity_tracker.reset()
        Order.reset_id_counter()

        total_bars = len(data)

        # Pre-convert for fast iteration
        bars = data.to_dict("records")
        timestamps = pd.to_datetime(data[datetime_column]).tolist()

        pending_order: Optional[Order] = None
        pending_order_age: int = 0

        # Main Event Loop
        # For each bar, process event in order:
        # 1. Execute pending orders
        # 2. Manage open positions (e.g., check for stop-loss, take-profit, EOD close)
        # 3. Generate new signals (if not skipped by session manager)
        # 4. Update equity curve
        for idx in range(total_bars):
            if progress_callback is not None:
                progress_callback(idx, total_bars)

            bar: dict = bars[idx]
            timestamp = timestamps[idx]

            # --- 0. DAILY P&L TRACKING ---
            self.trade_manager.update_daily_pnl(timestamp)

            # --- 1. EXECUTION ---
            # This first action occured when a signal was generated on the previous bar,
            # so we execute it at the open of the current bar (open price of current bar)
            if pending_order is not None:
                if self.order_ttl > 0 and pending_order_age >= self.order_ttl:
                    logger.debug(
                        "Order expired after %d bars: %s",
                        pending_order_age,
                        pending_order,
                    )
                    pending_order.expire()
                    pending_order = None
                    pending_order_age = 0
                elif self.trade_manager.execute_order(pending_order, bar, timestamp):
                    logger.debug("Order executed: %s", pending_order)
                    self.trade_manager.open_position(pending_order, timestamp)
                    pending_order = None
                    pending_order_age = 0
                else:
                    pending_order_age += 1

            # --- 2. POSITION MANAGEMENT ---
            # Check for stop-loss, take-profit, EOD close, etc.
            was_flat_before = self.trade_manager.position.is_flat
            self.trade_manager.check_sl_tp(bar, timestamp)
            # Record trade P&L if position was just closed by SL/TP
            if not was_flat_before and self.trade_manager.position.is_flat:
                if (
                    self.trade_manager._current_trade is None
                    and self.trade_manager._trades
                ):
                    last_trade = self.trade_manager._trades[-1]
                    if last_trade.pnl != 0:
                        self.trade_manager.record_trade_pnl(last_trade.pnl)

            # Check EOD Close
            if (
                self.session_manager.should_close_eod(timestamp)
                and not self.trade_manager.position.is_flat
            ):
                self.trade_manager.close_position(
                    exit_price=bar["close"],
                    timestamp=timestamp,
                    exit_reason="EOD Close",
                )
                pending_order = None  # Cancel any pending orders at EOD
                pending_order_age = 0

            # --- 3. SIGNAL GENERATION ---
            skip_signal = self.session_manager.should_skip_signal_generation(timestamp)
            if not skip_signal and self.trade_manager.is_daily_loss_hit:
                skip_signal = True  # Max daily loss reached, no new trades
            if not skip_signal:
                try:
                    signal = self.strategy.generate_signal(
                        bar=bar,
                        current_position=self.trade_manager.position,
                    )
                    new_order = self.trade_manager.create_order_from_signal(
                        signal=signal,
                        bar=bar,
                        timestamp=timestamp,
                    )

                    if new_order is not None:
                        if pending_order is None:
                            pending_order = new_order
                            pending_order_age = 0
                            logger.debug("Generated new order: %s", new_order)
                        else:
                            logger.debug(
                                "Skipping new order; pending order exists: %s",
                                pending_order,
                            )

                except KeyError as e:
                    logger.error("Missing key in bar data: %s", e)
                except ValueError as e:
                    logger.error("Error generating signal: %s", e)
                except Exception as e:
                    logger.error(
                        "Unexpected error during signal generation at %s: %s",
                        timestamp,
                        e,
                        exc_info=True,
                    )

            # --- 4. EQUITY UPDATE ---
            self.trade_manager.update_equity(bar["close"])
            self.equity_tracker.record(
                timestamp=timestamp,
                position=self.trade_manager.position.side.value,
                cash=self.trade_manager.cash,
                equity=self.trade_manager.equity,
                unrealized_pnl=self.trade_manager.position.unrealized_pnl,
                close_price=bar["close"],
            )

        # End of Backtest: Close remaining position
        if not self.trade_manager.position.is_flat and len(data) > 0:
            last_timestamp = timestamps[-1]
            last_close = bars[-1]["close"]
            self.trade_manager.close_position(
                exit_price=last_close,
                timestamp=last_timestamp,
                exit_reason="End of Backtest",
            )
            self.trade_manager.update_equity(last_close)
            self.equity_tracker.record(
                timestamp=last_timestamp,
                position=self.trade_manager.position.side.value,
                cash=self.trade_manager.cash,
                equity=self.trade_manager.equity,
                unrealized_pnl=self.trade_manager.position.unrealized_pnl,
                close_price=last_close,
            )

        # Build results
        equity_df = self.equity_tracker.to_dataframe()
        metrics = self._calculate_metrics(equity_df)

        return BacktestResult(
            trades=self.trade_manager.trades.copy(),
            equity_curve=equity_df,
            metrics=metrics,
            signals=self.trade_manager.get_signals().copy(),
            parameters=self.strategy.params,
        )

    def _calculate_metrics(self, equity_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics from the equity curve and trade history.

        Uses the internal MetricsCalculator to compute statistics such as CAGR,
        Sharpe Ratio, Max Drawdown, and various trade-level statistics.

        Args:
            equity_df: DataFrame containing the recorded equity curve.

        Returns:
            Dictionary mapping metric names (e.g., 'sharpe_ratio') to their values.
        """
        if equity_df.empty:
            return {}

        return self.metrics_calculator.calculate(
            equity=equity_df, trades=self.trade_manager.trades
        ).to_dict()
