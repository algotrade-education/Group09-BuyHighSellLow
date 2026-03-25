"""
Bar-by-bar backtesting engine.

This module provides the Backtester class for running historical backtests
with realistic order execution, slippage, and session management.

Key features:
- Bar-by-bar execution (no look-ahead bias)
- Pluggable position sizing and slippage models
- Session-aware trading (market hours, EOD close)
- Daily loss limits and risk management
- Trailing stop support
- Order TTL (time-to-live)
- Warmup period support

V2 improvements:
- EOD close uses next bar open (not current close) to avoid look-ahead bias
- Trade P&L recording automatic in AccountState.close_position()
- Bar iteration uses itertuples for better memory efficiency
- Progress callback called at end of loop (not beginning)
- AccountState.reset() automatically resets Order ID counter
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from src.engine.account import AccountState, PositionSizer
from src.engine.equity_tracker import EquityTracker, SimpleEquityTracker
from src.engine.execution.order import Order
from src.engine.execution.slippage import SlippageModel
from src.engine.result import BacktestResult
from src.engine.session import SessionManager, VN30Session
from src.metrics.metrics import MetricsCalculator
from src.strategy.base import StrategyBase

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"open", "high", "low", "close"}


class Backtester:
    """
    Bar-by-bar backtesting engine.

    Executes strategy on historical data with realistic order execution,
    slippage, and session management.

    Execution flow per bar:
    1. Daily tracking reset (new trading day)
    2. Execute pending order (fill at bar open)
    3. Check stop loss / take profit
    4. EOD close (using next bar open to avoid look-ahead bias)
    5. Generate signal
    6. Update equity and record

    Usage:
        ```python
        config   = ORBConfig.from_json("config/strategy_params/orb_default.json")
        strategy = ORBStrategy(config)
        registry = ORBStrategy.build_registry(atr_period=14)

        loader    = DataLoader(data_service)
        pipeline  = DataPipeline(registry)
        raw_data  = loader.load("VN30F1M", "2023-01-01", "2024-12-31")
        processed = preprocessor.prepare(raw_data, freq="5min")
        data      = pipeline.run(processed)

        bt     = Backtester(strategy, initial_capital=500_000_000)
        result = bt.run(data)
        print(result.metrics)
        ```
    """

    def __init__(
        self,
        strategy: StrategyBase,
        initial_capital: float = 500_000_000.0,
        commission_rate: float = 0.00015,
        contract_multiplier: float = 100_000.0,
        margin_rate: float = 0.18,
        position_size: int = 1,
        position_sizer: PositionSizer | None = None,
        slippage_model: SlippageModel | None = None,
        order_ttl_bars: int = 0,
        use_trailing_stop: bool = False,
        trailing_atr_multiplier: float = 2.0,
        max_daily_loss_pct: float = 0.0,
        entry_cutoff_seconds: float = 0.0,
        allow_late_entry: bool = False,
        session_manager: SessionManager | None = None,
        equity_tracker: EquityTracker | None = None,
        freq_minutes: int = 5,
    ) -> None:
        """
        Initialize backtester.

        Args:
            strategy: Trading strategy to backtest
            initial_capital: Starting capital
            commission_rate: Commission as fraction of notional (e.g., 0.00015 = 0.015%)
            contract_multiplier: Contract multiplier for futures
            margin_rate: Margin requirement as fraction (e.g., 0.18 = 18%)
            position_size: Default position size (used if no sizer provided)
            position_sizer: Position sizing strategy (default: FixedSizer)
            slippage_model: Slippage model (default: FixedSlippage)
            order_ttl_bars: Order time-to-live in bars (0 = no expiry)
            use_trailing_stop: Enable trailing stop based on ATR
            trailing_atr_multiplier: ATR multiplier for trailing stop
            max_daily_loss_pct: Max daily loss as % of equity (0 = disabled)
            entry_cutoff_seconds: Block entries within N seconds of session end
            allow_late_entry: Allow entries even within cutoff period
            session_manager: Session manager (default: VN30Session)
            equity_tracker: Equity tracker (default: SimpleEquityTracker)
            freq_minutes: Bar frequency in minutes (for metrics calculation)
        """

        self.strategy = strategy
        self.order_ttl_bars = order_ttl_bars
        self.entry_cutoff_sec = entry_cutoff_seconds
        self.allow_late_entry = allow_late_entry
        self.freq_minutes = freq_minutes

        self.account = AccountState(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            contract_multiplier=contract_multiplier,
            margin_rate=margin_rate,
            position_size=position_size,
            position_sizer=position_sizer,
            slippage_model=slippage_model,
            use_trailing_stop=use_trailing_stop,
            trailing_atr_multiplier=trailing_atr_multiplier,
            max_daily_loss_pct=max_daily_loss_pct,
        )
        self.session_manager = session_manager or VN30Session()
        self.equity_tracker = equity_tracker or SimpleEquityTracker()
        self.metrics_calc = MetricsCalculator(freq_minutes=freq_minutes)

        logger.info(
            "Backtester initialized: strategy=%s, capital=%.0f",
            strategy.name,
            initial_capital,
        )

    def run(
        self,
        data: pd.DataFrame,
        datetime_col: str = "datetime",
        progress_callback: Callable[[int, int], None] | None = None,
        warmup_bars: int = 0,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: OHLCV + indicator DataFrame
            datetime_col: Name of datetime column
            progress_callback: Optional callback(current_bar, total_bars) for progress tracking
            warmup_bars: First N bars are warmup - signals generated but no trades executed

        Returns:
            BacktestResult with trades, equity curve, and performance metrics
        """
        self._validate_data(data, datetime_col)
        logger.info("Running backtest: %d bars", len(data))

        # Reset
        self.account.reset()
        self.equity_tracker.reset()
        self.strategy.reset()

        total_bars = len(data)
        timestamps = pd.to_datetime(data[datetime_col]).tolist()

        # Convert to list of dicts for fast access
        # Note: This uses ~2x memory vs DataFrame but provides O(1) access
        # For large datasets (>1M bars), consider chunked processing
        bars: list[dict[str, Any]] = data.to_dict("records")

        pending_order: Order | None = None
        pending_order_age: int = 0

        for idx in range(total_bars):
            bar = bars[idx]
            timestamp = timestamps[idx]
            is_warmup = idx < warmup_bars

            # ── 0. Daily reset ────────────────────────────────────
            self.account.update_daily(timestamp)

            # ── 1. Execute pending order ──────────────────────────
            if pending_order is not None and not is_warmup:
                if self.order_ttl_bars > 0 and pending_order_age >= self.order_ttl_bars:
                    pending_order.expire()
                    logger.debug("Order expired after %d bars", pending_order_age)
                    pending_order = None
                    pending_order_age = 0
                elif self.account.execute_order(pending_order, bar, timestamp):
                    pending_order = None
                    pending_order_age = 0
                else:
                    pending_order_age += 1

            # ── 2. SL/TP check ────────────────────────────────────
            self.account.check_sl_tp(bar, timestamp)

            # ── 3. EOD close ──────────────────────────────────────
            # V2 fix: Use next bar open to avoid look-ahead bias
            if (
                self.session_manager.should_close_eod(timestamp)
                and not self.account.position.is_flat
                and not is_warmup
            ):
                next_open = self._get_next_open(bars, idx)
                self.account.close_position(
                    exit_price=next_open,
                    timestamp=timestamp,
                    exit_reason="EOD Close",
                )
                # Note: _record_trade_pnl is now called inside close_position
                pending_order = None
                pending_order_age = 0

            # ── 4. Signal generation ──────────────────────────────
            skip = self.session_manager.should_skip_signal(timestamp)
            skip = skip or self.account.is_daily_loss_hit

            if not skip:
                skip = self._is_entry_blocked(timestamp)

            if not skip:
                try:
                    signal = self.strategy.generate_signal(
                        bar=bar,
                        position=self.account.position_snapshot,
                        is_warmup=is_warmup,
                    )
                    if signal.is_entry and not is_warmup:
                        new_order = self.account.create_order(signal, bar, timestamp)
                        if new_order is not None and pending_order is None:
                            pending_order = new_order
                            pending_order_age = 0
                except Exception as e:
                    logger.error(
                        "Signal generation error at %s: %s",
                        timestamp,
                        e,
                        exc_info=True,
                    )

            # ── 5. Equity update ──────────────────────────────────
            self.account.update_equity(float(bar["close"]))
            self.equity_tracker.record(
                timestamp=timestamp,
                position=self.account.position.side.value,
                cash=self.account.cash,
                equity=self.account.equity,
                unrealized_pnl=self.account.position.unrealized_pnl,
                close_price=float(bar["close"]),
            )

            # Progress callback - called at end of loop
            if progress_callback is not None:
                progress_callback(idx + 1, total_bars)

        # ── End of backtest: close remaining position ─────────────
        if not self.account.position.is_flat and total_bars > 0:
            last_ts = timestamps[-1]
            last_close = float(bars[-1]["close"])
            self.account.close_position(last_close, last_ts, "End of Backtest")
            self.account.update_equity(last_close)
            self.equity_tracker.record(
                timestamp=last_ts,
                position=self.account.position.side.value,
                cash=self.account.cash,
                equity=self.account.equity,
                unrealized_pnl=self.account.position.unrealized_pnl,
                close_price=last_close,
            )

        # ── Build result ──────────────────────────────────────────
        equity_df = self.equity_tracker.to_dataframe()
        metrics = self.metrics_calc.calculate(
            equity=equity_df,
            trades=self.account.trades,
        ).to_dict()

        return BacktestResult(
            trades=self.account.trades,
            equity_curve=equity_df,
            metrics=metrics,
            signals=self.account.signals,
            parameters=getattr(self.strategy, "_params", {}),
        )

    # ── Helpers ───────────────────────────────────────────────────

    def _validate_data(self, data: pd.DataFrame, datetime_col: str) -> None:
        if data.empty:
            raise ValueError("Data is empty.")
        missing = (_REQUIRED_COLS | {datetime_col}) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _get_next_open(self, bars: list[dict[str, Any]], idx: int) -> float:
        """
        Get open price of next bar for EOD close.
        Falls back to current close if last bar.
        """
        if idx + 1 < len(bars):
            return float(bars[idx + 1]["open"])
        return float(bars[idx]["close"])

    def _is_entry_blocked(self, timestamp: Any) -> bool:
        if self.entry_cutoff_sec <= 0:
            return False
        dt = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
        return self.session_manager.is_entry_blocked(
            dt=dt,
            cutoff_seconds=self.entry_cutoff_sec,
            allow_late=self.allow_late_entry,
        )
