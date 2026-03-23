"""
Event-driven backtesting engine.

This module implements an event-driven architecture for backtesting,
providing better separation of concerns and easier extensibility compared
to the procedural Backtester.

Architecture:
    DataFeed → MarketEvent
    MarketEvent → StrategyHandler → SignalEvent
    SignalEvent → RiskHandler → OrderEvent
    OrderEvent → SimBrokerHandler → FillEvent
    FillEvent → AccountHandler → (update AccountState)

Key Benefits:
    - Decoupled components (handlers don't know about each other)
    - Easy to test individual handlers in isolation
    - Simple to extend (add new handlers without modifying existing code)
    - Seamless transition to paper/live trading (swap SimBroker with PaperBroker)

Status: Production ready for backtesting.

Comparison:
    Both EventDrivenBacktester and Backtester use the same AccountState,
    so results should be identical on the same data and strategy.

    Usage:
        result_v1 = EventDrivenBacktester(strategy, ...).run(data)
        result_v2 = Backtester(strategy, ...).run(data)
        assert result_v1.total_pnl ≈ result_v2.total_pnl
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from src.engine.account.account import AccountState
from src.engine.core.event_bus import EventBus
from src.engine.core.events import EventType, MarketEvent
from src.engine.core.handlers import (
    AccountHandler,
    RiskHandler,
    StrategyHandler,
)
from src.engine.equity_tracker import SimpleEquityTracker
from src.engine.execution.sim_broker import SimBroker
from src.engine.result import BacktestResult
from src.engine.session import SessionManager, VN30Session
from src.metrics.metrics import MetricsCalculator
from src.strategy.base import StrategyBase

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Event-Driven Backtester
# ──────────────────────────────────────────────────────────────────


class EventDrivenBacktester:
    """
    Event-driven backtesting engine.

    This is an alternative implementation to the procedural Backtester,
    using event-driven architecture for better modularity and extensibility.

    Both implementations use the same AccountState, so results should be
    identical on the same data and strategy.

    Architecture:
        1. Market data arrives → emit MarketEvent
        2. StrategyHandler receives MarketEvent → emit SignalEvent
        3. RiskHandler receives SignalEvent → emit OrderEvent
        4. SimBrokerHandler receives OrderEvent → emit FillEvent
        5. AccountHandler receives FillEvent → update AccountState

    Benefits over procedural approach:
        - Handlers are decoupled and independently testable
        - Easy to add new handlers (e.g., portfolio manager, risk monitor)
        - Seamless transition to paper/live trading (swap SimBroker with PaperBroker)
        - Clear separation of concerns (strategy, risk, execution, accounting)

    Usage:
        ```python
        strategy = ORBStrategy(config)
        account = AccountState(initial_capital=500_000_000)

        backtester = EventDrivenBacktester(
            strategy=strategy,
            account=account,
            session_manager=VN30Session(),
            freq_minutes=5,
        )

        result = backtester.run(data)
        print(result.metrics)
        ```

    Comparison with Backtester:
        ```python
        result_v1 = EventDrivenBacktester(strategy, account).run(data)
        result_v2 = Backtester(strategy, initial_capital=500_000_000).run(data)

        # Should produce identical results
        assert abs(result_v1.total_pnl - result_v2.total_pnl) < 1.0
        ```
    """

    def __init__(
        self,
        strategy: StrategyBase,
        account: AccountState,
        session_manager: SessionManager | None = None,
        freq_minutes: int = 5,
    ) -> None:
        """
        Initialize event-driven backtester.

        Args:
            strategy: Trading strategy instance
            account: Account state (manages cash, position, trades)
            session_manager: Session manager for market hours (default: VN30Session)
            freq_minutes: Bar frequency in minutes (for metrics calculation)
        """
        self._strategy = strategy
        self._account = account
        self._session_manager = session_manager or VN30Session()
        self._equity_tracker = SimpleEquityTracker()
        self._metrics_calc = MetricsCalculator(freq_minutes=freq_minutes)

        # Mutable dict - handlers hold reference, updated each bar
        self._current_bar: dict[str, Any] = {}

        # Build event bus and wire handlers
        self._bus = EventBus()
        self._wire_handlers()

    def _wire_handlers(self) -> None:
        """
        Connect all handlers to the event bus.

        This creates the event pipeline:
            MARKET → Strategy → SIGNAL → Risk → ORDER → Broker → FILL → Account
        """
        strategy_h = StrategyHandler(self._strategy, self._account, self._bus)
        risk_h = RiskHandler(self._account, self._bus, self._current_bar)
        broker = SimBroker(self._account, self._bus)
        account_h = AccountHandler(self._account)

        self._bus.subscribe(EventType.MARKET, strategy_h.on_market)
        self._bus.subscribe(EventType.SIGNAL, risk_h.on_signal)
        self._bus.subscribe(EventType.ORDER, broker.on_order)
        self._bus.subscribe(EventType.FILL, account_h.on_fill)

        # Store broker reference for bar updates
        self._broker = broker

        logger.debug("Event bus wired: %s", self._bus.subscriber_count)

    def run(
        self,
        data: pd.DataFrame,
        datetime_col: str = "datetime",
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BacktestResult:
        """
        Run event-driven backtest on historical data.

        Args:
            data: OHLCV DataFrame with indicators
            datetime_col: Name of datetime column (default: "datetime")
            progress_callback: Optional callback(current, total) for progress tracking

        Returns:
            BacktestResult with trades, equity curve, and metrics

        Execution flow per bar:
            1. Update daily tracking (reset on new trading day)
            2. Check SL/TP (direct call - not event-driven for simplicity)
            3. EOD close if needed (using next bar open)
            4. Emit MarketEvent → triggers entire event pipeline
            5. Update equity and record

        Note:
            SL/TP and EOD close are handled directly (not via events) for simplicity.
            In a full event-driven implementation, these would be:
                - bus.emit(BarCloseEvent) → SLTPHandler
                - bus.emit(EODEvent) → EODHandler
        """
        self._account.reset()
        self._equity_tracker.reset()
        self._strategy.reset()

        timestamps = pd.to_datetime(data[datetime_col]).tolist()
        bars: list[dict[str, Any]] = data.to_dict("records")
        total = len(bars)

        for idx, (bar, timestamp) in enumerate(zip(bars, timestamps, strict=False)):
            # Update shared bar reference (handlers see this)
            self._current_bar.clear()
            self._current_bar.update(bar)

            # Update broker with current bar
            self._broker.update_bar(bar)

            # Daily tracking
            self._account.update_daily(timestamp)

            # SL/TP check - handled directly (not event-driven for simplicity)
            # Could be: bus.emit(BarCloseEvent) → SLTPHandler
            self._account.check_sl_tp(bar, timestamp)

            # EOD close
            if (
                self._session_manager.should_close_eod(timestamp)
                and not self._account.position.is_flat
            ):
                next_open = float(bars[idx + 1]["open"]) if idx + 1 < total else float(bar["close"])
                self._account.close_position(next_open, timestamp, "EOD Close")

            # Skip signal generation?
            skip = (
                self._session_manager.should_skip_signal(timestamp)
                or self._account.is_daily_loss_hit
            )

            # Emit MarketEvent → triggers entire pipeline
            if not skip:
                self._bus.emit(
                    MarketEvent(
                        timestamp=timestamp,
                        bar=bar,
                    )
                )

            # Equity update
            self._account.update_equity(float(bar["close"]))
            self._equity_tracker.record(
                timestamp=timestamp,
                position=self._account.position.side.value,
                cash=self._account.cash,
                equity=self._account.equity,
                unrealized_pnl=self._account.position.unrealized_pnl,
                close_price=float(bar["close"]),
            )

            if progress_callback:
                progress_callback(idx + 1, total)

        # Cleanup: close remaining position
        if not self._account.position.is_flat and total > 0:
            self._account.close_position(
                float(bars[-1]["close"]), timestamps[-1], "End of Backtest"
            )
            self._account.update_equity(float(bars[-1]["close"]))

        # Build result
        equity_df = self._equity_tracker.to_dataframe()
        metrics = self._metrics_calc.calculate(
            equity=equity_df,
            trades=self._account.trades,
        ).to_dict()

        return BacktestResult(
            trades=self._account.trades,
            equity_curve=equity_df,
            metrics=metrics,
            signals=self._account.signals,
            parameters=getattr(self._strategy, "_params", {}),
        )
