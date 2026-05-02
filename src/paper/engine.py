"""
PaperEngine - Thin orchestrator for paper trading system.

The PaperEngine is a lightweight coordinator that wires together the feed,
handlers, and account components. It delegates all business logic to specialized
handlers and maintains minimal state.

Key responsibilities:
- Wire feed subscription to on_bar callback
- Orchestrate handler pipeline: BarHandler → RiskHandler → SignalHandler
- Manage engine lifecycle (start/stop with warmup flow)
- Store deferred exit reasons for cross-bar coordination
- Track background asyncio tasks for clean shutdown

The engine runs on asyncio single-threaded event loop - no threading, no locks.
All concurrency safety relies on cooperative multitasking.

V2 improvements over V1:
- Thin orchestrator: business logic moved to handlers
- Background task tracking: prevents task leaks
- Independent timer task: not blocked by DB queries
- Warmup flow: strategy warmup + reconciliation (bar preload/seed handled upstream)
- Sim mode support: historical replay via SimFeed
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from src.engine.session.base import SessionManager
    from src.paper.account.reconciler import Reconciler
    from src.paper.account.tracker import Tracker
    from src.paper.execution.order_manager import OrderManager
    from src.paper.feeds.base import FeedBase
    from src.paper.handlers.bar_handler import BarHandler
    from src.paper.handlers.risk_handler import RiskHandler
    from src.paper.handlers.signal_handler import SignalHandler
    from src.paper.stats import SessionStats
    from src.strategy.base import StrategyBase

logger = logging.getLogger(__name__)


class PaperEngine:
    """Thin orchestrator for paper trading system.

    Wires feed → handlers → account/orders without implementing business logic.
    All trading decisions are delegated to handlers.

    Handler pipeline on each bar:
    1. BarHandler: Update equity, handle deferred exits
    2. RiskHandler: Check SL/TP, trailing stop, force-flat
    3. SignalHandler: Generate strategy signals, submit orders

    If BarHandler or RiskHandler returns True (exit submitted), skip downstream handlers.
    """

    def __init__(
        self,
        feed: FeedBase,
        bar_handler: BarHandler,
        risk_handler: RiskHandler,
        signal_handler: SignalHandler,
        tracker: Tracker,
        reconciler: Reconciler,
        order_manager: OrderManager,
        stats: SessionStats,
        session_manager: SessionManager,
        strategy: StrategyBase,
        symbol: str,
        close_on_shutdown: bool = True,
        force_hard_exit: bool = False,
        output_dir: str = "results/paper",
    ) -> None:
        """Initialize PaperEngine.

        Args:
            feed: Market data feed (RedisFeed or SimFeed).
            bar_handler: Handler for equity updates and deferred exits.
            risk_handler: Handler for SL/TP and force-flat checks.
            signal_handler: Handler for strategy signal generation.
            tracker: Account tracker for position and P&L.
            reconciler: Broker state reconciler.
            order_manager: Order submission manager.
            stats: Session statistics calculator.
            session_manager: Trading session manager.
            strategy: Trading strategy for warmup.
            symbol: Trading symbol.
            close_on_shutdown: Whether to close open positions on shutdown.
            force_hard_exit: Whether to force hard exit (os._exit) on shutdown.
            output_dir: Output directory for session results.
        """
        self._feed = feed
        self._bar_handler = bar_handler
        self._risk_handler = risk_handler
        self._signal_handler = signal_handler
        self._tracker = tracker
        self._reconciler = reconciler
        self._order_manager = order_manager
        self._stats = stats
        self._session_manager = session_manager
        self._strategy = strategy
        self._symbol = symbol
        self._close_on_shutdown = close_on_shutdown
        self._force_hard_exit = force_hard_exit
        self._output_dir = output_dir

        # State
        self._running = False
        self._last_close: float = 0.0
        self._deferred_exit_reason: str | None = None
        self._bars_processed: int = 0

        # Background task tracking (prevents task leaks)
        self._bg_tasks: set[asyncio.Task] = set()

    # --- Lifecycle ---

    async def start(
        self,
        historical_df: pd.DataFrame | None = None,
        sim_df: pd.DataFrame | None = None,
    ) -> None:
        """Start the engine and begin processing market data.

        Supports three modes:
        - LIVE: historical_df provided, no sim_df
        - DRY-RUN: same as LIVE but orders are simulated
        - SIM: sim_df provided, replays historical data

        Workflow (LIVE/DRY-RUN):
        1. Warmup: strategy state from historical bars
        2. Reconcile broker state (position, cash, orders)
        3. Subscribe to feed with on_bar callback
        4. Feed begins emitting bars → on_bar pipeline executes

        Workflow (SIM):
        1. Replay sim_df through SimFeed
        2. Stop when replay completes

        Args:
            historical_df: Historical bars for indicator warmup (LIVE/DRY-RUN).
            sim_df: Historical dataset for sim mode replay (SIM).
        """
        if self._running:
            logger.warning("Engine already running, ignoring duplicate start")
            return

        self._running = True

        # Mode selection
        if sim_df is not None:
            await self._run_sim(sim_df)
        else:
            await self._run_live(historical_df)

    async def _run_live(
        self,
        historical_df: pd.DataFrame | None = None,
    ) -> None:
        """Run live/dry-run mode with warmup flow.

        Warmup flow:
        1. Warmup strategy state with historical bars
        2. Reconcile broker state
        3. Subscribe to feed

        Note:
            BarAggregator preload/seed is performed during runtime construction
            in ``run_paper_trade`` before ``engine.start()`` is called.
        """
        # Note: BarAggregator warmup is handled in bootstrap/run_paper_trade
        # because BarAggregator is constructed there and passed to RedisFeed.
        # The engine doesn't have direct access to BarAggregator in the current architecture.
        # This is by design - the engine is a thin orchestrator that doesn't manage
        # low-level feed details.

        # 1. Warmup: strategy state
        if historical_df is not None and not historical_df.empty:
            logger.info("Warming up strategy state with %d bars...", len(historical_df))
            self._warmup_strategy(historical_df)

        # 2. Reconcile broker state before starting feed
        logger.info("Reconciling broker state...")
        try:
            self._reconciler.reconcile_position()
            self._reconciler.reconcile_cash()
            self._reconciler.reconcile_orders()

            # Log initial account state after reconciliation
            logger.info("=" * 60)
            logger.info("INITIAL ACCOUNT STATE")
            logger.info("=" * 60)
            logger.info("Cash:     %s VND", f"{self._tracker.cash:,.2f}")
            logger.info("Equity:   %s VND", f"{self._tracker.equity:,.2f}")

            position = self._tracker.position
            if position.quantity != 0:
                logger.info(
                    "Position: %s %d @ %.2f (Unrealized P&L: %s VND)",
                    position.side.value,
                    position.quantity,
                    position.entry_price,
                    f"{position.unrealized_pnl:,.2f}",
                )
            else:
                logger.info("Position: FLAT")
            logger.info("=" * 60)

        except Exception:
            logger.exception("Broker reconciliation failed")
            # Continue anyway - reconciler logs errors internally

        # 3. Subscribe to feed
        logger.info("Starting feed subscription for %s...", self._symbol)
        try:
            await asyncio.wait_for(self._feed.subscribe(self._symbol, self._on_bar), timeout=10.0)
        except TimeoutError:
            logger.error("Feed subscription timed out after 10 seconds")
            logger.error("This usually means Redis connection is not available or not responding")
            logger.error(
                "Check MARKET_REDIS_HOST, MARKET_REDIS_PORT, and MARKET_REDIS_PASSWORD in .env"
            )
            raise
        except Exception as e:
            logger.error("Feed subscription failed: %s", e)
            raise

        logger.info("Engine started successfully")

    async def _run_sim(self, sim_df: pd.DataFrame) -> None:
        """Run sim mode by replaying historical bars.

        Args:
            sim_df: Historical OHLCV DataFrame for replay.
        """
        logger.info("[SIM] Replaying %d bars for %s...", len(sim_df), self._symbol)

        await self._feed.subscribe(self._symbol, self._on_bar)
        await self._feed.wait_for_completion()

        logger.info("[SIM] Replay completed")

    async def stop(self) -> None:
        """Stop the engine and clean up resources.

        Workflow:
        1. Cancel background tasks
        2. Unsubscribe from feed
        3. Close open position if configured (with correct price handling)
        4. Print and save session statistics
        5. Optionally force hard exit

        Fix: If _last_close == 0.0, pass None as exit price (not 0.0).
        """
        if not self._running:
            return

        self._running = False

        logger.info("Stopping engine...")

        # Cancel background tasks (prevents task leaks)
        for task in list(self._bg_tasks):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Unsubscribe from feed
        try:
            await self._feed.unsubscribe(self._symbol)
        except Exception:
            logger.exception("Feed unsubscribe failed")

        # Close feed
        try:
            await self._feed.close()
        except Exception:
            logger.exception("Feed close failed")

        # Close open position if configured
        if not self._tracker.is_flat and self._close_on_shutdown:
            # Fix: If _last_close == 0.0, pass None as exit price (not 0.0)
            exit_price = self._last_close if self._last_close > 0 else None

            if exit_price is None:
                logger.warning(
                    "Closing position on shutdown with no valid price "
                    "(last_close=%.2f) - submitting MARKET order",
                    self._last_close,
                )
            else:
                logger.info(
                    "Closing position on shutdown at last_close=%.2f",
                    exit_price,
                )

            self._order_manager.submit_exit(
                reason="Shutdown Close",
                price=exit_price,
                ord_type="MARKET" if exit_price is None else "LIMIT",
                timestamp=None,
            )

        # Print and save session statistics
        logger.info("=" * 60)
        logger.info("FINAL ACCOUNT STATE")
        logger.info("=" * 60)
        logger.info("Cash:     %s VND", f"{self._tracker.cash:,.2f}")
        logger.info("Equity:   %s VND", f"{self._tracker.equity:,.2f}")

        position = self._tracker.position
        if position.quantity != 0:
            logger.info(
                "Position: %s %d @ %.2f (Unrealized P&L: %s VND)",
                position.side.value,
                position.quantity,
                position.entry_price,
                f"{position.unrealized_pnl:,.2f}",
            )
        else:
            logger.info("Position: FLAT")
        logger.info("=" * 60)

        self._stats.print_summary()

        try:
            output_path = self._stats.save(self._output_dir)
            logger.info("Session results saved to %s", output_path)
        except Exception:
            logger.exception("Failed to save session statistics")

        logger.info("Engine stopped")

        # Force hard exit if configured
        if self._force_hard_exit:
            logger.info("Force hard exit enabled - calling os._exit(0)")
            os._exit(0)

    def _warmup_strategy(self, df: pd.DataFrame) -> None:
        """Warm up strategy internal state with historical bars.

        Feeds historical bars into strategy with is_warmup=True to initialize
        any internal state (e.g., ORB session range, indicator buffers).

        Also initializes daily P&L tracking so baseline is ready for today.

        Args:
            df: Historical OHLCV DataFrame with indicator columns.
        """
        warmup_errors = 0
        for raw_row in df.to_dict(orient="records"):
            bar: dict[str, Any] = {str(k): v for k, v in raw_row.items()}
            try:
                # Initialize P&L tracking during warmup
                dt = bar.get("datetime") or bar.get("timestamp") or datetime.now()
                if not isinstance(dt, datetime):
                    dt = datetime.now()

                self._tracker.update_daily_pnl(dt)
                self._tracker.update_unrealized(float(bar.get("close", 0.0) or 0.0))

                # Warmup strategy state
                self._strategy.generate_signal(
                    bar=bar,
                    position=self._tracker.position_snapshot,
                    is_warmup=True,
                )
            except Exception as exc:
                # Log warmup errors but continue - don't let one bad bar stop warmup
                warmup_errors += 1
                if warmup_errors <= 3:  # Only log first 3 errors to avoid spam
                    logger.warning(
                        "Warmup error on bar %s: %s",
                        bar.get("datetime", "unknown"),
                        exc,
                        exc_info=False,  # Don't log full traceback for warmup errors
                    )

        if warmup_errors > 0:
            logger.warning(
                "Strategy warmup completed with %d errors (out of %d bars)",
                warmup_errors,
                len(df),
            )

    # --- Bar Handler ---

    def _on_bar(self, bar: dict) -> None:
        """Handle new bar through handler pipeline.

        Pipeline:
        1. BarHandler: Update equity, handle deferred exits
        2. RiskHandler: Check SL/TP, trailing stop, force-flat
        3. SignalHandler: Generate strategy signals, submit orders

        If BarHandler or RiskHandler returns True (exit submitted), skip downstream.

        Args:
            bar: Bar dict with keys: datetime, open, high, low, close, volume, indicators.
        """
        try:
            # Extract bar time and close price
            bar_time_raw = bar.get("datetime")
            if not isinstance(bar_time_raw, datetime):
                logger.error("Bar missing valid datetime: %s", bar)
                return

            bar_time: datetime = bar_time_raw
            close = float(bar.get("close", 0.0))

            # Update last close for shutdown handling
            self._last_close = close

            # Increment bar counter
            self._bars_processed += 1

            # Extract key indicators for logging
            high = bar.get("high", 0.0)
            low = bar.get("low", 0.0)
            volume = bar.get("volume", 0.0)

            # Find ATR (could be atr_14, atr_20, etc.)
            atr = next((float(v) for k, v in bar.items() if str(k).startswith("atr_") and v), None)

            # Find ADX (could be adx_14, adx_20, etc.)
            adx = next((float(v) for k, v in bar.items() if str(k).startswith("adx_") and v), None)

            # Build log message with available data
            log_parts = [
                f"Bar {self._bars_processed}: {bar_time.strftime('%Y-%m-%d %H:%M')}",
                f"H={high:.2f} L={low:.2f} C={close:.2f}",
                f"vol={volume:.0f}",
            ]

            if atr is not None:
                log_parts.append(f"atr={atr:.2f}")
            if adx is not None:
                log_parts.append(f"adx={adx:.1f}")

            log_parts.append(f"pos={self._tracker.position.side.value}")
            log_parts.append(f"equity={self._tracker.equity:.0f}")

            logger.info(" | ".join(log_parts))

            # 1. BarHandler: Update equity, handle deferred exits
            if (
                self._bar_handler is None
                or self._risk_handler is None
                or self._signal_handler is None
            ):
                logger.error("_on_bar called before handlers are wired - bar dropped")
                return

            exit_submitted, self._deferred_exit_reason = self._bar_handler.on_bar(
                bar, bar_time, self._deferred_exit_reason
            )

            if exit_submitted:
                logger.debug("BarHandler submitted deferred exit, skipping downstream handlers")
                return

            # 2. RiskHandler: Check SL/TP, trailing stop, force-flat
            risk_exit_triggered = self._risk_handler.on_bar(bar, bar_time)

            if risk_exit_triggered:
                logger.debug("RiskHandler triggered exit, skipping SignalHandler")
                return

            # 3. SignalHandler: Generate strategy signals, submit orders
            self._signal_handler.on_bar(bar, bar_time)

        except Exception:
            logger.exception("Error in on_bar pipeline for bar %s", bar.get("datetime"))

    # --- Properties ---

    @property
    def stats(self) -> SessionStats:
        """Get session statistics calculator."""
        return self._stats

    @property
    def tracker(self) -> Tracker:
        """Get account tracker."""
        return self._tracker

    @property
    def running(self) -> bool:
        """Check if engine is running."""
        return self._running
