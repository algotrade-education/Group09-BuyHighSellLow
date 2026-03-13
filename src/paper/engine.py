"""
PaperTrader Engine - Live Execution and Simulation Orchestrator.

The `PaperTrader` is the central coordinator for real-time trading. It
integrates data ingestion, strategy signal generation, and order execution
into a unified asynchronous lifecycle.

Key Responsibilities:
1.  **Lifecycle Management**: Handles startup (connection, sync) and
    shutdown (cleanup, statistics).
2.  **Event Orchestration**: Wires the `BarProvider` (data) to the
    `Strategy` (logic) and the `OrderManager` (execution).
3.  **State Synchronization**: Recovers open positions and pending orders
    from the broker on restart to prevent "state-blind" trading.
4.  **Simulation Replay**: Provides a high-fidelity 'sim' mode that replays
    historical bars through the live indicator and risk pipeline.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, Optional

from config.config import (
    COMMISSION_RATE,
    CONTRACT_MULTIPLIER,
    DEFAULT_INITIAL_CAPITAL,
)
from config.runtime_config import get_paper_bar_runtime_config, get_paper_runtime_config
from src.engine.position_sizer import FixedSizer, PercentRiskSizer, PositionSizer
from src.engine.session_manager import SessionManager, VN30Session
from src.paper.bar_fallback import load_fallback_bar_for_bucket
from src.paper.bar_provider import BarProvider
from src.paper.broker_sync import sync_broker_state
from src.paper.order_manager import OrderManager
from src.paper.position_tracker import PositionTracker
from src.paper.risk_manager import RiskManager
from src.paper.stats import SessionStats
from src.strategy.base import Signal, Strategy, TradeSignal

if TYPE_CHECKING:
    import pandas as pd
    from paperbroker.client import PaperBrokerClient
    from paperbroker.market_data import RedisMarketDataClient

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Central Controller for Live Strategy Execution.

    This class implements the 'trading brain'. It listens for completed bars,
    evaluates them against the strategy, checks risk limits (SL/TP), and
    dispatches orders to the broker.

    Pipeline:
    BarProvider (Data) -> PaperTrader (Check SL/TP -> Strategy Signal) -> OrderManager (Execution)
    """

    _FREQ_TO_MINUTES: Dict[str, int] = {
        "1min": 1,
        "5min": 5,
        "15min": 15,
        "30min": 30,
        "1h": 60,
    }

    def __init__(
        self,
        strategy: Strategy,
        symbol: str,
        config: Dict[str, Any],
        client: Optional["PaperBrokerClient"] = None,
        redis_client: Optional["RedisMarketDataClient"] = None,
        bar_freq: str = "5min",
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        dry_run: bool = False,
    ):
        """
        Args:
            strategy:        Initialised ORB (or any) Strategy instance.
            symbol:          Market symbol to trade (e.g. 'HNXDS:VN30F2601').
            config:          Full strategy config dict (risk params, etc.).
            client:          PaperBrokerClient (required in live mode).
            redis_client:    RedisMarketDataClient (required in live mode).
            bar_freq:        OHLC bar frequency: '1min', '5min', '15min', '30min'.
            initial_capital: Starting capital in VND.
            dry_run:         If True, log orders but don't send via FIX.
        """
        self.strategy = strategy
        self.symbol = symbol
        self.config = config
        self.bar_freq = bar_freq
        self.dry_run = dry_run

        self._client = client
        self._redis_client = redis_client

        # Determine position sizing from config
        risk = config.get("risk", {})
        self._position_sizer = self._build_position_sizer(risk)

        # Managers
        self._tracker = PositionTracker(
            initial_capital=initial_capital,
            commission_rate=COMMISSION_RATE,
            contract_multiplier=CONTRACT_MULTIPLIER,
        )

        self._order_mgr = OrderManager(
            client=client,  # type: ignore - may be None in sim mode
            tracker=self._tracker,
            symbol=symbol,
            dry_run=dry_run,
        )

        runtime_config = get_paper_runtime_config(risk)

        self._session_mgr: SessionManager = VN30Session()
        self._risk_mgr = RiskManager(
            use_trailing_stop=bool(risk.get("use_trailing_stop", False)),
            trailing_atr_multiplier=float(
                risk.get("trailing_atr_multiplier", 2.0) or 2.0
            ),
            max_daily_loss_pct=float(risk.get("max_daily_loss", 0.0) or 0.0),
            initial_capital=initial_capital,
        )

        atr_period = config.get("strategy", {}).get("atr_period", 14)
        bar_runtime_config = get_paper_bar_runtime_config(
            freq_minutes=self._FREQ_TO_MINUTES.get(bar_freq, 5),
            risk=risk,
        )
        self._bar_provider = BarProvider(
            bar_freq=bar_freq,
            atr_period=atr_period,
            on_bar=self._on_new_bar,
            fallback_bar_provider=self._fallback_bar_for_bucket,
            runtime_config=bar_runtime_config,
        )

        self._stats = SessionStats(self._tracker)
        self._running = False
        self._bars_processed = 0
        self._last_close: float = 0.0

        # Engine execution config
        self._enable_db_bar_fallback = runtime_config["enable_db_bar_fallback"]
        self._close_on_shutdown = runtime_config["close_on_shutdown"]
        self._force_hard_exit = runtime_config["force_hard_exit"]
        self._defer_exit_outside_session = runtime_config["defer_exit_outside_session"]

        # Session forcing
        self._entry_cutoff_seconds = runtime_config["entry_cutoff_seconds"]
        self._allow_late_entry = runtime_config["allow_late_entry"]
        self._force_flat_on_session_close = runtime_config[
            "force_flat_on_session_close"
        ]
        self._force_flat_preclose_seconds = runtime_config[
            "force_flat_preclose_seconds"
        ]
        self._force_flat_on_last_candle = runtime_config["force_flat_on_last_candle"]

        if (
            self._force_flat_on_session_close
            and self._force_flat_preclose_seconds <= 0
            and not self._force_flat_on_last_candle
        ):
            self._force_flat_preclose_seconds = 15.0
            logger.warning(
                "force_flat_on_session_close is enabled with no proactive trigger; "
                "using fallback preclose window: %.1fs",
                self._force_flat_preclose_seconds,
            )

        self._deferred_exit_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _sync_broker_state(self) -> None:
        """
        Fetch current portfolio and orders from PaperBroker to initialize
        the PositionTracker and OrderManager on startup.
        """
        if self._client is None:
            return

        sync_broker_state(
            client=self._client,
            symbol=self.symbol,
            tracker=self._tracker,
            order_manager=self._order_mgr,
            logger=logger,
        )

    async def start(
        self,
        historical_df: Optional["pd.DataFrame"] = None,
        incomplete_bar: Optional[Dict[str, Any]] = None,
        sim_df: Optional["pd.DataFrame"] = None,
    ) -> None:
        """
        Start the trading activity based on the current mode (Live or Sim).

        Args:
            historical_df: Recent daily/intraday bars for indicator warmup.
            incomplete_bar: Partially formed bar to seed current live state.
            sim_df:        Historical dataset for pure sim/backtesting replay.
        """
        self._running = True

        if sim_df is not None:
            await self._run_sim(sim_df)
        else:
            await self._run_live(historical_df, incomplete_bar)

    async def _run_live(
        self,
        historical_df: Optional["pd.DataFrame"] = None,
        incomplete_bar: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Execute the live trading lifecycle.

        Phase 1: Indicator & Strategy Warmup (injecting `historical_df`).
        Phase 2: Broker Sync (recovery of positions/orders via REST).
        Phase 3: Connection (FIX login).
        Phase 4: Market Data Feed (Redis subscription).
        Phase 5: Main Loop (system-time monitoring for bar rollovers).
        """
        if self._client is None or self._redis_client is None:
            logger.error(
                "Live mode requires both client and redis_client. "
                "Use --sim flag if Redis is not available."
            )
            return

        # Inject historical data for indicator warmup BEFORE live ticks arrive
        if historical_df is not None and not historical_df.empty:
            logger.info(
                "Injecting %d historical bars for BarProvider warmup...",
                len(historical_df),
            )
            self._bar_provider.preload_history(historical_df)

        if incomplete_bar is not None:
            self._bar_provider.seed_current_live_bar(incomplete_bar)

        # Synchronize current broker state (Open Positions / Pending Orders)
        # We do this BEFORE warmup so Initializing daily P&L uses real data.
        # We also do it in dry-run mode if a client is available for context.
        if self._client is not None:
            self._sync_broker_state()

        # Warm up strategy internal state (e.g. ORB session range)
        if historical_df is not None and not historical_df.empty:
            logger.info("Warming up strategy internal state...")
            for raw_row in historical_df.to_dict(orient="records"):
                bar: Dict[str, Any] = {str(k): v for k, v in raw_row.items()}
                try:
                    # Initialize P&L tracking during warmup so the baseline is ready for today
                    dt = bar.get("datetime") or bar.get("timestamp") or datetime.now()
                    bar_time = self._resolve_bar_time(dt, datetime.now())
                    self._tracker.update_daily_pnl(bar_time)
                    self._tracker.update_unrealized(float(bar.get("close", 0.0) or 0.0))

                    self.strategy.generate_signal(
                        bar=bar,
                        current_position=self._tracker.position,
                        is_warmup=True,
                    )
                except Exception:
                    pass

        # FIX connection
        if not self.dry_run:
            logger.info("Connecting to PaperBroker (FIX)…")
            self._client.connect()
            if not self._client.wait_until_logged_on(timeout=60):
                err = self._client.last_logon_error()
                logger.error("FIX logon failed: %s", err)
                return
            logger.info("FIX session established.")

            # Wire execution report handler
            logger.info("Wiring execution report handler...")
            self._client.on("fix:execution_report", self._order_mgr.on_execution_report)
            logger.info("Execution report handler wired successfully.")
        else:
            logger.info("[DRY-RUN] Skipping FIX connection.")

        # Redis subscription
        logger.info("Subscribing to Redis market data for %s…", self.symbol)
        try:
            await self._redis_client.subscribe(self.symbol, self._redis_quote_callback)
        except Exception as exc:
            logger.error("Redis subscription failed: %s", exc)
            logger.error(
                "Hint: set MARKET_REDIS_HOST / MARKET_REDIS_PORT / MARKET_REDIS_PASSWORD "
                "in your .env, or use --sim to run without Redis."
            )
            return

        logger.info("Listening for %s quotes (press Ctrl+C to stop)…", self.symbol)
        try:
            while self._running:
                # Force clock-aligned bar rollovers (resolves illiquid market tick delays)
                self._maybe_force_flat_by_clock(datetime.now())
                self._bar_provider.check_time()
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Live loop cancelled.")
        except Exception as exc:
            logger.error("Unhandled exception in live loop: %s", exc, exc_info=True)
        finally:
            await self.stop()

    async def _run_sim(self, sim_df: "pd.DataFrame") -> None:
        """Run sim mode by replaying historical bars through `BarProvider`."""
        logger.info(
            "[SIM] Replaying %d bars for %s at %s freq…",
            len(sim_df),
            self.symbol,
            self.bar_freq,
        )
        await self._bar_provider.replay(sim_df, speed=0.0)
        await self.stop()

    async def stop(self) -> None:
        """Stop the engine, attempt cleanup, and print session statistics."""
        self._running = False

        # Optionally close any open position at last price
        if not self._tracker.is_flat:
            if self._close_on_shutdown:
                logger.info(
                    "Closing open position on shutdown (last_close=%.2f)…",
                    self._last_close,
                )
                self._order_mgr.submit_exit(
                    reason="Shutdown Close", price=self._last_close or None
                )
            else:
                logger.warning(
                    "Shutdown detected with open position; preserving position "
                    "(set PAPER_CLOSE_ON_SHUTDOWN=true to auto-close)."
                )

        if self._redis_client is not None:
            try:
                await self._redis_client.close()
            except Exception:
                pass

        self._stats.print_summary()

        if self._client is not None and not self.dry_run and self._force_hard_exit:
            try:
                os._exit(
                    0
                )  # Optional: avoid QuickFIX cleanup segfault (see connect.py)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _redis_quote_callback(
        self,
        instrument_or_snapshot: Any,
        quote: Optional[Any] = None,
    ) -> None:
        """
        Adapter for Redis market data callbacks.

        Supports both callback shapes observed across client/type stubs:
        - (instrument, quote)
        - (quote_snapshot)
        """
        if quote is None:
            snapshot = instrument_or_snapshot
            instrument = getattr(snapshot, "instrument", self.symbol)
        else:
            instrument = str(instrument_or_snapshot or self.symbol)
            snapshot = quote

        result = self._bar_provider.on_quote(instrument, snapshot)

        # on_quote may be async; if it returns a coroutine, schedule it
        if asyncio.iscoroutine(result):
            asyncio.create_task(result)

    def _on_new_bar(self, bar: Dict[str, Any]) -> None:
        """Handle one completed bar through risk checks, strategy, and orders."""
        dt: datetime = bar.get("datetime", datetime.now())
        process_time = datetime.now()
        bar_time = self._resolve_bar_time(dt, process_time)
        close = float(bar.get("close", 0))
        self._last_close = close

        self._tracker.update_unrealized(close)
        self._tracker.update_daily_pnl(bar_time)

        trading_now = self._session_mgr.is_trading_hours(process_time)

        if self._deferred_exit_reason and not self._tracker.is_flat and trading_now:
            reason = self._deferred_exit_reason
            logger.info(
                "Submitting deferred exit at session reopen: %s | process_time=%s",
                reason,
                process_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            self._order_mgr.submit_exit(reason=reason, price=close)
            self._deferred_exit_reason = None

        if (
            self._force_flat_on_session_close
            and not self._tracker.is_flat
            and not trading_now
            and self._deferred_exit_reason is None
        ):
            self._submit_exit_or_defer(
                reason="Session Boundary Close",
                price=close,
                process_time=process_time,
            )

        if not self._tracker.is_flat and self._deferred_exit_reason is None:
            reason = self._session_mgr.get_force_close_reason(
                dt=bar_time, preclose_seconds=self._force_flat_preclose_seconds
            )

            # Additional heuristic: checking if the *next* candle will fall outside
            # the active window to handle edge-of-session closures explicitly.
            bar_is_last_in_window = False
            if self._force_flat_on_last_candle:
                # E.g. 14:25 + 5m = 14:30. If active window ends at 14:30, it's the last candle.
                bar_close_time = bar_time + timedelta(
                    minutes=self._bar_provider.freq_minutes
                )
                # Ensure the close time isn't still inside session
                if not self._session_mgr.is_trading_hours(
                    bar_close_time - timedelta(seconds=1)
                ):
                    bar_is_last_in_window = True

            if reason is not None or bar_is_last_in_window:
                if bar_is_last_in_window:
                    reason = "Last Candle Close"

                logger.info(
                    "Force-flat trigger: %s | bar_time=%s",
                    reason,
                    bar_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                self._submit_exit_or_defer(
                    reason=reason or "Session Boundary Close",
                    price=close,
                    process_time=bar_time,
                )
                return

        # Extract indicators for logging
        atr_key = f"atr_{self.config.get('strategy', {}).get('atr_period', 14)}"
        atr = bar.get(atr_key, 0.0)
        volume = bar.get("volume", 0.0)
        volume_ma_20 = bar.get("volume_ma_20", 0.0)

        # Build extra info string from available bar items (e.g. adx, rsi from strategy if present)
        extras = []
        adx_key = f"adx_{self.config.get('strategy', {}).get('adx_period', 14)}"
        if adx_key in bar:
            extras.append(f"adx={bar[adx_key]:.1f}")
        rsi_key = f"rsi_{self.config.get('strategy', {}).get('rsi_period', 14)}"
        if rsi_key in bar:
            extras.append(f"rsi={bar[rsi_key]:.1f}")

        extra_str = f" | {', '.join(extras)}" if extras else ""

        # Update unrealized P&L and take equity snapshot
        self._tracker.update_unrealized(close)
        self._tracker.equity_snapshot(dt)

        self._bars_processed += 1
        logger.info(
            "Bar %d: %s close=%.2f vol=%.0f vol_ma20=%.1f atr=%.2f%s | position=%s | equity=%.0f",
            self._bars_processed,
            dt.strftime("%Y-%m-%d %H:%M"),
            close,
            volume,
            volume_ma_20,
            atr,
            extra_str,
            self._tracker.position.side.value,
            self._tracker.equity,
        )

        # --- SL/TP check BEFORE generating new signal ---
        if not self._tracker.is_flat:
            exit_trigger = self._risk_mgr.get_exit_trigger(self._tracker.position, bar)
            if exit_trigger:
                self._submit_exit_or_defer(
                    reason=exit_trigger,
                    price=close,
                    process_time=process_time,
                )
                return  # Don't generate new entry on same bar

            self._risk_mgr.apply_trailing_stop(self._tracker.position, bar)

        # Track daily loss using RiskManager
        if self._tracker.is_flat and self._risk_mgr.is_daily_loss_hit(
            self._tracker.daily_pnl
        ):
            logger.info(
                "Skipping signal generation: max daily loss reached (daily_pnl=%.2f).",
                self._tracker.daily_pnl,
            )
            return

        # --- Strategy signal ---
        try:
            signal = self.strategy.generate_signal(
                bar=bar,
                current_position=self._tracker.position,
            )
        except Exception as exc:
            logger.error("Strategy error on bar %s: %s", dt, exc, exc_info=True)
            return

        if signal.signal in (Signal.LONG, Signal.SHORT) and self._tracker.is_flat:
            if self._session_mgr.is_entry_blocked(
                dt=process_time,
                cutoff_seconds=self._entry_cutoff_seconds,
                allow_late=self._allow_late_entry,
            ):
                reason = f"within entry cutoff ({self._entry_cutoff_seconds:.1f}s)"
                logger.info(
                    "Skipping %s entry at process_time=%s: %s",
                    signal.signal.name,
                    process_time.strftime("%Y-%m-%d %H:%M:%S"),
                    reason,
                )
                return

            logger.info(
                "Strategy signal: %s | close=%.2f | reason=%s",
                signal.signal.name,
                close,
                signal.reason or "N/A",
            )
            self._submit_entry(signal, bar)
        elif signal.signal == Signal.CLOSE and not self._tracker.is_flat:
            logger.info(
                "Strategy signal: CLOSE | close=%.2f | reason=%s",
                close,
                signal.reason or "N/A",
            )
            self._submit_exit_or_defer(
                reason="Strategy Close",
                price=close,
                process_time=process_time,
            )
        elif signal.signal == Signal.HOLD:
            logger.info(
                "Strategy signal: HOLD  | position=%s | vol=%.0f vol_ma20=%.1f | reason=%s",
                self._tracker.position.side.value
                if not self._tracker.is_flat
                else "FLAT",
                volume,
                volume_ma_20,
                signal.reason or "No entry/exit criteria met",
            )

    def _maybe_force_flat_by_clock(self, process_time: datetime) -> None:
        """Run a per-second preclose flat check so exits don't depend on bar-close timing."""
        if self._tracker.is_flat or self._force_flat_preclose_seconds <= 0:
            return

        reason = self._session_mgr.get_force_close_reason(
            dt=process_time,
            preclose_seconds=self._force_flat_preclose_seconds,
        )
        if not reason:
            return

        if self._last_close <= 0:
            logger.warning(
                "Clock preclose trigger ready but skipped: last_close unavailable."
            )
            return

        logger.info(
            "Clock preclose force-flat trigger: %s | process_time=%s",
            reason,
            process_time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._submit_exit_or_defer(
            reason=reason,
            price=self._last_close,
            process_time=process_time,
        )

    # ------------------------------------------------------------------
    # Execution Logic
    # ------------------------------------------------------------------

    def _submit_entry(self, signal: TradeSignal, bar: Dict[str, Any]) -> None:
        """Compute position size from configured PositionSizer and submit entry order."""
        entry_price = float(bar.get("close", 0.0) or 0.0)
        qty = self._position_sizer.calculate_size(
            equity=self._tracker.equity,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            contract_multiplier=CONTRACT_MULTIPLIER,
        )
        if qty <= 0:
            logger.warning("Skipping entry: PositionSizer returned invalid qty=%d", qty)
            return

        self._order_mgr.submit_entry(signal, qty=qty, bar=bar)

    def _submit_exit_or_defer(
        self,
        reason: str,
        price: float,
        process_time: datetime,
    ) -> None:
        """Submit exit immediately during session, otherwise defer to next tradable bar."""
        if self._session_mgr.is_trading_hours(process_time):
            self._order_mgr.submit_exit(reason=reason, price=price)
            self._deferred_exit_reason = None
            return

        if self._defer_exit_outside_session:
            self._deferred_exit_reason = reason
            logger.warning(
                "Deferring exit (%s): process_time=%s is outside trading session.",
                reason,
                process_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return

        logger.warning(
            "Skipping exit (%s): process_time=%s is outside trading session.",
            reason,
            process_time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_position_sizer(self, risk: Dict[str, Any]) -> PositionSizer:
        """Build position sizer from risk config for paper trading entries."""
        risk_pct = float(risk.get("risk_per_trade_pct", 0.0) or 0.0)
        min_size = int(risk.get("min_position_size", 1) or 1)
        max_size = int(
            risk.get("max_position_size", max(min_size, 1)) or max(min_size, 1)
        )

        if risk_pct > 0:
            logger.info(
                "PaperTrader using PercentRiskSizer: %.2f%% risk per trade (min=%d max=%d)",
                risk_pct,
                min_size,
                max_size,
            )
            return PercentRiskSizer(
                risk_per_trade_pct=risk_pct,
                min_size=min_size,
                max_size=max_size,
            )

        logger.info("PaperTrader using FixedSizer: size=%d", min_size)
        return FixedSizer(size=min_size)

    def _resolve_bar_time(self, dt: Any, fallback: datetime) -> datetime:
        """Return bar timestamp as python datetime for session-boundary checks."""
        if isinstance(dt, datetime):
            return dt

        to_py = getattr(dt, "to_pydatetime", None)
        if callable(to_py):
            converted = to_py()
            if isinstance(converted, datetime):
                return converted

        return fallback

    def _fallback_bar_for_bucket(self, bucket_dt: datetime) -> Optional[Dict[str, Any]]:
        """Load a closed bar from DB when live Redis delivered no trade ticks."""
        return load_fallback_bar_for_bucket(
            symbol=self.symbol,
            bucket_dt=bucket_dt,
            freq_minutes=self._bar_provider.freq_minutes,
            enabled=self._enable_db_bar_fallback,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> SessionStats:
        return self._stats

    def print_stats(self) -> None:
        self._stats.print_summary()
