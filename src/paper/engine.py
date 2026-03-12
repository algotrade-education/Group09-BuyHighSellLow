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
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from config.config import (
    COMMISSION_RATE,
    CONTRACT_MULTIPLIER,
    DEFAULT_INITIAL_CAPITAL,
)
from config.runtime_config import get_paper_bar_runtime_config, get_paper_runtime_config
from src.engine.position_sizer import FixedSizer, PercentRiskSizer, PositionSizer
from src.engine.session_gate import (
    vn30_is_entry_blocked,
    vn30_is_trading_time,
    vn30_seconds_to_window_end,
)
from src.paper.bar_fallback import load_fallback_bar_for_bucket
from src.paper.bar_provider import BarProvider
from src.paper.broker_sync import sync_broker_state
from src.paper.order_manager import OrderManager
from src.paper.position_tracker import PositionTracker
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

        # Sub-components
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
        runtime_config = get_paper_runtime_config(risk)
        self._enable_db_bar_fallback = runtime_config["enable_db_bar_fallback"]
        self._force_hard_exit = runtime_config["force_hard_exit"]
        self._entry_cutoff_seconds = runtime_config["entry_cutoff_seconds"]
        self._allow_late_entry = runtime_config["allow_late_entry"]
        self._force_flat_on_session_close = runtime_config[
            "force_flat_on_session_close"
        ]
        self._defer_exit_outside_session = runtime_config["defer_exit_outside_session"]
        self._deferred_exit_reason: Optional[str] = None

    _FREQ_TO_MINUTES: Dict[str, int] = {
        "1min": 1,
        "5min": 5,
        "15min": 15,
        "30min": 30,
        "1h": 60,
    }

    def _submit_exit_or_defer(
        self,
        reason: str,
        price: float,
        process_time: datetime,
    ) -> None:
        """Submit exit immediately during session, otherwise defer to next tradable bar."""
        if vn30_is_trading_time(process_time):
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _fallback_bar_for_bucket(self, bucket_dt: datetime) -> Optional[Dict[str, Any]]:
        """Load a closed bar from DB when live Redis delivered no trade ticks."""
        return load_fallback_bar_for_bucket(
            symbol=self.symbol,
            bucket_dt=bucket_dt,
            freq_minutes=self._bar_provider.freq_minutes,
            enabled=self._enable_db_bar_fallback,
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

        # Warm up strategy internal state (e.g. ORB session range)
        if historical_df is not None and not historical_df.empty:
            logger.info("Warming up strategy internal state...")
            for row in historical_df.to_dict(orient="records"):
                try:
                    self.strategy.generate_signal(
                        bar=row, current_position=self._tracker.position, is_warmup=True
                    )
                except Exception:
                    pass

        # Synchronize current broker state (Open Positions / Pending Orders)
        if not self.dry_run:
            self._sync_broker_state()

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
            await self._redis_client.subscribe(self.symbol, self._bar_provider.on_quote)
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

    def _sync_broker_state(self) -> None:
        """
        Fetch current portfolio and orders from PaperBroker to initialize
        the PositionTracker and OrderManager on startup.
        """
        if self._client is None or self.dry_run:
            return

        sync_broker_state(
            client=self._client,
            symbol=self.symbol,
            tracker=self._tracker,
            order_manager=self._order_mgr,
            logger=logger,
        )

    async def stop(self) -> None:
        """Stop the engine, attempt cleanup, and print session statistics."""
        self._running = False

        # Close any open position at last price
        if not self._tracker.is_flat:
            logger.info(
                "Closing open position on shutdown (last_close=%.2f)…", self._last_close
            )
            self._order_mgr.submit_exit(
                reason="Session End", price=self._last_close or None
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
    # Bar callback (called by BarProvider for each completed bar)
    # ------------------------------------------------------------------

    def _on_new_bar(self, bar: Dict[str, Any]) -> None:
        """Handle one completed bar through risk checks, strategy, and orders."""
        dt: datetime = bar.get("datetime", datetime.now())
        process_time = datetime.now()
        close = float(bar.get("close", 0))
        self._last_close = close

        trading_now = vn30_is_trading_time(process_time)

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
            exit_trigger = self._tracker.check_sl_tp(bar)
            if exit_trigger:
                self._submit_exit_or_defer(
                    reason=exit_trigger.replace("_", " ").title(),
                    price=close,
                    process_time=process_time,
                )
                return  # Don't generate new entry on same bar

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
            if vn30_is_entry_blocked(
                dt=process_time,
                entry_cutoff_seconds=self._entry_cutoff_seconds,
                allow_late_entry=self._allow_late_entry,
            ):
                seconds_to_end = vn30_seconds_to_window_end(process_time)
                if seconds_to_end is None:
                    reason = "outside trading session"
                else:
                    reason = (
                        f"within entry cutoff ({seconds_to_end:.1f}s <= "
                        f"{self._entry_cutoff_seconds:.1f}s)"
                    )

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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> SessionStats:
        return self._stats

    def print_stats(self) -> None:
        self._stats.print_summary()
