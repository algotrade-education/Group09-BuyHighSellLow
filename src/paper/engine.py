"""
PaperTrader — main async engine for live ORB paper trading.

Two modes:
  live  — subscribes to Redis market data (RedisMarketDataClient)
           and submits real FIX orders via PaperBrokerClient.
  sim   — replays a historical OHLC DataFrame through the BarProvider
           (useful when Redis is unavailable or for strategy testing).

Usage (live):
    trader = PaperTrader(client, redis_client, strategy, symbol, config)
    asyncio.run(trader.start())

Usage (sim):
    trader = PaperTrader(client=None, redis_client=None, strategy=strategy,
                         symbol="HNXDS:VN30F2601", config=cfg, dry_run=True)
    asyncio.run(trader.start(sim_df=historical_df))
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

from config.config import (
    COMMISSION_RATE,
    CONTRACT_MULTIPLIER,
    DEFAULT_INITIAL_CAPITAL,
    MARGIN_RATE,
)
from src.paper.bar_provider import BarProvider
from src.paper.order_manager import OrderManager
from src.paper.position_tracker import PositionTracker
from src.paper.stats import SessionStats
from src.strategy.base import Signal, Strategy

if TYPE_CHECKING:
    import pandas as pd
    from paperbroker.client import PaperBrokerClient
    from paperbroker.market_data import RedisMarketDataClient

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Orchestrates live or simulated paper trading.

    Wires together: BarProvider → Strategy → OrderManager → PositionTracker → SessionStats.
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
        self._qty: int = risk.get("min_position_size", 1)

        # Sub-components
        self._tracker = PositionTracker(
            initial_capital=initial_capital,
            commission_rate=COMMISSION_RATE,
            contract_multiplier=CONTRACT_MULTIPLIER,
        )

        self._order_mgr = OrderManager(
            client=client,  # type: ignore — may be None in sim mode
            tracker=self._tracker,
            symbol=symbol,
            dry_run=dry_run,
        )

        atr_period = config.get("strategy", {}).get("atr_period", 14)
        self._bar_provider = BarProvider(
            bar_freq=bar_freq,
            atr_period=atr_period,
            on_bar=self._on_new_bar,
        )

        self._stats = SessionStats(self._tracker)
        self._running = False
        self._bars_processed = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, sim_df: Optional["pd.DataFrame"] = None) -> None:
        """
        Start the paper trading engine.

        Args:
            sim_df: If provided, run in sim mode replaying this DataFrame.
                    If None, run in live mode (requires client + redis_client).
        """
        self._running = True

        if sim_df is not None:
            await self._run_sim(sim_df)
        else:
            await self._run_live()

    async def _run_live(self) -> None:
        """Live mode: connect FIX, subscribe Redis, process continuously."""
        if self._client is None or self._redis_client is None:
            logger.error(
                "Live mode requires both client and redis_client. "
                "Use --sim flag if Redis is not available."
            )
            return

        # FIX connection
        if not self.dry_run:
            logger.info("Connecting to PaperBroker (FIX)…")
            self._client.connect()
            if not self._client.wait_until_logged_on(timeout=60):
                err = self._client.last_logon_error()
                logger.error("FIX logon failed: %s", err)
                return
            logger.info("✅ FIX session established.")
            # Wire execution report handler
            self._client.on("fix:execution_report", self._order_mgr.on_execution_report)
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

        logger.info("📡 Listening for %s quotes (press Ctrl+C to stop)…", self.symbol)
        try:
            while self._running:
                # Emit equity snapshot every bar (BarProvider callback handles this)
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _run_sim(self, sim_df: "pd.DataFrame") -> None:
        """Sim mode: replay historical bars through the BarProvider."""
        logger.info(
            "[SIM] Replaying %d bars for %s at %s freq…",
            len(sim_df),
            self.symbol,
            self.bar_freq,
        )
        await self._bar_provider.replay(sim_df, speed=0.0)
        await self.stop()

    async def stop(self) -> None:
        """Gracefully stop the engine and print final stats."""
        self._running = False

        # Close any open position at last price
        if not self._tracker.is_flat:
            logger.info("Closing open position on shutdown…")
            self._order_mgr.submit_exit(reason="Session End")

        if self._redis_client is not None:
            try:
                await self._redis_client.close()
            except Exception:
                pass

        if self._client is not None and not self.dry_run:
            try:
                os._exit(0)  # Avoid QuickFIX cleanup segfault (see connect.py)
            except Exception:
                pass

        self._stats.print_summary()

    # ------------------------------------------------------------------
    # Bar callback (called by BarProvider for each completed bar)
    # ------------------------------------------------------------------

    def _on_new_bar(self, bar: Dict[str, Any]) -> None:
        """Process a completed OHLC + ATR bar through the strategy."""
        dt: datetime = bar.get("datetime", datetime.now())
        close = float(bar.get("close", 0))

        # Update unrealized P&L and take equity snapshot
        self._tracker.update_unrealized(close)
        self._tracker.equity_snapshot(dt)

        self._bars_processed += 1
        logger.debug(
            "Bar %d: %s close=%.2f | position=%s | equity=%.0f",
            self._bars_processed,
            dt.strftime("%Y-%m-%d %H:%M"),
            close,
            self._tracker.position.side.value,
            self._tracker.equity,
        )

        # --- SL/TP check BEFORE generating new signal ---
        if not self._tracker.is_flat:
            exit_trigger = self._tracker.check_sl_tp(bar)
            if exit_trigger:
                self._order_mgr.submit_exit(
                    reason=exit_trigger.replace("_", " ").title()
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
            self._submit_entry(signal, bar)
        elif signal.signal == Signal.CLOSE and not self._tracker.is_flat:
            self._order_mgr.submit_exit(reason="Strategy Close")

    def _submit_entry(self, signal, bar: Dict[str, Any]) -> None:
        """Calculate quantity and submit entry."""
        qty = self._qty
        # Simple percent-risk sizing if configured
        risk = self.config.get("risk", {})
        risk_pct = risk.get("risk_per_trade_pct", 0.0)
        if risk_pct > 0 and signal.stop_loss and signal.stop_loss > 0:
            entry = float(bar.get("close", 0))
            sl_dist = abs(entry - signal.stop_loss) * CONTRACT_MULTIPLIER
            if sl_dist > 0:
                risk_amount = self._tracker.equity * (risk_pct / 100)
                qty = max(1, int(risk_amount / sl_dist))
            max_qty = risk.get("max_position_size", 10)
            min_qty = risk.get("min_position_size", 1)
            qty = min(max_qty, max(min_qty, qty))

        self._order_mgr.submit_entry(signal, qty=qty, bar=bar)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> SessionStats:
        return self._stats

    def print_stats(self) -> None:
        self._stats.print_summary()
