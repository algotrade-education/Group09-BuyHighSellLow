"""Redis-based live market data feed for paper trading.

Wraps a Redis market data client to provide real-time tick data, filtering
invalid prices and out-of-session timestamps before forwarding to BarAggregator.

Key features:
- Tick filtering: drops price <= 0 and out-of-session timestamps
- Session time validation: delegated to SessionManager (no hardcoded times)
- Watchdog monitoring: logs warning if no quotes received for >300s during session
- Clock-based bar rollover: polling loop ensures bars emit even with sparse ticks
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from datetime import datetime
from time import monotonic
from typing import TYPE_CHECKING, Any

from src.paper.bar_aggregator import BarAggregator
from src.paper.feeds.base import FeedBase

if TYPE_CHECKING:
    from src.engine.session import SessionManager

logger = logging.getLogger(__name__)


class RedisFeed(FeedBase):
    """Live market data feed from Redis.

    Subscribes to Redis market data stream, filters ticks, and forwards
    valid ticks to BarAggregator for OHLC bar construction.

    Implements watchdog monitoring to detect feed silence during trading hours.
    """

    def __init__(
        self,
        redis_client: Any,
        bar_aggregator: BarAggregator,
        session_manager: SessionManager,
        watchdog_silence_seconds: float = 300.0,
    ) -> None:
        """Initialize RedisFeed.

        Args:
            redis_client: Redis market data client with subscribe/unsubscribe methods.
            bar_aggregator: BarAggregator instance to forward valid ticks to.
            session_manager: SessionManager for trading hours validation.
            watchdog_silence_seconds: Seconds of silence before logging warning.
        """
        self._redis_client = redis_client
        self._bar_aggregator = bar_aggregator
        self._session_manager = session_manager
        self._watchdog_silence_seconds = watchdog_silence_seconds

        self._subscribed_symbol: str | None = None
        self._last_quote_ts: datetime | None = None
        self._polling_task: asyncio.Task | None = None
        self._running = False

        # Diagnostics counters
        self._quote_callbacks: int = 0
        self._quote_dropped_no_price: int = 0
        self._quote_forwarded: int = 0
        self._quote_dropped_out_of_session: int = 0
        self._last_diag_log: float = 0.0

    async def subscribe(self, symbol: str, callback: Callable[[dict], None]) -> None:
        """Subscribe to market data for a symbol.

        Registers the on_bar callback with BarAggregator and starts the Redis
        subscription and polling loop.

        Args:
            symbol: Symbol to subscribe to (e.g. "VN30F1M").
            callback: Callback to invoke when a new bar is ready.
        """
        if self._running:
            logger.warning("RedisFeed already running, ignoring duplicate subscribe")
            return

        self._subscribed_symbol = symbol
        self._bar_aggregator.set_on_bar(callback)
        self._running = True

        # Register quote callback with Redis client and start subscription
        await self._redis_client.subscribe(symbol, self._on_quote)

        # Start polling loop for check_time() and watchdog
        self._polling_task = asyncio.create_task(self._polling_loop())

        logger.info("RedisFeed subscribed to %s", symbol)

    async def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from market data for a symbol.

        Args:
            symbol: Symbol to unsubscribe from.
        """
        if not self._running:
            return

        self._running = False

        # Cancel polling task
        if self._polling_task is not None:
            self._polling_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._polling_task
            self._polling_task = None

        # Unsubscribe from Redis
        if self._redis_client and symbol:
            await self._redis_client.unsubscribe(symbol)

        self._subscribed_symbol = None
        logger.info("RedisFeed unsubscribed from %s", symbol)

    async def close(self) -> None:
        """Close the feed and clean up resources."""
        if self._subscribed_symbol:
            await self.unsubscribe(self._subscribed_symbol)

        if self._redis_client:
            await self._redis_client.close()

        logger.info("RedisFeed closed")

    async def _on_quote(self, instrument_or_snapshot: Any, quote: Any = None) -> None:
        """Process incoming Redis quote snapshot.

        Supports both callback shapes from RedisMarketDataClient:
        - (instrument, quote_snapshot)
        - (quote_snapshot,)

        Filters invalid prices and out-of-session timestamps before forwarding
        to BarAggregator.

        Args:
            instrument_or_snapshot: Symbol string or QuoteSnapshot (if single-arg form).
            quote: QuoteSnapshot instance (if two-arg form).
        """
        # Normalise callback shape
        if quote is None:
            snapshot = instrument_or_snapshot
            instrument = getattr(snapshot, "instrument", self._subscribed_symbol or "")
        else:
            instrument = str(instrument_or_snapshot or self._subscribed_symbol or "")
            snapshot = quote

        self._quote_callbacks += 1

        # Filter 1: Drop invalid prices (bid/ask-only updates have no matched price)
        price = getattr(snapshot, "latest_matched_price", None)
        if price is None or price <= 0:
            self._quote_dropped_no_price += 1
            self._log_quote_diagnostics(instrument)
            return

        # Update watchdog timestamp
        self._last_quote_ts = datetime.now()

        # Filter 2: Drop out-of-session timestamps
        if not self._session_manager.is_trading_hours(self._last_quote_ts):
            self._quote_dropped_out_of_session += 1
            return

        # Forward valid tick to BarAggregator
        self._quote_forwarded += 1
        volume = float(getattr(snapshot, "latest_matched_quantity", 0.0) or 0.0)
        self._bar_aggregator.on_tick(self._last_quote_ts, price, volume)
        self._log_quote_diagnostics(instrument)

    async def _polling_loop(self) -> None:
        """Polling loop for check_time() and watchdog monitoring.

        Runs every second to:
        1. Call BarAggregator.check_time() for clock-based bar rollover
        2. Check for feed silence and log warning if >300s during session
        """
        try:
            while self._running:
                await asyncio.sleep(1)

                # Clock-based bar rollover check
                self._bar_aggregator.check_time()

                # Watchdog: check for feed silence during session
                self._check_watchdog()

        except asyncio.CancelledError:
            logger.debug("Polling loop cancelled")
            raise

    def _log_quote_diagnostics(self, instrument: str) -> None:
        """Log quote callback quality metrics every 15 seconds or every 100 callbacks."""
        now = monotonic()
        if (now - self._last_diag_log) <= 15 and self._quote_callbacks % 100 != 0:
            return
        self._last_diag_log = now
        logger.info(
            "Quote diag | %s | callbacks=%d forwarded=%d dropped_no_price=%d dropped_out_of_session=%d",
            instrument,
            self._quote_callbacks,
            self._quote_forwarded,
            self._quote_dropped_no_price,
            self._quote_dropped_out_of_session,
        )

    def _check_watchdog(self) -> None:
        """Check for feed silence and log warning if exceeded threshold during session."""
        now = datetime.now()

        # Only check during trading hours
        if not self._session_manager.is_trading_hours(now):
            return

        # No quotes ever received since startup
        if self._last_quote_ts is None:
            if not hasattr(self, "_subscribe_ts"):
                self._subscribe_ts: datetime = now
            silence_seconds = (now - self._subscribe_ts).total_seconds()
            if silence_seconds > self._watchdog_silence_seconds:
                logger.warning(
                    "Feed silence detected: no quotes received since startup (%.0f seconds) - check Redis connection and quote attribute names",
                    silence_seconds,
                )
            return

        # Calculate silence duration
        silence_seconds = (now - self._last_quote_ts).total_seconds()

        # Log warning if silence exceeds threshold
        if silence_seconds > self._watchdog_silence_seconds:
            logger.warning(
                "Feed silence detected: no quotes for %.0f seconds (threshold: %.0f) - check Redis connection and quote attribute names",
                silence_seconds,
                self._watchdog_silence_seconds,
            )
