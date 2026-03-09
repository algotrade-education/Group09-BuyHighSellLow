"""
Real-time Bar Aggregator and Replay Engine.

The `BarProvider` is responsible for transforming raw price ticks (from a live 
Redis feed) or historical OHLC rows (from a simulation) into a consistent 
stream of enriched bars for strategy execution.

Core Functions:
1.  **Live Aggregation**: Subscribes to `RedisMarketDataClient` and buckets 
    QuoteSnapshots into OHLC bars based on a system-time clock.
2.  **Simulation Replay**: Processes historical DataFrames bar-by-bar, 
    recomputing indicators at each step to mirror live data behavior.
3.  **Indicator Warmup**: Uses the `Preprocessor` and a historical buffer to 
    ensure indicators (like ATR) are stable before emitting the first bar.
"""

import asyncio
import logging
import os
from collections import deque
from datetime import datetime, time
from time import monotonic
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.data.preprocessor import Preprocessor

logger = logging.getLogger(__name__)

# VN30 session boundaries (matching ORB strategy constants)
MORNING_START = time(9, 0)
MORNING_END = time(11, 30)
AFTERNOON_START = time(13, 0)
AFTERNOON_END = time(14, 30)

# Bar frequency → timedelta minutes mapping
_FREQ_MINUTES: Dict[str, int] = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
}


def _bar_bucket(dt: datetime, freq_minutes: int) -> datetime:
    """Floor a datetime to the nearest bar boundary."""
    total_minutes = dt.hour * 60 + dt.minute
    bucket_start = (total_minutes // freq_minutes) * freq_minutes
    return dt.replace(
        hour=bucket_start // 60, minute=bucket_start % 60, second=0, microsecond=0
    )


def _in_session(t: time) -> bool:
    """Check if a time is within VN30 trading session."""
    return (MORNING_START <= t < MORNING_END) or (AFTERNOON_START <= t < AFTERNOON_END)


class BarProvider:
    """
    Stateful aggregator that converts ticks into strategy-ready bars.

    Supports two operation modes:
    - **Live Mode**: Uses `on_quote` as an async callback for Redis updates. 
      It uses system-time to bucket ticks, ensuring bars are emitted 
      precisely at boundary crossings (e.g., exactly at 10:00:00).
    - **Sim Mode**: Uses `replay` to push historical rows through the same 
      indicator pipeline, allowing for high-fidelity backtesting of 
      real-time logic.

    Attributes:
        bar_freq: E.g., '1min' or '5min'.
        atr_period: Lookback for technical indicator stability.
        on_bar: The main engine callback (receives enriched dicts).
    """

    def __init__(
        self,
        bar_freq: str = "5min",
        atr_period: int = 14,
        on_bar: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Args:
            bar_freq:   Bar frequency string, e.g. '1min', '5min', '15min'.
            atr_period: ATR lookback period (warmup = atr_period + 1 bars).
            on_bar:     Callback invoked with each completed, enriched bar dict.
        """
        if bar_freq not in _FREQ_MINUTES:
            raise ValueError(
                f"Unsupported bar_freq '{bar_freq}'. Choose from: {list(_FREQ_MINUTES)}"
            )

        self.bar_freq = bar_freq
        self.freq_minutes = _FREQ_MINUTES[bar_freq]
        self.atr_period = atr_period
        self.on_bar = on_bar

        self._preprocessor = Preprocessor(atr_period=atr_period)

        # Current open bar state
        self._current_bucket: Optional[datetime] = None
        self._bar_open: float = 0.0
        self._bar_high: float = 0.0
        self._bar_low: float = float("inf")
        self._bar_close: float = 0.0
        self._bar_volume: float = 0.0

        # History of completed raw bars (for ATR calc)
        self._history: List[Dict[str, Any]] = []
        self._warmup = atr_period + 1
        self._bars_emitted = 0

        # Optional quote diagnostics for live subscribe quality checks.
        # Enable with PAPER_DEBUG_QUOTES=1.
        self._debug_quotes = os.getenv("PAPER_DEBUG_QUOTES", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._quote_callbacks = 0
        self._quote_with_trade_price = 0
        self._quote_with_bidask = 0
        self._quote_dropped_no_trade_price = 0
        self._quote_last_diag_ts = monotonic()

    # ------------------------------------------------------------------
    # Live mode - async callback for RedisMarketDataClient.subscribe()
    # ------------------------------------------------------------------

    async def on_quote(self, instrument: str, quote: Any) -> None:
        """
        Async entry point for live market data (Redis Subscription).

        Execution Flow:
        1. Filters out invalid/zero prices.
        2. Detects session boundaries (skips ticks during breaks).
        3. Forwards valid ticks to the `_tick` accumulator.
        4. Periodically logs diagnostic quality metrics (if enabled).

        Args:
            instrument: Symbol string (e.g. 'HNXDS:VN30F2601').
            quote: QuoteSnapshot instance from Redis.
        """
        bid = getattr(quote, "bid_price_1", None)
        ask = getattr(quote, "ask_price_1", None)
        price = quote.latest_matched_price

        self._quote_callbacks += 1
        if (bid is not None and bid > 0) or (ask is not None and ask > 0):
            self._quote_with_bidask += 1

        if price is None or price <= 0:
            self._quote_dropped_no_trade_price += 1
            self._log_quote_diagnostics(instrument)
            return

        self._quote_with_trade_price += 1

        # Derive timestamp
        # Always use local system time for live bar bucketing to prevent 
        # stale trade timestamps from delaying bar emission.
        dt = datetime.now()

        if not _in_session(dt.time()):
            return

        volume = float(quote.latest_matched_quantity or 0.0)
        self._tick(dt, price, volume)
        self._log_quote_diagnostics(instrument)

    def _log_quote_diagnostics(self, instrument: str) -> None:
        """Periodically log subscribe callback quality metrics when enabled."""
        if not self._debug_quotes:
            return

        now = monotonic()
        if (now - self._quote_last_diag_ts) <= 15 and self._quote_callbacks % 100 != 0:
            return

        self._quote_last_diag_ts = now
        logger.info(
            "Quote diag | %s | callbacks=%d trade_px=%d bidask=%d dropped_no_trade_px=%d",
            instrument,
            self._quote_callbacks,
            self._quote_with_trade_price,
            self._quote_with_bidask,
            self._quote_dropped_no_trade_price,
        )

    def _tick(self, dt: datetime, price: float, volume: float) -> None:
        """Process a single price tick and accumulate into the current bar."""
        bucket = _bar_bucket(dt, self.freq_minutes)

        if self._current_bucket is None:
            # First tick ever
            self._start_bar(bucket, price)
        elif bucket != self._current_bucket:
            # New bar started - emit the completed bar first
            self._emit_bar()
            self._start_bar(bucket, price)

        # Update bar OHLC
        if price > self._bar_high:
            self._bar_high = price
        if price < self._bar_low:
            self._bar_low = price
        self._bar_close = price
        self._bar_volume += volume

    def _start_bar(self, bucket: datetime, price: float) -> None:
        self._current_bucket = bucket
        self._bar_open = price
        self._bar_high = price
        self._bar_low = price
        self._bar_close = price
        self._bar_volume = 0.0

    def _emit_bar(self) -> None:
        """Finalise the current bar, add it to history, and call on_bar if warmed up."""
        if self._current_bucket is None:
            return

        raw = {
            "datetime": self._current_bucket,
            "open": self._bar_open,
            "high": self._bar_high,
            "low": self._bar_low,
            "close": self._bar_close,
            "volume": self._bar_volume,
        }
        self._history.append(raw)

        # Keep only as much history as needed (atr_period * 3 buffer)
        max_history = self._warmup * 3
        if len(self._history) > max_history:
            self._history = self._history[-max_history:]

        if len(self._history) < self._warmup:
            logger.debug("Warming up (%d/%d bars)…", len(self._history), self._warmup)
            return

        # Build DataFrame and add indicators
        df = pd.DataFrame(self._history)
        df = self._preprocessor.add_all_indicators(df, copy=True)

        # Take the last row as the bar dict
        bar = df.iloc[-1].to_dict()
        self._bars_emitted += 1

        if self.on_bar is not None:
            try:
                self.on_bar(bar)
            except Exception as exc:
                logger.error("on_bar callback raised: %s", exc, exc_info=True)

    def check_time(self) -> None:
        """
        Called periodically (e.g. every second).
        If the current system time crosses into a new bucket,
        close the current bar and start a new empty one, keeping the time aligned.
        """
        if self._current_bucket is None:
            return

        now = datetime.now()
        from datetime import timedelta

        if not _in_session(now.time()):
            # If we just crossed out of a session (e.g. exactly 11:30:00)
            expected_bucket = _bar_bucket(now, self.freq_minutes)
            if expected_bucket > self._current_bucket:
                self._emit_bar()
                self._current_bucket = None
            return

        expected_bucket = _bar_bucket(now, self.freq_minutes)
        
        while expected_bucket > self._current_bucket:
            self._emit_bar()
            next_bucket = self._current_bucket + timedelta(minutes=self.freq_minutes)
            
            if not _in_session(next_bucket.time()):
                self._current_bucket = None
                break
                
            self._start_bar(next_bucket, self._bar_close)

    # ------------------------------------------------------------------
    # Pre-load History (Cold Start Fix)
    # ------------------------------------------------------------------

    def preload_history(self, df: pd.DataFrame) -> None:
        """
        Cold-Start Fix: Prime the internal history buffer.

        This method injects historical bars directly into the provider's 
        history list *without* triggering the `on_bar` callback. This 
        prevents "cold starts" where the strategy would otherwise have to 
        wait for `atr_period` bars of live data before generating a signal.

        Args:
            df: Historical OHLC DataFrame (must contain 'datetime' column).
        """
        if df.empty:
            logger.info("BarProvider.preload_history(): empty DataFrame, no history loaded.")
            return

        required = {"datetime", "open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"History DataFrame missing columns: {missing}")

        for row in df.itertuples(index=False):
            dt: datetime = (
                row.datetime
                if isinstance(row.datetime, datetime)
                else pd.Timestamp(row.datetime).to_pydatetime()
            )
            
            # Use raw datetimes if they are already buckets, otherwise bucket them
            bucket = _bar_bucket(dt, self.freq_minutes)

            raw = {
                "datetime": bucket,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(getattr(row, "volume", 0) or 0),
            }
            self._history.append(raw)

        # Keep only as much history as needed
        max_history = self._warmup * 3
        if len(self._history) > max_history:
            self._history = self._history[-max_history:]

        logger.info("Preloaded %d historical bars for indicator warmup.", len(self._history))

    # ------------------------------------------------------------------
    # Sim mode - replay a historical OHLC DataFrame
    # ------------------------------------------------------------------

    async def replay(
        self,
        df: pd.DataFrame,
        speed: float = 0.0,
    ) -> None:
        """
        Replay a pre-loaded OHLC DataFrame bar-by-bar through the same pipeline.

        Args:
            df:     DataFrame with columns: datetime, open, high, low, close, volume.
                    Should already be filtered to trading hours. Indicators will be
                    recomputed bar by bar so the strategy sees the same warmup as live.
            speed:  Seconds to sleep between bars (0 = as fast as possible).
        """
        if df.empty:
            logger.warning("BarProvider.replay(): empty DataFrame, nothing to replay.")
            return

        required = {"datetime", "open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Sim DataFrame missing columns: {missing}")

        logger.info(
            "Starting sim replay: %d bars @ %s freq (speed=%.2fs/bar)",
            len(df),
            self.bar_freq,
            speed,
        )

        for row in df.itertuples(index=False):
            dt: datetime = (
                row.datetime
                if isinstance(row.datetime, datetime)
                else pd.Timestamp(row.datetime).to_pydatetime()
            )
            price_close = float(row.close)
            price_open = float(row.open)
            price_high = float(row.high)
            price_low = float(row.low)
            volume = float(getattr(row, "volume", 0) or 0)

            if not _in_session(dt.time()):
                continue

            bucket = _bar_bucket(dt, self.freq_minutes)

            # Directly insert this bar into history (don't re-aggregate by tick)
            raw = {
                "datetime": bucket,
                "open": price_open,
                "high": price_high,
                "low": price_low,
                "close": price_close,
                "volume": volume,
            }
            self._history.append(raw)
            max_history = self._warmup * 3
            if len(self._history) > max_history:
                self._history = self._history[-max_history:]

            if len(self._history) < self._warmup:
                logger.debug("Warming up (%d/%d)…", len(self._history), self._warmup)
                if speed > 0:
                    await asyncio.sleep(speed)
                continue

            df_hist = pd.DataFrame(self._history)
            df_hist = self._preprocessor.add_all_indicators(
                df_hist, copy=True
            )
            bar = df_hist.iloc[-1].to_dict()
            self._bars_emitted += 1

            if self.on_bar is not None:
                try:
                    self.on_bar(bar)
                except Exception as exc:
                    logger.error("on_bar callback raised: %s", exc, exc_info=True)

            if speed > 0:
                await asyncio.sleep(speed)

        logger.info("Sim replay complete. %d bars emitted.", self._bars_emitted)
