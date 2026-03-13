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
from datetime import datetime, time, timedelta
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
    "1h": 60,
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
        fallback_bar_provider: Optional[
            Callable[[datetime], Optional[Dict[str, Any]]]
        ] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
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
        self._fallback_bar_provider = fallback_bar_provider

        self._preprocessor = Preprocessor(atr_period=atr_period)

        # Current open bar state
        self._current_bucket: Optional[datetime] = None
        self._bar_open: float = 0.0
        self._bar_high: float = 0.0
        self._bar_low: float = float("inf")
        self._bar_close: float = 0.0
        self._bar_volume: float = 0.0

        # Recovery and quality tracking for live bars
        self._bar_has_live_trade: bool = False
        self._bar_trade_count: int = 0
        self._bar_first_trade_ts: Optional[datetime] = None
        self._bar_last_trade_ts: Optional[datetime] = None
        self._bar_prev_trade_ts: Optional[datetime] = None
        self._bar_max_gap_seconds: float = 0.0
        self._bar_db_merged: bool = False

        runtime = runtime_config or {}
        default_stale_seconds = max(5, int(self.freq_minutes * 60 * 0.1))
        self._stale_trade_seconds = float(
            runtime.get("stale_trade_seconds", default_stale_seconds)
        )
        default_preclose_seconds = max(2, int(self._stale_trade_seconds))
        self._preclose_db_fetch_seconds = float(
            runtime.get("preclose_db_fetch_seconds", default_preclose_seconds)
        )
        self._min_live_updates = int(runtime.get("min_live_updates", 2))

        # History of completed raw bars (for ATR calc)
        self._history: List[Dict[str, Any]] = []
        self._warmup = atr_period + 1
        self._bars_emitted = 0

        # Optional quote diagnostics for live subscribe quality checks.
        # Enable with PAPER_DEBUG_QUOTES=1.
        self._debug_quotes = _as_bool(runtime.get("debug_quotes", False))
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

        # logger.info("Tick %s: %f %f %f", dt, bid, ask, price)
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

        # Recovery logic for live trade detection and gap tracking
        self._bar_has_live_trade = True
        self._bar_trade_count += 1

        # Get the first trade timestamp for gap calculations
        if self._bar_first_trade_ts is None:
            self._bar_first_trade_ts = dt

        # Update gap tracking with the previous trade timestamp
        if self._bar_prev_trade_ts is not None:
            gap_seconds = max(0.0, (dt - self._bar_prev_trade_ts).total_seconds())
            if gap_seconds > self._bar_max_gap_seconds:
                self._bar_max_gap_seconds = gap_seconds

        self._bar_prev_trade_ts = dt
        self._bar_last_trade_ts = dt

    def _start_bar(self, bucket: datetime, price: float) -> None:
        self._current_bucket = bucket
        self._bar_open = price
        self._bar_high = price
        self._bar_low = price
        self._bar_close = price
        self._bar_volume = 0.0

        self._bar_has_live_trade = False
        self._bar_trade_count = 0
        self._bar_first_trade_ts = None
        self._bar_last_trade_ts = None
        self._bar_prev_trade_ts = None
        self._bar_max_gap_seconds = 0.0
        self._bar_db_merged = False

    def _db_quality_reasons(self, reference_time: datetime) -> List[str]:
        """Return data quality reasons that require DB merge for current bucket."""
        if self._current_bucket is None:
            return []

        bucket_start = self._current_bucket
        bucket_end = bucket_start + timedelta(minutes=self.freq_minutes)
        check_time = min(reference_time, bucket_end)

        reasons: List[str] = []
        if not self._bar_has_live_trade or self._bar_trade_count == 0:
            reasons.append("no_live_trade")
            return reasons

        if self._bar_trade_count < self._min_live_updates:
            reasons.append("too_few_updates")

        if self._bar_max_gap_seconds >= self._stale_trade_seconds:
            reasons.append("large_internal_gap")

        if self._bar_first_trade_ts is not None:
            start_gap = max(
                0.0, (self._bar_first_trade_ts - bucket_start).total_seconds()
            )
            if start_gap >= self._stale_trade_seconds:
                reasons.append("start_gap")

        if self._bar_last_trade_ts is not None:
            end_gap = max(0.0, (check_time - self._bar_last_trade_ts).total_seconds())
            if end_gap >= self._stale_trade_seconds:
                reasons.append("end_gap")

        return reasons

    def _merge_db_bar_into_current(
        self,
        fallback: Dict[str, Any],
        reasons: List[str],
        *,
        final_emit: bool,
    ) -> None:
        """Merge DB bucket bar with current Redis-aggregated bar state."""
        db_open = float(fallback.get("open", self._bar_open))
        db_high = float(fallback.get("high", self._bar_high))
        db_low = float(fallback.get("low", self._bar_low))
        db_close = float(fallback.get("close", self._bar_close))
        db_volume = float(fallback.get("volume", self._bar_volume))

        if not self._bar_has_live_trade:
            self._bar_open = db_open
            self._bar_high = db_high
            self._bar_low = db_low
            self._bar_close = db_close
            self._bar_volume = db_volume
        else:
            if "start_gap" in reasons:
                self._bar_open = db_open

            self._bar_high = max(self._bar_high, db_high)
            self._bar_low = min(self._bar_low, db_low)

            if "end_gap" in reasons or final_emit:
                self._bar_close = db_close

            self._bar_volume = max(self._bar_volume, db_volume)

        # Set this flag to prevent multiple merges in the same bar when not final emitting
        self._bar_db_merged = True
        logger.warning(
            "DB bar merge %s for %s | rows=%d | reasons=%s | live_updates=%d max_gap=%.0fs",
            "final" if final_emit else "preclose",
            self._current_bucket.strftime("%Y-%m-%d %H:%M")
            if self._current_bucket
            else "-",
            fallback.get("rows", 0),
            ",".join(reasons) if reasons else "none",
            self._bar_trade_count,
            self._bar_max_gap_seconds,
        )

    def _maybe_merge_db_bar(self, *, now: datetime, final_emit: bool) -> bool:
        """Fetch DB bar and merge into current bar when quality rules indicate staleness."""
        if self._current_bucket is None or self._fallback_bar_provider is None:
            return False

        # Get data quality reasons that indicate the current bar may be stale or incomplete
        reasons = self._db_quality_reasons(now)
        if not reasons:
            return False
        if self._bar_db_merged and not final_emit:
            return False

        try:
            fallback = self._fallback_bar_provider(self._current_bucket)
        except Exception as exc:
            logger.warning(
                "Bar fallback lookup failed for %s: %s",
                self._current_bucket.strftime("%Y-%m-%d %H:%M"),
                exc,
            )
            return False

        if not fallback:
            logger.warning(
                "DB bar merge unavailable for %s | reasons=%s",
                self._current_bucket.strftime("%Y-%m-%d %H:%M"),
                ",".join(reasons),
            )
            return False

        self._merge_db_bar_into_current(fallback, reasons, final_emit=final_emit)
        return True

    def _emit_bar(self) -> None:
        """Finalise the current bar, add it to history, and call on_bar if warmed up."""
        if self._current_bucket is None:
            return

        self._maybe_merge_db_bar(now=datetime.now(), final_emit=True)

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

        if self._current_bucket is not None:
            # Check if we should pre-merge a DB bar due to staleness before the close of the current bucket
            bucket_end = self._current_bucket + timedelta(minutes=self.freq_minutes)
            seconds_to_close = max(0.0, (bucket_end - now).total_seconds())
            if seconds_to_close <= self._preclose_db_fetch_seconds:
                self._maybe_merge_db_bar(now=now, final_emit=False)

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
            logger.info(
                "BarProvider.preload_history(): empty DataFrame, no history loaded."
            )
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

        logger.info(
            "Preloaded %d historical bars for indicator warmup.", len(self._history)
        )

    def seed_current_live_bar(self, bar_dict: Dict[str, Any]) -> None:
        """
        Seeds the currently forming live bar with intraday data fetched from the DB.
        This prevents dropping the first partially complete bar of the session when
        starting the engine mid-day.

        Args:
            bar_dict: Dictionary containing datetime, open, high, low, close, volume
        """
        if not bar_dict or "datetime" not in bar_dict:
            return

        dt: datetime = (
            bar_dict["datetime"]
            if isinstance(bar_dict["datetime"], datetime)
            else pd.Timestamp(bar_dict["datetime"]).to_pydatetime()
        )

        # Verify the seeded bar belongs to the current actual time bucket
        now = datetime.now()
        expected_bucket = _bar_bucket(now, self.freq_minutes)
        bar_bucket = _bar_bucket(dt, self.freq_minutes)

        if expected_bucket != bar_bucket:
            logger.warning(
                "Skipping live bar seed: seeded bucket %s does not match current bucket %s",
                bar_bucket.strftime("%H:%M"),
                expected_bucket.strftime("%H:%M"),
            )
            return

        self._current_bucket = bar_bucket
        self._bar_open = float(bar_dict.get("open", 0.0))
        self._bar_high = float(bar_dict.get("high", 0.0))
        self._bar_low = float(bar_dict.get("low", 0.0))
        self._bar_close = float(bar_dict.get("close", 0.0))
        self._bar_volume = float(bar_dict.get("volume", 0.0))

        self._bar_has_live_trade = True
        now_ts = datetime.now()
        self._bar_trade_count = max(1, int(bar_dict.get("trade_count", 1) or 1))
        self._bar_first_trade_ts = now_ts
        self._bar_last_trade_ts = now_ts
        self._bar_prev_trade_ts = now_ts
        self._bar_max_gap_seconds = 0.0
        self._bar_db_merged = False

        logger.info(
            "Seeded incomplete live bar for %s: O=%.1f H=%.1f L=%.1f C=%.1f V=%.0f",
            bar_bucket.strftime("%H:%M"),
            self._bar_open,
            self._bar_high,
            self._bar_low,
            self._bar_close,
            self._bar_volume,
        )

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

        # Optimization: if the DataFrame already contains indicators (calculated during 
        # prepare_backtest_dataset), we should use them instead of re-calculating 
        # on an empty history, which would cause discrepancies due to EWM warmup.
        has_indicators = "atr" in df.columns or f"atr_{self._preprocessor.atr_period}" in df.columns
        if has_indicators:
            logger.info("Sim replay detecting pre-calculated indicators; skipping re-calculation.")

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
            if has_indicators:
                # Use the existing row columns as the bar
                bar = row._asdict()
            else:
                self._history.append(raw)
                max_history = self._warmup * 10 # Increased for better EWM stability
                if len(self._history) > max_history:
                    self._history = self._history[-max_history:]

                if len(self._history) < self._warmup:
                    logger.debug("Warming up (%d/%d)…", len(self._history), self._warmup)
                    if speed > 0:
                        await asyncio.sleep(speed)
                    continue

                df_hist = pd.DataFrame(self._history)
                df_hist = self._preprocessor.add_all_indicators(df_hist, copy=True)
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
