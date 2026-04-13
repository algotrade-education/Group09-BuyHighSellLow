"""Real-time bar aggregation for paper trading.

Converts tick-level price updates into OHLCV bars with deterministic bucket timestamps.
Supports both live trading (Redis ticks) and simulation (historical replay).

Key features:
- Deterministic bucket timestamps (floor to bar frequency boundary)
- Session time filtering (morning/afternoon trading sessions)
- Data quality assessment and DB bar merging for sparse tick periods
- Indicator pipeline integration with warmup period
- Quote diagnostics for monitoring tick quality

Live mode: on_quote() receives ticks from Redis, bars emitted at bucket boundaries
Sim mode: replay() pushes historical bars through the indicator pipeline
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from time import monotonic
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.data.pipeline import DataPipeline
from src.paper.data_quality import BarState, DataQualityConfig, maybe_merge_db_bar

if TYPE_CHECKING:
    from src.engine.session.base import SessionManager

logger = logging.getLogger(__name__)


def _floor_to_bucket(dt: datetime, freq_minutes: int) -> datetime:
    """Floor a datetime to the nearest bucket boundary.

    Args:
        dt: Input datetime to floor.
        freq_minutes: Bucket frequency in minutes (e.g. 5 for 5-minute bars).

    Returns:
        Floored datetime representing the bucket start time.
    """
    total_minutes = dt.hour * 60 + dt.minute
    floored_minutes = (total_minutes // freq_minutes) * freq_minutes
    return dt.replace(
        hour=floored_minutes // 60,
        minute=floored_minutes % 60,
        second=0,
        microsecond=0,
    )


class BarAggregator:
    """Aggregates ticks into OHLC bars with deterministic bucket timestamps.

    Supports two operating modes:
    - Live: on_quote() receives ticks from Redis, bars emitted at bucket boundaries
    - Sim: replay() pushes historical bars through the indicator pipeline

    Includes data quality assessment and automatic DB bar merging when live
    tick coverage is insufficient (sparse ticks, large gaps, late starts/ends).
    """

    def __init__(
        self,
        freq_minutes: int,
        atr_period: int,
        fallback_bar_provider: Callable | None,
        runtime_config: dict,
        session_manager: SessionManager,
    ) -> None:
        """Initialize the bar aggregator.

        Args:
            freq_minutes: Bar frequency in minutes (e.g. 5 for 5-minute bars).
            atr_period: ATR period for indicator calculation and warmup.
            fallback_bar_provider: Optional callable(bucket_start) -> dict for DB bar fallback.
            runtime_config: Configuration dict with keys:
                - pipeline: DataPipeline instance for indicator computation
                - stale_trade_seconds: Max gap seconds before triggering DB merge
                - min_live_updates: Minimum live ticks required per bar
                - debug_quotes: Enable quote diagnostics logging
            session_manager: SessionManager instance for trading hours validation (required).
        """
        self.freq_minutes = freq_minutes
        self.atr_period = atr_period
        self._fallback_bar_provider = fallback_bar_provider
        self._runtime_config = runtime_config
        self._session_manager = session_manager

        # Downstream callback registered via set_on_bar()
        self._on_bar: Callable[[dict], None] | None = None

        # Current bucket start time
        self._current_bucket: datetime | None = None

        # OHLC accumulators
        self._open: float | None = None
        self._high: float | None = None
        self._low: float | None = None
        self._close: float | None = None
        self._volume: float = 0.0

        # Trade metadata for quality assessment
        self._bar_first_trade_ts: datetime | None = None
        self._bar_last_trade_ts: datetime | None = None
        self._bar_prev_trade_ts: datetime | None = None
        self._trade_count: int = 0
        self._has_live_trade: bool = False
        self._max_gap_seconds: float = 0.0

        # History buffer for indicator calculation (list of bar dicts)
        self._history: list[dict] = []
        self._bars_emitted: int = 0

        self._pipeline: DataPipeline | None = runtime_config.get("pipeline")
        self._warmup = atr_period + 1

        default_stale = max(5, int(freq_minutes * 60 * 0.1))
        self._stale_trade_seconds = float(runtime_config.get("stale_trade_seconds", default_stale))
        self._min_live_updates = int(runtime_config.get("min_live_updates", 2))
        self._preclose_fetch_seconds = float(runtime_config.get("preclose_fetch_seconds", 30.0))
        self._bar_db_merged: bool = False  # prevents duplicate preclose merges

        self._debug_quotes = bool(runtime_config.get("debug_quotes", False))
        self._quote_callbacks = 0
        self._quote_with_price = 0
        self._quote_no_price = 0
        self._diag_last_ts = monotonic()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    # --- Live Mode ---

    def check_time(self) -> None:
        """Check wall clock for bucket rollover and emit bar if needed.

        Called every second by the polling loop. Forces a bar emit when the
        wall clock has crossed into a new bucket and there is accumulated data.
        Also triggers a preclose DB merge when within preclose_fetch_seconds of
        bucket end, matching the old bar_provider behaviour.
        """
        now = datetime.now()
        current_wall_bucket = _floor_to_bucket(now, self.freq_minutes)

        if self._current_bucket is not None:
            # Preclose DB fetch: merge DB bar before bucket closes to fill gaps
            bucket_end = self._current_bucket + timedelta(minutes=self.freq_minutes)
            seconds_to_close = max(0.0, (bucket_end - now).total_seconds())
            if (
                seconds_to_close <= self._preclose_fetch_seconds
                and not self._bar_db_merged
                and self._fallback_bar_provider is not None
            ):
                bar_state, dq_config, reference_time = self._build_quality_state()
                from src.paper.data_quality import get_quality_reasons, merge_db_bar

                reasons = get_quality_reasons(bar_state, now, dq_config)
                if reasons:
                    try:
                        db_bar = self._fallback_bar_provider(self._current_bucket)
                        if db_bar:
                            current = self._build_bar_dict()
                            merged = merge_db_bar(current, db_bar, reasons)
                            self._open = merged["open"]
                            self._high = merged["high"]
                            self._low = merged["low"]
                            self._close = merged["close"]
                            self._volume = merged["volume"]
                            self._bar_db_merged = True
                            logger.warning(
                                "DB bar preclose merge for %s | reasons=%s | live_updates=%d",
                                self._current_bucket.strftime("%H:%M"),
                                ",".join(reasons),
                                self._trade_count,
                            )
                    except Exception:
                        logger.warning(
                            "Preclose DB fetch failed for %s", self._current_bucket, exc_info=True
                        )

        if (
            self._current_bucket is not None
            and current_wall_bucket != self._current_bucket
            and self._has_live_trade
        ):
            self._emit_bar()

    async def on_quote(self, instrument: str, quote: Any) -> None:
        """Process incoming Redis quote snapshot.

        Filters invalid prices, checks session times, and forwards valid ticks
        to the accumulator. Includes optional diagnostics logging.

        Args:
            instrument: Symbol string (e.g. 'HNXDS:VN30F2601').
            quote: QuoteSnapshot instance from Redis.
        """
        price = getattr(quote, "latest_matched_price", None)
        self._quote_callbacks += 1

        if price is None or price <= 0:
            self._quote_no_price += 1
            self._maybe_log_diagnostics(instrument)
            return

        self._quote_with_price += 1
        dt = datetime.now()

        # Filter out-of-session timestamps using SessionManager
        if not self._session_manager.is_trading_hours(dt):
            return

        volume = float(getattr(quote, "latest_matched_quantity", 0.0) or 0.0)
        self.on_tick(dt, price, volume)
        self._maybe_log_diagnostics(instrument)

    # --- Tick accumulation ---

    def on_tick(self, dt: datetime, price: float, volume: float) -> None:
        """Accumulate a tick into the current OHLC bucket.

        If the tick belongs to a new bucket, emits the previous bucket first.

        Args:
            dt: Tick timestamp.
            price: Tick price.
            volume: Tick volume.
        """
        bucket = _floor_to_bucket(dt, self.freq_minutes)

        if self._current_bucket is not None and bucket != self._current_bucket:
            self._emit_bar()

        if self._current_bucket is None or bucket != self._current_bucket:
            self.start_bar(bucket, price, volume, dt)
            return

        if self._high is None or price > self._high:
            self._high = price
        if self._low is None or price < self._low:
            self._low = price
        self._close = price
        self._volume += volume

        if self._bar_prev_trade_ts is not None:
            gap = (dt - self._bar_prev_trade_ts).total_seconds()
            self._max_gap_seconds = max(self._max_gap_seconds, gap)

        self._bar_prev_trade_ts = dt
        self._bar_last_trade_ts = dt
        self._trade_count += 1
        self._has_live_trade = True

    def start_bar(self, bucket: datetime, price: float, volume: float, dt: datetime) -> None:
        """Initialize a new bar bucket with first tick data.

        Args:
            bucket: Bucket start datetime (floored to frequency boundary).
            price: First tick price.
            volume: First tick volume.
            dt: First tick timestamp.
        """
        self._current_bucket = bucket
        self._open = price
        self._high = price
        self._low = price
        self._close = price
        self._volume = volume
        self._bar_first_trade_ts = dt
        self._bar_last_trade_ts = dt
        self._bar_prev_trade_ts = dt
        self._trade_count = 1
        self._has_live_trade = True
        self._max_gap_seconds = 0.0

    # --- Bar emission ---

    def set_on_bar(self, callback: Callable[[dict], None]) -> None:
        """Register the downstream callback invoked on each completed bar."""
        self._on_bar = callback

    def _emit_bar(self) -> None:
        """Emit the current accumulated bar and reset accumulators.

        Workflow:
        1. Build bar dict from accumulators
        2. Assess quality and merge with DB bar if needed
        3. Append to history buffer and trim
        4. Run indicator pipeline if past warmup
        5. Invoke downstream callback

        Quality assessment checks for:
        - Sparse tick coverage (too few live updates)
        - Large gaps between ticks (stale data)
        - Late bar start/end (session boundary issues)

        If quality issues detected, merges with DB bar to fill gaps.
        """
        if self._current_bucket is None or not self._has_live_trade:
            self._reset_accumulators()
            return

        bar = self._build_bar_dict()

        # Assess bar quality and merge with DB if needed
        # This prevents indicator corruption from sparse/gappy tick data
        bar_state, dq_config, reference_time = self._build_quality_state()
        bar = maybe_merge_db_bar(
            bar, bar_state, reference_time, dq_config, self._fallback_bar_provider
        )

        # Log bar formation details
        logger.info(
            "Bar formed | %s | O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f | ticks=%d gap=%.0fs | %s-%s",
            bar["datetime"].strftime("%Y-%m-%d %H:%M")
            if isinstance(bar.get("datetime"), datetime)
            else bar.get("datetime"),
            bar.get("open", 0.0),
            bar.get("high", 0.0),
            bar.get("low", 0.0),
            bar.get("close", 0.0),
            bar.get("volume", 0.0),
            self._trade_count,
            self._max_gap_seconds,
            self._bar_first_trade_ts.strftime("%H:%M:%S") if self._bar_first_trade_ts else "N/A",
            self._bar_last_trade_ts.strftime("%H:%M:%S") if self._bar_last_trade_ts else "N/A",
        )

        self._history.append(bar)
        self._trim_history()

        if len(self._history) < self._warmup:
            logger.debug("Warming up: %d/%d", len(self._history), self._warmup)
            self._reset_accumulators()
            if self._pipeline is None:
                self._invoke_callback(bar)
            return

        enriched_bar = self._run_pipeline_on_history()
        if enriched_bar is not None:
            bar = enriched_bar

        self._reset_accumulators()

        self._invoke_callback(bar)

    # --- Indicator enrichment ---

    # --- Preload (warmup) ---

    def preload_history(self, df: pd.DataFrame) -> None:
        """Preload historical bars into the history buffer for indicator warmup.

        Skips rows where high < low with a warning. Does not invoke on_bar callbacks.

        Args:
            df: Historical OHLC DataFrame with datetime, open, high, low, close, volume.
        """
        if df.empty:
            logger.info("preload_history: empty DataFrame, no history loaded.")
            return

        required = {"datetime", "open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"History DataFrame missing columns: {missing}")

        count = 0
        for _, row in df.iterrows():
            if row["high"] < row["low"]:
                logger.warning(
                    "preload_history: skipping row with high=%s < low=%s at %s",
                    row["high"],
                    row["low"],
                    row.get("datetime", "unknown"),
                )
                continue

            bar_dict = row.to_dict()
            # Ensure datetime is bucketed
            dt = bar_dict["datetime"]
            if isinstance(dt, pd.Timestamp):
                dt = dt.to_pydatetime()
            bar_dict["datetime"] = _floor_to_bucket(dt, self.freq_minutes)

            self._history.append(bar_dict)
            count += 1

        self._trim_history()
        logger.info("Preloaded %d historical bars for indicator warmup.", count)

    def seed_current_live_bar(self, bar_dict: dict, validate_bucket: bool = True) -> None:
        """Seed the current live bar from an incomplete DB bar.

        Uses the bar's bucket datetime (not datetime.now()) for trade timestamps
        to prevent spurious DB merge triggers during quality assessment.

        Args:
            bar_dict: Dictionary containing datetime, open, high, low, close, volume.
            validate_bucket: If True, validates that bar bucket matches current time bucket.
                           Set to False for testing or when seeding historical data.
        """
        if not bar_dict or "datetime" not in bar_dict:
            return

        dt = bar_dict["datetime"]
        if not isinstance(dt, datetime):
            dt = pd.Timestamp(dt).to_pydatetime()

        bar_bucket = _floor_to_bucket(dt, self.freq_minutes)

        if validate_bucket:
            now = datetime.now()
            expected_bucket = _floor_to_bucket(now, self.freq_minutes)
            if expected_bucket != bar_bucket:
                logger.warning(
                    "Skipping seed: bucket %s ≠ current bucket %s",
                    bar_bucket.strftime("%H:%M"),
                    expected_bucket.strftime("%H:%M"),
                )
                return

        bucket_dt = bar_bucket
        self._current_bucket = bucket_dt
        self._open = float(bar_dict.get("open", 0.0))
        self._high = float(bar_dict.get("high", 0.0))
        self._low = float(bar_dict.get("low", 0.0))
        self._close = float(bar_dict.get("close", 0.0))
        self._volume = float(bar_dict.get("volume", 0.0))
        self._bar_first_trade_ts = bucket_dt
        self._bar_last_trade_ts = bucket_dt
        self._bar_prev_trade_ts = bucket_dt
        self._trade_count = max(1, int(bar_dict.get("trade_count", 1) or 1))
        self._has_live_trade = True
        self._max_gap_seconds = 0.0

        logger.info(
            "Seeded bar for %s: O=%.1f H=%.1f L=%.1f C=%.1f V=%.0f",
            bucket_dt.strftime("%H:%M"),
            self._open,
            self._high,
            self._low,
            self._close,
            self._volume,
        )

    # --- Sim Mode ---

    async def replay(self, df: pd.DataFrame, speed: float = 0.0) -> None:
        """Replay historical OHLCV DataFrame through the indicator pipeline.

        Each bar is individually enriched with indicators computed on the
        accumulated history, mirroring live indicator behavior exactly.

        Args:
            df: Historical OHLCV DataFrame with datetime, open, high, low, close, volume.
            speed: Seconds to sleep between bars (0 = as fast as possible).
        """
        if df.empty:
            logger.warning("replay(): empty DataFrame.")
            return

        required = {"datetime", "open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Sim DataFrame missing columns: {missing}")

        logger.info("Sim replay: %d bars @ %dmin freq", len(df), self.freq_minutes)

        for row in df.itertuples(index=False):
            row_dt = row.datetime
            dt: datetime = (
                row_dt
                if isinstance(row_dt, datetime)
                else pd.Timestamp(str(row_dt)).to_pydatetime()
            )

            # Filter out-of-session timestamps using SessionManager
            if not self._session_manager.is_trading_hours(dt):
                continue

            open_val = float(str(row.open)) if hasattr(row, "open") else 0.0
            high_val = float(str(row.high)) if hasattr(row, "high") else 0.0
            low_val = float(str(row.low)) if hasattr(row, "low") else 0.0
            close_val = float(str(row.close)) if hasattr(row, "close") else 0.0
            volume_val = float(str(getattr(row, "volume", 0.0) or 0.0))

            raw = {
                "datetime": _floor_to_bucket(dt, self.freq_minutes),
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "close": close_val,
                "volume": volume_val,
            }

            if high_val < low_val:
                logger.warning("Skipping invalid bar (high < low): %s", raw["datetime"])
                continue

            self._history.append(raw)
            self._trim_history()

            if len(self._history) < self._warmup:
                if speed > 0:
                    await asyncio.sleep(speed)
                continue

            enriched = self._run_pipeline_on_history()
            if enriched and self._on_bar:
                self._invoke_callback(enriched)

            if speed > 0:
                await asyncio.sleep(speed)

        logger.info("Sim replay complete: %d bars emitted.", self._bars_emitted)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim_history(self) -> None:
        """Trim history buffer to max size (warmup * 5).

        Keep 5x warmup period for indicator stability. For example, a 14-period ATR
        needs approximately 70 bars (5 * 14) to produce reliable values with proper
        smoothing. This prevents memory growth while maintaining indicator accuracy.
        """
        max_hist = self._warmup * 5
        if len(self._history) > max_hist:
            self._history = self._history[-max_hist:]

    def _build_bar_dict(self) -> dict:
        """Build bar dict from current accumulators."""
        return {
            "datetime": self._current_bucket,
            "open": self._open,
            "high": self._high,
            "low": self._low,
            "close": self._close,
            "volume": self._volume,
        }

    def _build_quality_state(self) -> tuple[BarState, DataQualityConfig, datetime]:
        """Build quality assessment inputs from current bar state.

        Returns:
            Tuple of (bar_state, config, reference_time) for quality assessment.
        """
        if self._current_bucket is None:
            raise ValueError("Cannot build quality state without current bucket")

        dq_config = DataQualityConfig(
            stale_trade_seconds=self._stale_trade_seconds,
            min_live_updates=self._min_live_updates,
            freq_minutes=self.freq_minutes,
        )
        bar_state = BarState(
            has_live_trade=self._has_live_trade,
            trade_count=self._trade_count,
            first_trade_ts=self._bar_first_trade_ts,
            last_trade_ts=self._bar_last_trade_ts,
            max_gap_seconds=self._max_gap_seconds,
            bucket_start=self._current_bucket,
        )
        reference_time = self._current_bucket + timedelta(minutes=self.freq_minutes)
        return bar_state, dq_config, reference_time

    def _run_pipeline_on_history(self) -> dict | None:
        """Run indicator pipeline on history buffer and return enriched latest bar.

        Returns:
            Enriched bar dict with indicators, or None if pipeline fails.
        """
        if self._pipeline is None:
            return None

        try:
            history_df = pd.DataFrame(self._history)
            history_df = self._pipeline.run(history_df)
            self._history = history_df.to_dict(orient="records")
            return self._history[-1]
        except Exception:
            logger.exception("Indicator pipeline failed for bar %s", self._current_bucket)
            return None

    def _invoke_callback(self, bar: dict) -> None:
        """Invoke on_bar callback with error handling."""
        self._bars_emitted += 1
        if self._on_bar is not None:
            try:
                self._on_bar(bar)
            except Exception as e:
                logger.error("on_bar callback error: %s", e, exc_info=True)

    def _reset_accumulators(self) -> None:
        """Reset all per-bucket accumulators after emitting a bar."""
        self._current_bucket = None
        self._open = None
        self._high = None
        self._low = None
        self._close = None
        self._volume = 0.0
        self._bar_first_trade_ts = None
        self._bar_last_trade_ts = None
        self._bar_prev_trade_ts = None
        self._trade_count = 0
        self._has_live_trade = False
        self._max_gap_seconds = 0.0
        self._bar_db_merged = False

    def _maybe_log_diagnostics(self, instrument: str) -> None:
        """Periodically log quote callback quality metrics when enabled."""
        if not self._debug_quotes:
            return

        now = monotonic()
        if now - self._diag_last_ts < 60:
            return

        self._diag_last_ts = now
        total = self._quote_callbacks
        logger.debug(
            "Quote diagnostics [%s]: total=%d with_price=%d no_price=%d",
            instrument,
            total,
            self._quote_with_price,
            self._quote_no_price,
        )
