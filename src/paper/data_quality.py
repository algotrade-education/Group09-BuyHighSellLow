"""
Bar quality assessment and database merge logic.

Pure-function module for evaluating live bar quality and merging with database
fallback bars when live tick coverage is insufficient. All functions are stateless
with no side effects or I/O operations.

Quality assessment criteria:
- No live trades received during the bar period
- Too few tick updates (below min_live_updates threshold)
- Large gaps between ticks (exceeds stale_trade_seconds)
- Late start (first tick arrives too late after bucket start)
- Early end (last tick arrives too early before bucket end)

Merge strategy:
- No live trades: Replace all OHLC with DB values
- Start gap: Replace open with DB open
- End gap: Replace close with DB close
- Any quality issue: high = max(live, DB), low = min(live, DB), volume = max(live, DB)

Typical usage:
    bar_state = BarState(has_live_trade=True, trade_count=3, ...)
    config = DataQualityConfig(stale_trade_seconds=60.0, min_live_updates=2, freq_minutes=5)
    reasons = get_quality_reasons(bar_state, reference_time=bucket_end, config=config)
    if reasons:
        merged = maybe_merge_db_bar(bar, bar_state, bucket_end, config, fallback_provider)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BarState:
    """Snapshot of live-trade metadata for a single bar bucket.

    Attributes:
        has_live_trade: Whether at least one live tick was received in this bucket.
        trade_count: Total number of ticks received in this bucket.
        first_trade_ts: Timestamp of the first tick in the bucket, or None if no ticks.
        last_trade_ts: Timestamp of the last tick in the bucket, or None if no ticks.
        max_gap_seconds: Largest gap (in seconds) between consecutive ticks in the bucket.
        bucket_start: The bucket's start datetime (floor of the bar period).
    """

    has_live_trade: bool
    trade_count: int
    first_trade_ts: datetime | None
    last_trade_ts: datetime | None
    max_gap_seconds: float
    bucket_start: datetime


@dataclass
class DataQualityConfig:
    """Quality assessment thresholds for bar evaluation.

    Attributes:
        stale_trade_seconds: Gap duration (seconds) above which gaps trigger DB merge.
        min_live_updates: Minimum number of live ticks required per bar.
        freq_minutes: Bar frequency in minutes (used to compute bucket end time).
    """

    stale_trade_seconds: float
    min_live_updates: int
    freq_minutes: int


def get_quality_reasons(
    bar_state: BarState,
    reference_time: datetime,
    config: DataQualityConfig,
) -> list[str]:
    """Evaluate bar quality and return all failure reasons.

    Short-circuits with ["no_live_trade"] when no live tick was received.
    Otherwise evaluates all conditions independently so multiple reasons can
    be returned simultaneously.

    Possible reasons:
    - "no_live_trade": No live tick received
    - "too_few_updates": Tick count below min_live_updates
    - "large_internal_gap": Largest intra-bucket gap exceeds stale_trade_seconds
    - "start_gap": First tick arrived too late after bucket start
    - "end_gap": Last tick arrived too early before bucket end

    Args:
        bar_state: Live-trade metadata snapshot for the bucket.
        reference_time: Bucket end time for end-gap calculation.
        config: Quality thresholds.

    Returns:
        List of reason strings (empty if bar passes all checks).
    """
    if not bar_state.has_live_trade:
        return ["no_live_trade"]

    reasons: list[str] = []

    if bar_state.trade_count < config.min_live_updates:
        reasons.append("too_few_updates")

    if bar_state.max_gap_seconds > config.stale_trade_seconds:
        reasons.append("large_internal_gap")

    if bar_state.first_trade_ts is not None:
        start_gap = (bar_state.first_trade_ts - bar_state.bucket_start).total_seconds()
        if start_gap > config.stale_trade_seconds:
            reasons.append("start_gap")

    if bar_state.last_trade_ts is not None:
        end_gap = (reference_time - bar_state.last_trade_ts).total_seconds()
        if end_gap > config.stale_trade_seconds:
            reasons.append("end_gap")

    return reasons


def merge_db_bar(
    live_bar: dict,
    db_bar: dict,
    reasons: list[str],
) -> dict:
    """Merge DB bar data into live bar based on quality reasons.

    Pure function with deterministic merge logic based on quality reasons.

    Merge strategy:
    - No reasons: Return live bar unchanged
    - "no_live_trade": Replace all OHLC with DB values
    - "start_gap": Replace open with DB open
    - "end_gap": Replace close with DB close
    - Any quality issue: high = max(live, DB), low = min(live, DB), volume = max(live, DB)

    Args:
        live_bar: Current bar dict with keys: open, high, low, close, volume.
        db_bar: Fallback bar dict from database with same keys.
        reasons: List of quality reason strings from get_quality_reasons().

    Returns:
        Merged bar dict with improved quality data.
    """
    if not reasons:
        return dict(live_bar)

    merged = dict(live_bar)

    if "no_live_trade" in reasons:
        merged["open"] = db_bar.get("open", live_bar["open"])
        merged["high"] = db_bar.get("high", live_bar["high"])
        merged["low"] = db_bar.get("low", live_bar["low"])
        merged["close"] = db_bar.get("close", live_bar["close"])
    else:
        if "start_gap" in reasons:
            merged["open"] = db_bar.get("open", live_bar["open"])
        if "end_gap" in reasons:
            merged["close"] = db_bar.get("close", live_bar["close"])

        merged["high"] = max(live_bar["high"], db_bar.get("high", live_bar["high"]))
        merged["low"] = min(live_bar["low"], db_bar.get("low", live_bar["low"]))

    merged["volume"] = max(live_bar["volume"], db_bar.get("volume", live_bar["volume"]))

    return merged


def maybe_merge_db_bar(
    bar: dict,
    bar_state: BarState,
    reference_time: datetime,
    config: DataQualityConfig,
    fallback_bar_provider: Callable | None,
) -> dict:
    """Orchestrate DB bar merge decision and execution.

    Encapsulates the full merge workflow:
    1. Assess bar quality using get_quality_reasons()
    2. If quality issues exist, fetch DB fallback bar
    3. Merge DB data using merge_db_bar()
    4. Log merge operations

    Args:
        bar: Current bar dict to potentially merge with DB data.
        bar_state: Live-trade metadata snapshot for quality assessment.
        reference_time: Bucket end time for end-gap calculation.
        config: Quality thresholds configuration.
        fallback_bar_provider: Optional callable(bucket_start) -> dict that fetches DB bar.

    Returns:
        Merged bar dict (or original if no merge needed/possible).
    """
    if fallback_bar_provider is None or bar_state.bucket_start is None:
        return bar

    reasons = get_quality_reasons(bar_state, reference_time, config)
    if not reasons:
        return bar

    try:
        db_bar = fallback_bar_provider(bar_state.bucket_start)
    except Exception:
        logger.warning(
            "DB fallback lookup failed for bucket %s", bar_state.bucket_start, exc_info=True
        )
        return bar

    if db_bar is None:
        logger.warning(
            "DB bar merge unavailable for %s | reasons=%s",
            bar_state.bucket_start.strftime("%H:%M"),
            ",".join(reasons),
        )
        return bar

    if db_bar.get("volume", 0) == 0:
        logger.warning(
            "DB fallback bar for %s has zero volume; merging OHLC anyway", bar_state.bucket_start
        )

    merged = merge_db_bar(live_bar=bar, db_bar=db_bar, reasons=reasons)

    logger.info(
        "DB bar merged for %s | reasons=%s | live_updates=%d max_gap=%.0fs",
        bar_state.bucket_start.strftime("%H:%M"),
        ",".join(reasons),
        bar_state.trade_count,
        bar_state.max_gap_seconds,
    )

    return merged
